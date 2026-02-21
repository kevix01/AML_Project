import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gc

# 1. Setup Dispositivo e VRAM profonda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
gc.collect()


# =====================================================================
# 2. SISTEMA DI HOOKING DELL'ATTENZIONE (Cuore di DiffAttack)
# =====================================================================
class SelfAttentionHook:
    def __init__(self):
        self.reference_maps = []
        self.adversarial_maps = []
        self.is_reference = True
        self.hooks = []
        # Catturiamo solo layer specifici per non far esplodere i 6GB VRAM.
        # Nelle UNet di SD, l'attenzione a 32x32 e 16x16 (mid_block e up_blocks.1)
        # è sufficiente per preservare la struttura semantica dell'immagine.
        self.target_blocks = ["mid_block", "up_blocks.1", "up_blocks.2"]

    def register(self, unet):
        for name, module in unet.named_modules():
            # Filtriamo solo i layer di pura "Attention" nei blocchi scelti
            if isinstance(module, Attention) and any(b in name for b in self.target_blocks):
                # Il layer 'attn1' in SD1.5 è la Self-Attention (attn2 è Cross-Attention)
                if "attn1" in name:
                    self.hooks.append(module.register_forward_hook(self.hook_fn))

    def hook_fn(self, module, input, output):
        # Nelle versioni moderne di Diffusers, l'output dell'attention block
        # è il tensore già processato. Lo usiamo come "firma strutturale".
        attn_out = output[0] if isinstance(output, tuple) else output

        if self.is_reference:
            # Salviamo S_t(fix) senza gradienti (risparmio massiccio VRAM)
            self.reference_maps.append(attn_out.detach().clone())
        else:
            # Salviamo S_t con gradienti per il backprop
            self.adversarial_maps.append(attn_out)

    def compute_structure_loss(self):
        loss = 0.0
        # Calcola MSE tra mappe pulite e mappe corrotte
        for ref, adv in zip(self.reference_maps, self.adversarial_maps):
            # Usiamo float32 per evitare overflow durante la L2 norm
            loss += F.mse_loss(adv.float(), ref.float())
        return loss

    def clear(self):
        if self.is_reference:
            self.reference_maps.clear()
        else:
            self.adversarial_maps.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# =====================================================================
# 3. CARICAMENTO MODELLI
# =====================================================================
print("[*] Caricamento Stable Diffusion 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # Obbligatorio per i 6GB
    safety_checker=None
).to(device)

# --- TRUCCHI SALVA-VRAM ---
pipe.enable_xformers_memory_efficient_attention()  # Indispensabile per 512x512
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.unet.enable_gradient_checkpointing()  # Riduce del 60% la VRAM richiesta in backprop

# Blocca i pesi originali
for comp in [pipe.unet, pipe.vae, pipe.text_encoder]:
    for param in comp.parameters():
        param.requires_grad = False

print("[*] Caricamento YOLOv8 nano...")
yolo_model = YOLO("yolov8n.pt").model.to(device).eval().half()
for param in yolo_model.parameters():
    param.requires_grad = False


# =====================================================================
# 4. FUNZIONE PRINCIPALE: DIFFATTACK ORIGINALE ADATTATO
# =====================================================================
def run_diffattack_full_res(image_path, text_prompt, num_steps=40, alpha=5.0, lr=0.005):
    # NOTA: alpha molto più alto (5.0) e lr molto più basso (0.005)

    print("\n[*] Preparazione immagine 512x512...")
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    init_tensor = torch.from_numpy(np.array(init_image)).half().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    init_tensor = init_tensor.to(device)

    with torch.no_grad():
        clean_latents = pipe.vae.encode(init_tensor).latent_dist.sample()
        clean_latents = clean_latents * pipe.vae.config.scaling_factor

        text_inputs = pipe.tokenizer(text_prompt, padding="max_length", max_length=77, return_tensors="pt")
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    attn_hook = SelfAttentionHook()
    attn_hook.register(pipe.unet)

    # TIMESTEP ABBASSATO A 100: la U-Net non impazzisce ricevendo latenti puliti
    timestep = torch.tensor([100], device=device, dtype=torch.long)

    print("[*] Estrazione Self-Attention dall'immagine originale...")
    attn_hook.is_reference = True
    with torch.no_grad():
        _ = pipe.unet(clean_latents, timestep, encoder_hidden_states=prompt_embeds)
    attn_hook.is_reference = False

    delta_latent = torch.zeros_like(clean_latents, dtype=torch.float32, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([delta_latent], lr=lr)

    print("[*] Inizio Attacco Avversario (Impercettibile)...")
    for step in range(num_steps):
        optimizer.zero_grad()
        attn_hook.clear()

        # 1. Creiamo il latente avversario
        adv_latents = clean_latents + delta_latent.half()

        # 2. Decodifichiamo DIRETTAMENTE l'immagine avversaria per YOLO
        # (Niente sottrazioni con noise_pred, passiamo l'immagine esatta che salveremo)
        adv_image_dec = pipe.vae.decode(adv_latents / pipe.vae.config.scaling_factor)[0]
        adv_image_norm = (adv_image_dec / 2 + 0.5).clamp(0, 1)
        adv_image_yolo = F.interpolate(adv_image_norm, size=(640, 640), mode='bilinear')

        # 3. Calcolo L_task (Ingannare YOLO)
        yolo_preds = yolo_model(adv_image_yolo)
        confidence_scores = yolo_preds[0][:, 4:, :]
        l_task = torch.max(confidence_scores)

        # 4. Calcolo L_structure (Mantenere l'aspetto visivo tramite U-Net)
        # Facciamo un forward pass nella U-Net solo per innescare gli Hook e catturare le mappe
        _ = pipe.unet(adv_latents, timestep, encoder_hidden_states=prompt_embeds)
        l_structure = attn_hook.compute_structure_loss()

        # 5. Backpropagation
        total_loss = l_task + alpha * l_structure
        total_loss.backward()
        optimizer.step()

        # 6. CLIPPING MOLTO SEVERO: massimo 5% di deviazione nel latente
        with torch.no_grad():
            delta_latent.clamp_(-0.05, 0.05)

        torch.cuda.empty_cache()

        print(
            f"Step {step + 1}/{num_steps} | L_task YOLO: {l_task.item():.4f} | L_struct Attn: {l_structure.item():.4f}")

    attn_hook.remove()

    with torch.no_grad():
        final_latents = clean_latents + delta_latent.half()
        final_image = pipe.vae.decode(final_latents / pipe.vae.config.scaling_factor)[0]
        final_image = (final_image / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        final_image = (final_image * 255).astype(np.uint8)

    return Image.fromarray(final_image)


# =====================================================================
# ESECUZIONE TEST
# =====================================================================
if __name__ == "__main__":
    import urllib.request
    import os

    test_img = "test_512.jpg"
    if not os.path.exists(test_img):
        # Scarichiamo un'immagine quadrata a buona risoluzione
        url = "https://ultralytics.com/images/bus.jpg"
        urllib.request.urlretrieve(url, test_img)

    # YOLO originale
    yolo_ufficiale = YOLO("yolov8n.pt")
    res_orig = yolo_ufficiale(test_img)[0]
    res_orig.save("pre_attacco.jpg")
    print(f"\n[+] Oggetti rilevati prima: {len(res_orig.boxes)}")

    # ATTACCO (512x512, Full Attention Constraint)
    img_adv = run_diffattack_full_res(
        image_path=test_img,
        text_prompt="a bus with people",  # Adatta il prompt all'immagine
        num_steps=50,
        alpha=50.0,  # Aumenta alpha (es. 0.1) se l'immagine si distorce troppo
        lr=0.08
    )
    img_adv.save("adv_diffattack.png")

    # YOLO dopo l'attacco
    res_adv = yolo_ufficiale("adv_diffattack.png")[0]
    res_adv.save("post_attacco.jpg")
    print(f"\n[+] Oggetti rilevati dopo: {len(res_adv.boxes)}")

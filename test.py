import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gc
import os
import urllib.request

# =====================================================================
# 1. SETUP AMBIENTE E VRAM
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
gc.collect()

print(f"[*] Dispositivo: {device}")


# =====================================================================
# 2. GESTORE DUALE DI ATTENZIONE (Self + Cross Attention)
# =====================================================================
class DualAttentionHook:
    def __init__(self):
        self.sa_maps = []  # Self-Attention (Struttura Geometrica)
        self.ca_maps = []  # Cross-Attention (Semantica Testo-Immagine)
        self.hooks = []
        # Targettizziamo solo i layer a risoluzione media per risparmiare VRAM
        self.target_blocks = ["mid_block", "up_blocks.1", "up_blocks.2"]

    def register(self, unet):
        for name, module in unet.named_modules():
            if isinstance(module, Attention) and any(b in name for b in self.target_blocks):
                # In SD 1.5: 'attn1' = Self-Attention | 'attn2' = Cross-Attention
                if "attn1" in name:
                    self.hooks.append(module.register_forward_hook(self.hook_sa(name)))
                elif "attn2" in name:
                    self.hooks.append(module.register_forward_hook(self.hook_ca(name)))

    def hook_sa(self, name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.sa_maps.append(out)

        return hook

    def hook_ca(self, name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.ca_maps.append(out)

        return hook

    def clear(self):
        self.sa_maps.clear()
        self.ca_maps.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# =====================================================================
# 3. CARICAMENTO MODELLI
# =====================================================================
print("[*] Caricamento Stable Diffusion 1.5 in FP16...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# Ottimizzazioni obbligatorie
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.unet.enable_gradient_checkpointing()

# Congelamento Pesi
for comp in [pipe.unet, pipe.vae, pipe.text_encoder]:
    for param in comp.parameters():
        param.requires_grad = False

print("[*] Caricamento YOLOv8...")
yolo_model = YOLO("yolov8n.pt").model.to(device).eval().half()
for param in yolo_model.parameters():
    param.requires_grad = False


# =====================================================================
# 4. IMPLEMENTAZIONE DIFFATTACK FEDELE (MULTI-STEP)
# =====================================================================
def faithful_diffattack(image_path, text_prompt, num_steps=30, lr=0.02, alpha=2.0, beta=0.5):
    """
    alpha: Peso Self-Attention (Vincolo strutturale: minimizza la differenza)
    beta: Peso Cross-Attention (Distruzione semantica: massimizza la differenza)
    """
    # 1. PREPARAZIONE (Risoluzione a 256 per limitare i consumi)
    init_image = Image.open(image_path).convert("RGB").resize((256, 256))
    img_tensor = torch.from_numpy(np.array(init_image)).half().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    # Il rumore avversario è applicato direttamente sui PIXEL dell'immagine
    # Creiamo il tensore direttamente sulla GPU come tensore foglia
    delta_image = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)
    delta_image.requires_grad = True

    optimizer = torch.optim.Adam([delta_image], lr=lr)

    # 2. ENCODING DEL PROMPT
    with torch.no_grad():
        text_inputs = pipe.tokenizer(text_prompt, padding="max_length", max_length=77, return_tensors="pt")
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # I timestep per simulare la catena della diffusione (ridotti a 2 per la VRAM)
    t_samples = [150, 400]

    # 3. REFERENCE PASS: Estrazione Mappe Pulite a vari step di rumore
    print("[*] Estrazione mappe di attenzione originali...")
    clean_maps_sa = {}
    clean_maps_ca = {}
    attn_hook = DualAttentionHook()
    attn_hook.register(pipe.unet)

    with torch.no_grad():
        clean_img_gpu = img_tensor.to(device)
        clean_latents = pipe.vae.encode(clean_img_gpu).latent_dist.sample() * pipe.vae.config.scaling_factor

        for t in t_samples:
            attn_hook.clear()
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)

            # Creiamo l'immagine rumorosa originale allo step t
            noise = torch.randn_like(clean_latents)
            noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, t_tensor)

            # Forward pass per salvare la firma visiva
            _ = pipe.unet(noisy_latents, t_tensor, encoder_hidden_states=prompt_embeds)

            clean_maps_sa[t] = [m.detach().clone() for m in attn_hook.sa_maps]
            clean_maps_ca[t] = [m.detach().clone() for m in attn_hook.ca_maps]

    # 4. LOOP DI OTTIMIZZAZIONE
    print("[*] Inizio Attacco (Multi-Step con Cross-Attention)...")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Creiamo l'immagine avversaria nei pixel e la normalizziamo
        adv_image = img_tensor.to(device) + delta_image.half()
        adv_image_norm = (adv_image / 2 + 0.5).clamp(0, 1)

        # LOSS 1: Evasione YOLO
        adv_image_yolo = F.interpolate(adv_image_norm, size=(640, 640), mode='bilinear')
        yolo_preds = yolo_model(adv_image_yolo)
        # Troviamo l'ancora con confidenza massima e la minimizziamo
        l_task = torch.max(yolo_preds[0][:, 4:, :])

        # Codifichiamo l'immagine corrotta nel latente (passando per il VAE)
        adv_latents = pipe.vae.encode(adv_image).latent_dist.sample() * pipe.vae.config.scaling_factor

        l_sa_total = 0.0
        l_ca_total = 0.0

        # LOSS 2 & 3: Calcolo su molteplici step temporali
        for t in t_samples:
            attn_hook.clear()
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)

            noise = torch.randn_like(adv_latents)
            noisy_adv_latents = pipe.scheduler.add_noise(adv_latents, noise, t_tensor)

            # Forward nella U-Net per popolare attn_hook
            _ = pipe.unet(noisy_adv_latents, t_tensor, encoder_hidden_states=prompt_embeds)

            # Self-Attention: Vogliamo mantenere la struttura SIMILE (Minimizziamo MSE)
            for adv_sa, ref_sa in zip(attn_hook.sa_maps, clean_maps_sa[t]):
                l_sa_total += F.mse_loss(adv_sa.float(), ref_sa.float())

            # Cross-Attention: Vogliamo DISTRUGGERE l'ancoraggio semantico (Massimizziamo MSE -> Segno -)
            for adv_ca, ref_ca in zip(attn_hook.ca_maps, clean_maps_ca[t]):
                l_ca_total -= F.mse_loss(adv_ca.float(), ref_ca.float())  # Nota il MENO

        # Mediamo le loss temporali
        l_sa_total = l_sa_total / len(t_samples)
        l_ca_total = l_ca_total / len(t_samples)

        # Loss Combinata (L_Task + Struttura - Semantica)
        total_loss = l_task + (alpha * l_sa_total) + (beta * l_ca_total)

        # Backpropagation attraverso U-Net -> VAE Decoder -> Immagine RGB
        total_loss.backward()
        optimizer.step()

        # Vincolo rigoroso (L_infinity norm): le modifiche sui pixel non superano un epsilon limitato (es. 16/255)
        with torch.no_grad():
            delta_image.clamp_(-16 / 255, 16 / 255)

        torch.cuda.empty_cache()

        print(
            f"Step {step + 1}/{num_steps} | L_YOLO: {l_task.item():.4f} | L_SA: {l_sa_total.item():.4f} | L_CA (Negativa): {l_ca_total.item():.4f}")

    attn_hook.remove()

    # Generazione Output
    with torch.no_grad():
        final_image = img_tensor.to(device) + delta_image.half()
        final_image = (final_image / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        final_image = (final_image * 255).astype(np.uint8)

    return Image.fromarray(final_image)


# =====================================================================
# 5. BLOCCO PRINCIPALE DI ESECUZIONE (MAIN)
# =====================================================================
if __name__ == "__main__":
    # 1. Download immagine di test
    img_name = "test_fedele.jpg"
    if not os.path.exists(img_name):
        print("[*] Download immagine di test...")
        url = "https://ultralytics.com/images/zidane.jpg"
        urllib.request.urlretrieve(url, img_name)

    # Questo è il prompt da "distruggere" nella Cross-Attention
    prompt = "two men in suits"

    # 2. YOLO Ufficiale (Pre-Attacco)
    print("\n--- TEST PRE-ATTACCO ---")
    yolo_full = YOLO("yolov8n.pt")
    res_orig = yolo_full(img_name)
    res_orig[0].save("01_pre_attacco.jpg")
    print(f"[+] Oggetti trovati (Originale): {len(res_orig[0].boxes)}")

    # 3. Lancio DiffAttack (Fedele)
    print("\n--- AVVIO DIFFATTACK (Pixel Space + SA/CA Control) ---")
    adv_img = faithful_diffattack(
        image_path=img_name,
        text_prompt=prompt,
        num_steps=30,  # Iterazioni dell'ottimizzatore
        lr=0.03,  # Learning rate per i pixel
        alpha=2.0,  # Forza di mantenimento visivo
        beta=0.5  # Forza di distruzione concettuale (Cross-Attention)
    )

    adv_img.save("02_immagine_avversaria.png")
    print("[+] Immagine avversaria salvata come '02_immagine_avversaria.png'")

    # 4. YOLO Ufficiale (Post-Attacco)
    print("\n--- TEST POST-ATTACCO ---")
    res_adv = yolo_full("02_immagine_avversaria.png")
    res_adv[0].save("03_post_attacco.jpg")
    print(f"[+] Oggetti trovati (Avversaria): {len(res_adv[0].boxes)}")

    if len(res_adv[0].boxes) < len(res_orig[0].boxes):
        print("\n[!!!] ATTACCO COMPLETATO CON SUCCESSO! YOLO ha perso traccia degli oggetti.")
    else:
        print("\n[?] Attacco resistito da YOLO. Prova a incrementare lr a 0.05 o ridurre alpha a 1.0.")

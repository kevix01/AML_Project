import argparse
import os
import urllib.request
import torch
from ultralytics import YOLO
from attack_engine import AttackEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sd15", choices=["sd15", "sdxl", "tiny_sd", "flux"])
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token per modelli gated")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Carica modello di diffusione
    if args.model == "sd15":
        from diffusion_models.sd15_model import SD15Model
        from diffusion_hooks.sd15_hook import SD15AttentionHook
        diffusion_model = SD15Model(device, dtype=torch.float16)
        diffusion_model.load_pipeline()
        hook_class = SD15AttentionHook
        print("Stable Diffusion 1.5 caricato")
    elif args.model == "tiny_sd":
        from diffusion_models.tiny_sd_model import TinySDModel
        # Riutilizza l'hook di SD15 (stessa architettura)
        from diffusion_hooks.sd15_hook import SD15AttentionHook

        diffusion_model = TinySDModel(device, dtype=torch.float16)
        diffusion_model.load_pipeline()  # usa OFA-Sys/small-stable-diffusion-v0
        hook_class = SD15AttentionHook
        print("Tiny Stable Diffusion caricato")
    elif args.model == "flux":
        from diffusion_models.flux_model import Flux2KleinFP8Model
        from diffusion_hooks.flux_hook import Flux2KleinAttentionHook

        diffusion_model = Flux2KleinFP8Model(device, dtype=torch.float16,
                                             auto_download=True, token=args.hf_token,
                                             checkpoint_path="black-forest-labs/FLUX.2-klein-4b-fp8/flux-2-klein-4b-fp8.safetensors")
        diffusion_model.load_pipeline()
        hook_class = Flux2KleinAttentionHook
        print("Flux2-Klein caricato")
    else:  # sdxl
        from diffusion_models.sdxl_model import SDXLFP8Model
        from diffusion_hooks.sdxl_hook import SDXLAttentionHook
        diffusion_model = SDXLFP8Model(device, dtype=torch.float16)
        diffusion_model.load_pipeline()  # eventualmente passa model_id specifico
        hook_class = SDXLAttentionHook
        print("Stable Diffusion XL caricato")

    # Carica YOLO (identico)
    yolo_model = YOLO("yolov8n.pt").model.to(device).eval().half()
    for param in yolo_model.parameters():
        param.requires_grad = False

    engine = AttackEngine(diffusion_model, yolo_model, hook_class)

    # Download immagine di test
    img_name = "test_bus.png"
    if not os.path.exists(img_name):
        urllib.request.urlretrieve("https://ultralytics.com/images/zidane.jpg", img_name)

    # Valutazione pre-attacco
    yolo_full = YOLO("yolov8n.pt")
    res_orig = yolo_full(img_name)
    res_orig[0].save("01_pre_attacco.jpg")
    print(f"Oggetti originali: {len(res_orig[0].boxes)}")

    # Esegui attacco
    adv_img = engine.run(
        image_path=img_name,
        text_prompt="bus with people",
        num_steps=100,
        lr=0.01,
        alpha=10.0,
        beta=0.5,
        t_samples=[1, 5, 10, 15, 20, 30],
        image_size=512
    )
    adv_img.save("02_immagine_avversaria.png")

    # Valutazione post-attacco
    res_adv = yolo_full("02_immagine_avversaria.png")
    res_adv[0].save("03_post_attacco.jpg")
    print(f"Oggetti dopo attacco: {len(res_adv[0].boxes)}")
    if len(res_adv[0].boxes) < len(res_orig[0].boxes):
        print("ATTACCO RIUSCITO!")
    else:
        print("Attacco fallito.")
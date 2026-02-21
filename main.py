import os
import urllib.request
import torch
from ultralytics import YOLO
from diffusion_hooks.sd15_hook import SD15AttentionHook
from diffusion_models.sd15_model import SD15Model
from attack_engine import AttackEngine

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Carica modello di diffusione (SD1.5)
    diffusion_model = SD15Model(device, dtype=torch.float16)
    diffusion_model.load_pipeline()

    # Carica YOLO (come prima)
    yolo_model = YOLO("yolov8n.pt").model.to(device).eval().half()
    for param in yolo_model.parameters():
        param.requires_grad = False

    # Crea engine con hook specifico per SD1.5
    engine = AttackEngine(diffusion_model, yolo_model, SD15AttentionHook)

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
        alpha=5.0,
        beta=0.5,
        t_samples=[5, 10, 20, 40],
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
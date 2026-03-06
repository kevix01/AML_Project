import argparse
import torch
from ultralytics import YOLO
from attack_engine import AttackEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single adversarial attack test")
    parser.add_argument("--model", type=str, default="sd15",
                        choices=["sd15", "tiny_sd", "vega"],
                        help="Modello di diffusione da usare")
    parser.add_argument(
        "--yolo",
        type=str,
        default="yolov8n",
        choices=[
            "yolov8n",
            "yolov10n",
            "yolo11n",
            "yolo26n"
        ],
        help="Variante YOLO da usare (es. yolov8n, yolo11m, ...)",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate per l'attacco")
    parser.add_argument("--steps", type=int, default=100, help="Numero di step di ottimizzazione")
    parser.add_argument("--alpha", type=float, default=10.0, help="Peso per le mappe self-attention")
    parser.add_argument("--beta", type=float, default=0.5, help="Peso per le mappe cross-attention")
    parser.add_argument("--epsilon", type=float, default=16/255, help="Perturbazione massima L-inf")
    parser.add_argument("--t_samples", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30],
                        help="Timestep da campionare per le loss di attenzione")
    parser.add_argument("--image_size", type=int, default=512, help="Dimensione dell'immagine (larghezza=altezza)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Token Hugging Face per modelli con accesso riservato")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device su cui eseguire (cuda/cpu)")
    parser.add_argument("--image", type=str, default="bus.jpg",
                        help="Percorso dell'immagine di test")
    parser.add_argument("--prompt", type=str, default="bus with people",
                        help="Prompt testuale per l'attacco")
    args = parser.parse_args()

    yolo_weights = f"{args.yolo}.pt"
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # ---- Caricamento modello di diffusione ----
    if args.model == "sd15":
        from diffusion_models.sd15_model import SD15Model
        from diffusion_hooks.sd15_hook import SD15AttentionHook
        diffusion_model = SD15Model(device, dtype=torch.float16)
        diffusion_model.load_pipeline(hf_token=args.hf_token)
        hook_class = SD15AttentionHook
        print("Stable Diffusion 1.5 caricato")
    elif args.model == "tiny_sd":
        from diffusion_models.tiny_sd_model import TinySDModel
        from diffusion_hooks.sd15_hook import SD15AttentionHook
        diffusion_model = TinySDModel(device, dtype=torch.float16)
        diffusion_model.load_pipeline(hf_token=args.hf_token)
        hook_class = SD15AttentionHook
        print("Tiny Stable Diffusion caricato")
    elif args.model == "vega":
        from diffusion_models.vega_model import SegmindVegaModel
        from diffusion_hooks.vega_hook import SegmindVegaAttentionHook
        diffusion_model = SegmindVegaModel(device, dtype=torch.float16)
        diffusion_model.load_pipeline(hf_token=args.hf_token)
        hook_class = SegmindVegaAttentionHook
        print("Segmind-Vega caricato")
    else:
        raise ValueError(f"Modello di diffusione non valido: {args.model}")

    # ---- Caricamento YOLO per l'attacco ----
    yolo_model = YOLO(yolo_weights).model.to(device).eval().half()
    for param in yolo_model.parameters():
        param.requires_grad = False

    # Disabilita modalità NMS-free (se presente)
    yolo_head = yolo_model.model[-1]
    if hasattr(yolo_head, "end2end"):
        yolo_head.end2end = False
        print(f"Modalità NMS-free disabilitata su {yolo_weights}")
    else:
        print(f"Avviso: nessun attributo 'end2end' trovato in {yolo_weights}")

    print(f"YOLO caricato: {yolo_weights}")

    engine = AttackEngine(diffusion_model, yolo_model, hook_class)

    # ---- Valutazione pre-attacco ----
    yolo_full = YOLO(yolo_weights)
    res_orig = yolo_full(args.image)
    res_orig[0].save("01_pre_attacco.jpg")
    print(f"Oggetti originali: {len(res_orig[0].boxes)}")

    # ---- Esecuzione attacco ----
    print("Esecuzione attacco...")
    adv_img = engine.run(
        image_path=args.image,
        text_prompt=args.prompt,
        num_steps=args.steps,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        t_samples=args.t_samples,
        image_size=args.image_size
    )
    adv_img.save("02_immagine_avversaria.png")

    # ---- Valutazione post-attacco ----
    res_adv = yolo_full("02_immagine_avversaria.png")
    res_adv[0].save("03_post_attacco.jpg")
    print(f"Oggetti dopo attacco: {len(res_adv[0].boxes)}")
    if len(res_adv[0].boxes) < len(res_orig[0].boxes):
        print("ATTACCO RIUSCITO!")
    else:
        print("Attacco fallito.")
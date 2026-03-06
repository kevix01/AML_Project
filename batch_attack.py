import argparse
import os
import csv
import gc
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from attack_engine import AttackEngine

# ----------------------------------------------------------------------
# Parametri fissi per tutti gli attacchi (modificabili qui o da riga di comando)
# ----------------------------------------------------------------------
DEFAULT_LR = 0.01
DEFAULT_STEPS = 100
DEFAULT_ALPHA = 10.0
DEFAULT_BETA = 0.5
DEFAULT_EPSILON = 16 / 255
DEFAULT_T_SAMPLES = [5, 10, 15, 20, 25, 30]
DEFAULT_IMAGE_SIZE = 512
TEST_IMAGE = "bus.jpg"
TEXT_PROMPT = "bus with people"

# ----------------------------------------------------------------------
# Definizione delle varianti da testare
# ----------------------------------------------------------------------
DIFFUSION_MODELS = [
    "sd15",
    "tiny_sd",
    "vega"
]
YOLO_MODELS = [
    "yolov8n",
    # "yolov10n",
    "yolo11n",
    "yolo26n"
]


# ----------------------------------------------------------------------
# Mappa: modello diffusione -> (classe modello, classe hook)
# ----------------------------------------------------------------------
def get_diffusion_classes(model_name):
    if model_name == "sd15":
        from diffusion_models.sd15_model import SD15Model
        from diffusion_hooks.sd15_hook import SD15AttentionHook
        return SD15Model, SD15AttentionHook
    elif model_name == "tiny_sd":
        from diffusion_models.tiny_sd_model import TinySDModel
        from diffusion_hooks.sd15_hook import SD15AttentionHook
        return TinySDModel, SD15AttentionHook
    elif model_name == "vega":
        from diffusion_models.vega_model import SegmindVegaModel
        from diffusion_hooks.vega_hook import SegmindVegaAttentionHook
        return SegmindVegaModel, SegmindVegaAttentionHook
    else:
        raise ValueError(f"Modello di diffusione sconosciuto: {model_name}")


# ----------------------------------------------------------------------
# Funzione per disabilitare la modalità NMS‑free (end2end) sulla testa YOLO
# ----------------------------------------------------------------------
def disable_nms_free(yolo_model):
    head = yolo_model.model[-1]
    if hasattr(head, "end2end"):
        if head.end2end:
            head.end2end = False
            print("   [YOLO] Modalità NMS‑free disabilitata.")
    else:
        print("   [YOLO] Avviso: nessun attributo 'end2end' trovato, skip.")


# ----------------------------------------------------------------------
# Funzione per eseguire un singolo attacco in un contesto isolato
# ----------------------------------------------------------------------
def run_single_attack(diff_name, yolo_name, combo_dir, args):
    """
    Esegue un singolo attacco e salva i risultati su disco.
    Restituisce un dizionario con i metric e il percorso del file CSV delle loss.
    """
    yolo_weights = f"{yolo_name}.pt"

    print(f"\n--- YOLO: {yolo_name} ---")

    # Caricamento modello diffusione
    print("   Caricamento modello diffusione...")
    ModelClass, HookClass = get_diffusion_classes(diff_name)
    diffusion_model = ModelClass(device=args.device, dtype=torch.float16)
    diffusion_model.load_pipeline(hf_token=args.hf_token)
    print("   Modello diffusione caricato")

    # Caricamento YOLO per attack engine
    print("   Caricamento YOLO per attack...")
    yolo_model = YOLO(yolo_weights).model.to(args.device).eval().half()
    for param in yolo_model.parameters():
        param.requires_grad = False
    disable_nms_free(yolo_model)
    print(f"   Modello YOLO caricato: {yolo_weights}")

    # AttackEngine
    engine = AttackEngine(diffusion_model, yolo_model, HookClass)

    # Pre-attacco con YOLO full
    yolo_full = YOLO(yolo_weights)
    res_orig = yolo_full(TEST_IMAGE)
    orig_count = len(res_orig[0].boxes)
    pre_path = os.path.join(combo_dir, "01_pre_attacco.jpg")
    res_orig[0].save(pre_path)
    print(f"   Oggetti originali: {orig_count}")

    # Esecuzione attacco
    print("   Esecuzione attacco...")
    adv_img = engine.run(
        image_path=TEST_IMAGE,
        text_prompt=TEXT_PROMPT,
        num_steps=args.steps,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        t_samples=args.t_samples,
        image_size=args.image_size
    )
    adv_path = os.path.join(combo_dir, "02_immagine_avversaria.png")
    adv_img.save(adv_path)

    # Recupero loss e salvataggio su disco
    loss_csv = None
    if engine.loss_history is not None:
        loss_history = engine.loss_history
        loss_csv = os.path.join(combo_dir, "losses.csv")
        with open(loss_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "total_loss", "yolo_loss", "sa_loss", "ca_loss"])
            for step in range(len(loss_history['total'])):
                writer.writerow([step,
                                 loss_history['total'][step],
                                 loss_history['yolo'][step],
                                 loss_history['sa'][step],
                                 loss_history['ca'][step]])

        # Salva plot locale
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['yolo'], label='YOLO loss')
        plt.plot(loss_history['sa'], label='Self-Attention loss')
        plt.plot(loss_history['ca'], label='Cross-Attention loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title(f'Andamento loss - {diff_name} / {yolo_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(combo_dir, "loss_plot.png"), dpi=150)
        plt.close()

        print(f"   Loss salvate ({len(loss_history['total'])} steps)")
    else:
        print("   Attenzione: loss_history non disponibile.")

    # Post-attacco
    res_adv = yolo_full(adv_path)
    adv_count = len(res_adv[0].boxes)
    post_path = os.path.join(combo_dir, "03_post_attacco.jpg")
    res_adv[0].save(post_path)
    print(f"   Oggetti dopo attacco: {adv_count}")

    success = adv_count < orig_count
    print(f"   Attacco riuscito: {success}")

    # Metriche
    metrics_entry = {
        "diffusion": diff_name,
        "yolo": yolo_name,
        "orig_count": orig_count,
        "adv_count": adv_count,
        "success": success,
        "reduction": orig_count - adv_count,
    }

    # Pulizia esplicita dei modelli e della memoria
    del diffusion_model
    del yolo_model
    del engine
    del yolo_full
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_entry, loss_csv


# ----------------------------------------------------------------------
# Funzione per generare grafici aggregati leggendo i CSV salvati
# ----------------------------------------------------------------------
def generate_aggregated_plots(out_dir, diff_models, yolo_models):
    """Genera grafici aggregati leggendo i file CSV delle loss."""
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Raccogli tutti i percorsi dei CSV
    loss_data = {}  # (diff, yolo) -> lista di loss per componente

    for diff in diff_models:
        for yolo in yolo_models:
            csv_path = os.path.join(out_dir, diff, yolo, "losses.csv")
            if os.path.exists(csv_path):
                # Leggi il CSV
                steps = []
                yolo_losses = []
                sa_losses = []
                ca_losses = []
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        steps.append(int(row['step']))
                        yolo_losses.append(float(row['yolo_loss']))
                        sa_losses.append(float(row['sa_loss']))
                        ca_losses.append(float(row['ca_loss']))

                loss_data[(diff, yolo)] = {
                    'steps': steps,
                    'yolo': yolo_losses,
                    'sa': sa_losses,
                    'ca': ca_losses
                }

    if not loss_data:
        print("Nessun dato di loss trovato per i grafici aggregati")
        return

    # ------------------------------------------------------------------
    # 1. Grafici per ogni modello di diffusione (confronto tra YOLO)
    # ------------------------------------------------------------------
    for diff in diff_models:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        components = [('yolo', 'YOLO loss'), ('sa', 'Self-Attention loss'), ('ca', 'Cross-Attention loss')]

        for ax, (comp, comp_name) in zip(axes, components):
            for yolo in yolo_models:
                key = (diff, yolo)
                if key in loss_data:
                    ax.plot(loss_data[key]['steps'], loss_data[key][comp], label=yolo)
            ax.set_ylabel(comp_name)
            ax.legend(loc='upper right')
            ax.grid(True)

        axes[-1].set_xlabel('Step')
        plt.suptitle(f'Confronto loss per modello di diffusione: {diff}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(plot_dir, f"loss_by_diff_{diff}.png"), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 2. Grafici per ogni modello YOLO (confronto tra diffusori)
    # ------------------------------------------------------------------
    for yolo in yolo_models:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        components = [('yolo', 'YOLO loss'), ('sa', 'Self-Attention loss'), ('ca', 'Cross-Attention loss')]

        for ax, (comp, comp_name) in zip(axes, components):
            for diff in diff_models:
                key = (diff, yolo)
                if key in loss_data:
                    ax.plot(loss_data[key]['steps'], loss_data[key][comp], label=diff)
            ax.set_ylabel(comp_name)
            ax.legend(loc='upper right')
            ax.grid(True)

        axes[-1].set_xlabel('Step')
        plt.suptitle(f'Confronto loss per modello YOLO: {yolo}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(plot_dir, f"loss_by_yolo_{yolo}.png"), dpi=150)
        plt.close()


# ----------------------------------------------------------------------
# Funzione principale di esecuzione batch
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch adversarial attack")
    parser.add_argument("--out", type=str, default="batch_results",
                        help="Directory base per i risultati")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--t_samples", type=int, nargs="+", default=DEFAULT_T_SAMPLES)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Token Hugging Face per download più rapidi da HF Hub")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"Immagine di test '{TEST_IMAGE}' non trovata.")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    all_metrics = []  # lista di dict con i risultati di ogni run

    for diff_name in DIFFUSION_MODELS:
        print(f"\n{'=' * 60}\nModello diffusione: {diff_name}\n{'=' * 60}")
        diff_out_dir = os.path.join(args.out, diff_name)
        os.makedirs(diff_out_dir, exist_ok=True)

        for yolo_name in YOLO_MODELS:
            combo_dir = os.path.join(diff_out_dir, yolo_name)
            os.makedirs(combo_dir, exist_ok=True)

            try:
                metrics_entry, _ = run_single_attack(
                    diff_name, yolo_name, combo_dir, args
                )
                all_metrics.append(metrics_entry)

            except Exception as e:
                print(f"!!! ERRORE nella combinazione {diff_name}/{yolo_name}: {e}")
                # Prova a liberare memoria anche in caso di errore
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                continue

    # Genera grafici comparativi (leggendo i CSV dal disco)
    if all_metrics:
        # Prepara liste modelli per i grafici
        diff_models = sorted(set(m["diffusion"] for m in all_metrics))
        yolo_models = sorted(set(m["yolo"] for m in all_metrics))

        # Genera grafici aggregati
        generate_aggregated_plots(args.out, diff_models, yolo_models)

        # Genera grafici a barre e tabella riassuntiva
        generate_summary_plots(all_metrics, args.out)

    print("\nElaborazione completata. Risultati in:", args.out)


# ----------------------------------------------------------------------
def generate_summary_plots(metrics, out_dir):
    """Genera grafici a barre e tabella CSV dai metric."""
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    diff_models = sorted(set(m["diffusion"] for m in metrics))
    yolo_models = sorted(set(m["yolo"] for m in metrics))

    # Grafico a barre del successo
    success_matrix = []
    for diff in diff_models:
        row = []
        for yolo in yolo_models:
            match = next((m for m in metrics if m["diffusion"] == diff and m["yolo"] == yolo), None)
            row.append(1 if match and match["success"] else 0)
        success_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(yolo_models))
    width = 0.2
    for i, diff in enumerate(diff_models):
        ax.bar([xi + i * width for xi in x], success_matrix[i], width, label=diff)
    ax.set_xlabel("Modello YOLO")
    ax.set_ylabel("Successo (1 = sì, 0 = no)")
    ax.set_title("Successo dell'attacco per combinazione")
    ax.set_xticks([xi + width * (len(diff_models) - 1) / 2 for xi in x])
    ax.set_xticklabels(yolo_models)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "success_bar.png"), dpi=150)
    plt.close()

    # Grafico a barre della riduzione oggetti
    reduction_matrix = []
    for diff in diff_models:
        row = []
        for yolo in yolo_models:
            match = next((m for m in metrics if m["diffusion"] == diff and m["yolo"] == yolo), None)
            row.append(match["reduction"] if match else 0)
        reduction_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, diff in enumerate(diff_models):
        ax.bar([xi + i * width for xi in x], reduction_matrix[i], width, label=diff)
    ax.set_xlabel("Modello YOLO")
    ax.set_ylabel("Riduzione numero oggetti")
    ax.set_title("Riduzione oggetti dopo attacco")
    ax.set_xticks([xi + width * (len(diff_models) - 1) / 2 for xi in x])
    ax.set_xticklabels(yolo_models)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reduction_bar.png"), dpi=150)
    plt.close()

    # Tabella riassuntiva CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["diffusion", "yolo", "orig_count", "adv_count", "success", "reduction"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Riassunto salvato in {csv_path}")


if __name__ == "__main__":
    main()
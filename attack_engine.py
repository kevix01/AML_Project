import numpy as np
import torch
import gc
from PIL import Image
from attack_strategy import AttackStrategy


class AttackEngine:
    def __init__(self, diffusion_model, yolo_model, hook_class):
        self.diffusion_model = diffusion_model
        self.yolo_model = yolo_model
        self.attack_strategy = AttackStrategy(diffusion_model, yolo_model, hook_class)

    def run(self, image_path, text_prompt, num_steps=30, lr=0.02, alpha=2.0, beta=0.5,
            t_samples=None, image_size=256, epsilon=16/255):
        # Preparazione immagine
        if t_samples is None:
            t_samples = [150, 400]
        target_size = self._prepare_size(image_size)
        init_image = Image.open(image_path).convert("RGB").resize(target_size)
        img_tensor = self._pil_to_tensor(init_image).to(self.diffusion_model.device)

        # Inizializza perturbazione
        delta = torch.zeros_like(img_tensor, dtype=torch.float32, requires_grad=True, device=self.diffusion_model.device)
        optimizer = torch.optim.Adam([delta], lr=lr)

        # Encoding prompt
        prompt_embeds = self.diffusion_model.encode_prompt(text_prompt)

        # Reference pass
        clean_sa, clean_ca = self.attack_strategy.reference_pass(img_tensor, prompt_embeds, t_samples)

        print(f"{'Step':>6} | {'L_YOLO':>8} | {'L_SA':>8} | {'L_CA':>10}")
        print("-" * 42)

        # Loop di ottimizzazione
        for step in range(num_steps):
            optimizer.zero_grad()
            adv_image = img_tensor + delta.half()
            adv_image_norm = (adv_image / 2 + 0.5).clamp(0, 1)

            total_loss, l_task, l_sa, l_ca = self.attack_strategy.compute_losses(
                adv_image_norm, prompt_embeds, clean_sa, clean_ca, t_samples, alpha, beta
            )

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)

            print(f"{step + 1:6d} | {l_task.item():8.4f} | {l_sa.item():8.4f} | {l_ca.item():+10.4f}")

            gc.collect()
            torch.cuda.empty_cache()

        # Genera immagine finale
        final_image = (img_tensor + delta.half()).detach()
        final_image = (final_image / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1,2,0).cpu().numpy()
        final_image = (final_image * 255).astype(np.uint8)
        return Image.fromarray(final_image)

    @staticmethod
    def _prepare_size(size):
        if isinstance(size, int):
            return size, size
        return size

    @staticmethod
    def _pil_to_tensor(img):
        arr = np.array(img).astype(np.float16) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
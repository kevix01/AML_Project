import torch
import torch.nn.functional as F

class AttackStrategy:
    def __init__(self, diffusion_model, yolo_model, hook_class):
        self.diffusion_model = diffusion_model
        self.yolo_model = yolo_model
        self.hook_class = hook_class

    def reference_pass(self, image_tensor, prompt_embeds, t_samples):
        """Calcola mappe di attenzione pulite per ogni timestep."""
        clean_maps_sa = {}
        clean_maps_ca = {}
        hook = self.hook_class()
        hook.register(self.diffusion_model.pipe.unet)

        with torch.no_grad():
            clean_latents = self.diffusion_model.encode_image(image_tensor)
            for t in t_samples:
                hook.clear()
                t_tensor = torch.tensor([t], device=self.diffusion_model.device, dtype=torch.long)
                noise = torch.randn_like(clean_latents)
                noisy_latents = self.diffusion_model.add_noise(clean_latents, noise, t_tensor)
                _ = self.diffusion_model.forward_unet(noisy_latents, t_tensor, prompt_embeds, hook)
                clean_maps_sa[t] = [m.detach().clone() for m in hook.sa_maps]
                clean_maps_ca[t] = [m.detach().clone() for m in hook.ca_maps]
        hook.remove()
        return clean_maps_sa, clean_maps_ca

    def compute_losses(self, adv_image_tensor, prompt_embeds, clean_maps_sa, clean_maps_ca, t_samples, alpha, beta):
        """Calcola le loss combinate in modo agnostico per tutte le versioni di YOLO."""
        # YOLO loss
        adv_image_yolo = F.interpolate(adv_image_tensor, size=(640, 640), mode='bilinear')
        yolo_preds = self.yolo_model(adv_image_yolo)

        # Estrai il tensore principale (Ultralytics spesso restituisce una tupla)
        out = yolo_preds[0] if isinstance(yolo_preds, (tuple, list)) else yolo_preds

        # ── COMPATIBILITÀ MULTI-YOLO ──────────────────────────────────────────
        if len(out.shape) == 3 and out.shape[-1] == 6:
            # YOLOv10 / YOLO26 (NMS-free): [batch, num_preds, 6]
            # Formato: [x1, y1, x2, y2, confidenza, classe]
            # Massimizziamo l'errore sulla colonna della confidenza (indice 4)
            l_task = torch.max(out[..., 4])

        elif len(out.shape) == 3 and out.shape[1] < out.shape[2]:
            # YOLOv8 / YOLOv9 / YOLO11: [batch, 4+classes, anchors]
            # Esempio: [1, 84, 8400]
            # Le probabilità delle classi partono dall'indice 4 della dimensione 1
            l_task = torch.max(out[:, 4:, :])

        elif len(out.shape) == 3 and out.shape[2] > 6:
            # Formati legacy (YOLOv5): [batch, anchors, 5+classes]
            # Esempio: [1, 25200, 85]
            # Objectness e probabilità iniziano dall'indice 4 dell'ultima dimensione
            l_task = torch.max(out[..., 4:])

        else:
            # Fallback di sicurezza se l'architettura è sconosciuta
            l_task = torch.max(out)
        # ─────────────────────────────────────────────────────────────────────

        # Codifica in latenti
        adv_latents = self.diffusion_model.encode_image(adv_image_tensor)

        l_sa = 0.0
        l_ca = 0.0
        hook = self.hook_class()
        hook.register(self.diffusion_model.pipe.unet)

        for t in t_samples:
            hook.clear()
            t_tensor = torch.tensor([t], device=self.diffusion_model.device, dtype=torch.long)
            noise = torch.randn_like(adv_latents)
            noisy_adv_latents = self.diffusion_model.add_noise(adv_latents, noise, t_tensor)
            _ = self.diffusion_model.forward_unet(noisy_adv_latents, t_tensor, prompt_embeds, hook)

            for adv_sa, ref_sa in zip(hook.sa_maps, clean_maps_sa[t]):
                l_sa += F.mse_loss(adv_sa.float(), ref_sa.float())

            for adv_ca, ref_ca in zip(hook.ca_maps, clean_maps_ca[t]):
                l_ca -= F.mse_loss(adv_ca.float(), ref_ca.float())

        hook.remove()

        l_sa /= len(t_samples)
        l_ca /= len(t_samples)

        total_loss = l_task + alpha * l_sa + beta * l_ca
        return total_loss, l_task, l_sa, l_ca
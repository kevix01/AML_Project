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
        """Calcola le loss combinate."""
        # YOLO loss
        adv_image_yolo = F.interpolate(adv_image_tensor, size=(640, 640), mode='bilinear')
        yolo_preds = self.yolo_model(adv_image_yolo)
        l_task = torch.max(yolo_preds[0][:, 4:, :])

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
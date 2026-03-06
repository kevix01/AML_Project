import torch
from diffusers import StableDiffusionPipeline
from diffusion_models.model_base import DiffusionModel


class TinySDModel(DiffusionModel):
    def load_pipeline(self, model_id="OFA-Sys/small-stable-diffusion-v0", hf_token: str = None, **kwargs):
        """
        Carica Tiny Stable Diffusion.
        Di default usa il modello small di OFA-Sys (solo 0.6B parametri).
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            hf_token=hf_token,
            requires_safety_checker=False,
            **kwargs
        )# .to(self.device) // disabilitato per offload modello

        self.scheduler = self.pipe.scheduler

        # Ottimizzazioni
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.unet.enable_gradient_checkpointing()

        # Offload delle componenti del modello
        self.pipe.enable_model_cpu_offload()

        # Congela i pesi
        for component in [self.pipe.unet, self.pipe.vae, self.pipe.text_encoder]:
            if component is not None:
                for param in component.parameters():
                    param.requires_grad = False

        print(f"Tiny SD caricato: {model_id}")
        print(f"UNet parametri: {sum(p.numel() for p in self.pipe.unet.parameters()) / 1e6:.1f}M")

    def encode_prompt(self, prompt):
        """Identico a SD15, restituisce embeddings del testo."""
        with torch.no_grad():
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            prompt_embeds = self.pipe.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        return prompt_embeds

    def encode_image(self, image_tensor):
        """Da immagine RGB ([-1,1]) a latenti."""
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * self.get_vae_scaling_factor()
        return latents

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = latents / self.get_vae_scaling_factor()
            image = self.pipe.vae.decode(latents).sample
        return image

    def add_noise(self, latents, noise, t):
        return self.scheduler.add_noise(latents, noise, t)

    def forward_unet(self, noisy_latents, t, prompt_embeds, hook=None):
        if hook is not None:
            hook.clear()
        output = self.pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=prompt_embeds
        )
        return output

    def get_vae_scaling_factor(self):
        if self.vae_scaling_factor is None:
            self.vae_scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 0.18215)
        return self.vae_scaling_factor
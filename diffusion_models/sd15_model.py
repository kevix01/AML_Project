import torch
from diffusers import StableDiffusionPipeline
from diffusion_models.model_base import DiffusionModel


class SD15Model(DiffusionModel):
    def load_pipeline(self, model_id="runwayml/stable-diffusion-v1-5", **kwargs):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            **kwargs
        ).to(self.device)
        self.scheduler = self.pipe.scheduler
        # Ottimizzazioni consigliate
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.unet.enable_gradient_checkpointing()
        # Congela i pesi
        for comp in [self.pipe.unet, self.pipe.vae, self.pipe.text_encoder]:
            for param in comp.parameters():
                param.requires_grad = False

    def encode_prompt(self, prompt):
        with torch.no_grad():
            text_inputs = self.pipe.tokenizer(
                prompt, padding="max_length", max_length=77, return_tensors="pt"
            )
            prompt_embeds = self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return prompt_embeds

    def encode_image(self, image_tensor):
        # image_tensor: [B, C, H, W] in range [-1, 1]
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * self.get_vae_scaling_factor()
        return latents

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = latents / self.get_vae_scaling_factor()
            image = self.pipe.vae.decode(latents).sample
        return image  # range [-1, 1]

    def add_noise(self, latents, noise, t):
        return self.scheduler.add_noise(latents, noise, t)

    def forward_unet(self, noisy_latents, t, prompt_embeds, hook=None):
        if hook is not None:
            hook.clear()
            # Il forward popolerà le mappe tramite hook registrati in precedenza
        output = self.pipe.unet(noisy_latents, t, encoder_hidden_states=prompt_embeds)
        return output
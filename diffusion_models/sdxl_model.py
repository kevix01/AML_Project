import torch
from diffusers import StableDiffusionXLPipeline
from diffusion_models.model_base import DiffusionModel

class SDXLFP8Model(DiffusionModel):
    def load_pipeline(self, model_id="wangkanai/sdxl-fp8", **kwargs):
        """
        Carica la pipeline SDXL con pesi quantizzati FP8.
        Il repository wangkanai/sdxl-fp8 su Hugging Face fornisce già i pesi in FP8.
        """
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,          # di solito float16
            use_safetensors=True,
            variant="fp8" if "fp8" in model_id else None,  # se disponibile
            **kwargs
        ).to(self.device)

        self.scheduler = self.pipe.scheduler

        # Ottimizzazioni di memoria (xformers, slicing, tiling)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        # SDXL non ha gradient checkpointing abilitato di default, ma se vuoi:
        # self.pipe.unet.enable_gradient_checkpointing()

        # Congela tutti i pesi per non modificarli durante l'attacco
        for component in [self.pipe.unet, self.pipe.vae,
                          self.pipe.text_encoder, self.pipe.text_encoder_2]:
            if component is not None:
                for param in component.parameters():
                    param.requires_grad = False

    def encode_prompt(self, prompt):
        """
        SDXL usa due text encoder. Restituisce una tupla:
        (prompt_embeds, pooled_prompt_embeds)
        """
        with torch.no_grad():
            # Il metodo encode_prompt della pipeline restituisce direttamente gli embeddings
            prompt_embeds, pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False  # No CFG per attacco
            )
        return prompt_embeds, pooled_prompt_embeds

    def encode_image(self, image_tensor):
        """
        Da immagine RGB (valori in [-1,1]) a latenti.
        Il factor di scaling per SDXL è di solito 0.13025.
        """
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
        """
        Esegue il forward dell'UNet di SDXL.
        prompt_embeds deve essere una tupla (embeds, pooled_embeds).
        """
        if hook is not None:
            hook.clear()
        # Separa i due embeddings
        embeds, pooled_embeds = prompt_embeds
        output = self.pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=embeds,
            pooled_prompt_embeds=pooled_embeds,
            return_dict=False
        )[0]
        return output

    def get_vae_scaling_factor(self):
        if self.vae_scaling_factor is None:
            # Valore tipico per SDXL
            self.vae_scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 0.13025)
        return self.vae_scaling_factor
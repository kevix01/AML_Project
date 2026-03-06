import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusion_models.model_base import DiffusionModel


class SegmindVegaModel(DiffusionModel):
    def load_pipeline(self, model_id="segmind/Segmind-Vega", hf_token=None, **kwargs):
        """
        Carica la pipeline Segmind-Vega con il VAE fix per FP16.
        """
        # Carica il VAE fix (per evitare NaN in FP16)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            use_safetensors=True,
            hf_token=hf_token,
        )

        # Carica la pipeline principale
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            hf_token=hf_token
        )

        self.scheduler = self.pipe.scheduler

        # Ottimizzazioni di memoria (senza offload)
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_attention_slicing()
        self.pipe.unet.enable_gradient_checkpointing()

        self.pipe.enable_model_cpu_offload()

        # Sposta tutto sulla GPU
        # NOTA: disabilitato in quanto usato model offloading
        # self.pipe.to(self.device)

        # Congela i pesi
        for component in [self.pipe.unet, self.pipe.vae,
                          self.pipe.text_encoder, self.pipe.text_encoder_2]:
            if component is not None:
                for param in component.parameters():
                    param.requires_grad = False

        print(f"Segmind-Vega caricato con VAE fix FP16 su {self.device}")

    def encode_prompt(self, prompt):
        """Codifica il prompt text in embeddings."""
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
        return prompt_embeds, pooled_prompt_embeds

    def encode_image(self, image_tensor):
        """Da immagine RGB (valori in [-1,1]) a latenti."""

        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()

        # Applica scaling factor (tipico per SDXL/Vega: 0.13025)
        latents = latents * self.get_vae_scaling_factor()

        return latents

    def decode_latents(self, latents):
        """Decodifica i latenti in spazio RGB."""
        latents = latents / self.get_vae_scaling_factor()

        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample

        return image  # range [-1, 1]

    def add_noise(self, latents, noise, t):
        """Applica rumore secondo lo scheduler."""
        return self.scheduler.add_noise(latents, noise, t)

    def forward_unet(self, noisy_latents, t, prompt_embeds, hook=None):
        """
        Esegue forward dell'UNet (identico a SDXL).
        prompt_embeds deve essere una tupla (embeds, pooled_embeds).
        """
        if hook is not None:
            hook.clear()

        embeds, pooled_embeds = prompt_embeds

        # Calcola dimensioni per SDXL conditioning
        batch_size = noisy_latents.shape[0]
        latent_height = noisy_latents.shape[2]
        latent_width = noisy_latents.shape[3]

        vae_scale_factor = 8
        image_height = int(latent_height * vae_scale_factor)
        image_width = int(latent_width * vae_scale_factor)

        original_size = (image_height, image_width)
        target_size = (image_height, image_width)
        crops_coords_top_left = (0, 0)

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=embeds.dtype, device=embeds.device)

        if batch_size > 1:
            add_time_ids = add_time_ids.repeat(batch_size, 1)

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids
        }

        output = self.pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]

        return output

    def get_vae_scaling_factor(self):
        """Restituisce il fattore di scaling del VAE."""
        if self.vae_scaling_factor is None:
            self.vae_scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 0.13025)
        return self.vae_scaling_factor
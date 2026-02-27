import torch
import os
from diffusers import Flux2Pipeline  # Pipeline specifica per FLUX.2 Klein
from huggingface_hub import hf_hub_download
from diffusion_models.model_base import DiffusionModel

class Flux2KleinFP8Model(DiffusionModel):
    """
    FLUX.2 Klein 4B Base in FP8.
    - 4B parameters, FP8 quantized (~8-9GB VRAM)
    - Apache 2.0 license
    - Undistilled base model (50 steps recommended)
    """

    def __init__(self, device, dtype=torch.float16, checkpoint_path=None, auto_download=True, token=None):
        super().__init__(device, dtype)
        self.checkpoint_path = checkpoint_path
        self.auto_download = auto_download
        # Se token non fornito, prova a prenderlo da variabile d'ambiente
        self.token = token or os.environ.get("HF_TOKEN")
        if self.checkpoint_path is not None and self.token is None:
            raise ValueError(
                "Hugging Face token richiesto per accedere al modello. "
                "Passalo come argomento token=... o imposta la variabile d'ambiente HF_TOKEN."
            )

    def _download_checkpoint(self):
        """
        Scarica automaticamente il checkpoint FP8 da Hugging Face.
        Il repository è pubblico ma richiede login per i file.
        """
        repo_id = "black-forest-labs/FLUX.2-klein-base-4b-fp8"
        # Il nome esatto del file .safetensors nel repository; se non corrisponde,
        # il download fallirà e useremo from_pretrained.
        filename = "flux2-klein-base-4b-fp8.safetensors"

        try:
            print(f"Tentativo download checkpoint da: {repo_id}")
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=self.token
            )
            print(f"✓ Checkpoint scaricato: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            print(f"✗ Download fallito: {str(e)}")
            print("Si procede con from_pretrained (caricamento automatico del modello).")
            return None

    def load_pipeline(self, model_id=None, **kwargs):
        """
        Carica la pipeline FLUX.2 Klein.
        Se checkpoint_path è specificato, usa from_single_file.
        Altrimenti usa from_pretrained con il repository ufficiale.
        """
        if self.checkpoint_path is None and self.auto_download:
            self.checkpoint_path = self._download_checkpoint()

        if self.checkpoint_path is not None:
            print(f"Caricamento FLUX.2 Klein FP8 da: {self.checkpoint_path}")
            self.pipe = Flux2Pipeline.from_single_file(
                self.checkpoint_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                token=self.token,
                **kwargs
            ).to(self.device)
        else:
            # Carica direttamente dal repository (scarica i file necessari)
            model_id = model_id or "black-forest-labs/FLUX.2-klein-base-4b-fp8"
            print(f"Caricamento FLUX.2 Klein FP8 da {model_id} tramite from_pretrained")
            self.pipe = Flux2Pipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                # token=self.token,
                **kwargs
            ).to(self.device)

        self.scheduler = self.pipe.scheduler

        # Ottimizzazioni memoria (evitiamo cpu_offload perché rallenta l'attacco)
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        # FLUX.2 Klein usa Qwen3 text encoder (singolo)
        print("Text encoder: Qwen3 4B")

        # Congela tutti i pesi per evitare gradienti indesiderati
        for component in [self.pipe.transformer, self.pipe.vae,
                          getattr(self.pipe, 'text_encoder', None)]:
            if component is not None:
                for param in component.parameters():
                    param.requires_grad = False

        print(f"FLUX.2 Klein pipeline caricata con successo. Dtype: {self.dtype}")

    def encode_prompt(self, prompt):
        """
        Restituisce gli embeddings del testo.
        FLUX.2 Klein usa un singolo text encoder (Qwen3).
        """
        with torch.no_grad():
            # encode_prompt restituisce (prompt_embeds, pooled_prompt_embeds, text_ids)
            prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
        return prompt_embeds, pooled_prompt_embeds

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
        """Aggiunge rumore secondo lo scheduler (flow matching)."""
        return self.scheduler.add_noise(latents, noise, t)

    def forward_unet(self, noisy_latents, t, prompt_embeds, hook=None):
        """
        Forward del transformer.
        prompt_embeds è una tupla (embeds, pooled_embeds).
        """
        if hook is not None:
            hook.clear()
        embeds, pooled = prompt_embeds
        output = self.pipe.transformer(
            noisy_latents,
            timestep=t,
            encoder_hidden_states=embeds,
            pooled_projections=pooled,  # Parametro specifico di FLUX.2; se non serve, commentare
            return_dict=False
        )[0]
        return output

    def get_vae_scaling_factor(self):
        if self.vae_scaling_factor is None:
            # Valore tipico per VAE di FLUX (da configurazione)
            self.vae_scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 0.13025)
        return self.vae_scaling_factor
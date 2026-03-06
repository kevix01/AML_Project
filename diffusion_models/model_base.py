from abc import ABC, abstractmethod
import torch

class DiffusionModel(ABC):
    def __init__(self, device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.scheduler = None
        self.vae_scaling_factor = None

    @abstractmethod
    def load_pipeline(self, model_id, hf_token, **kwargs):
        """Carica la pipeline e assegna self.pipe, self.scheduler."""
        pass

    @abstractmethod
    def encode_prompt(self, prompt):
        """Restituisce gli embeddings del testo."""
        pass

    @abstractmethod
    def encode_image(self, image_tensor):
        """Da immagine RGB (valori in [-1,1]) a latenti."""
        pass

    @abstractmethod
    def decode_latents(self, latents):
        """Da latenti a immagine RGB (valori in [-1,1])."""
        pass

    @abstractmethod
    def add_noise(self, latents, noise, t):
        """Applica rumore secondo lo scheduler."""
        pass

    @abstractmethod
    def forward_unet(self, noisy_latents, t, prompt_embeds, hook=None):
        """Esegue forward U-Net e, se hook fornito, raccoglie attenzioni."""
        pass

    def get_vae_scaling_factor(self):
        if self.vae_scaling_factor is None:
            self.vae_scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 1.0)
        return self.vae_scaling_factor
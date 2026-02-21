from abc import ABC, abstractmethod


class AttentionHook(ABC):
    def __init__(self):
        self.sa_maps = []
        self.ca_maps = []
        self.hooks = []

    @abstractmethod
    def register(self, unet):
        """Registra gli hook sui layer di attenzione del modello."""
        pass

    def clear(self):
        self.sa_maps.clear()
        self.ca_maps.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
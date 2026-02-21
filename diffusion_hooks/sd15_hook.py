from diffusers.models.attention_processor import Attention
from diffusion_hooks.hook_base import AttentionHook


class SD15AttentionHook(AttentionHook):
    def __init__(self, target_blocks=None):
        super().__init__()
        if target_blocks is None:
            target_blocks = ["mid_block", "up_blocks.1", "up_blocks.2"]
        self.target_blocks = target_blocks

    def register(self, unet):
        for name, module in unet.named_modules():
            if isinstance(module, Attention) and any(b in name for b in self.target_blocks):
                if "attn1" in name:
                    self.hooks.append(module.register_forward_hook(self._hook_sa(name)))
                elif "attn2" in name:
                    self.hooks.append(module.register_forward_hook(self._hook_ca(name)))

    def _hook_sa(self, name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.sa_maps.append(out)
        return hook

    def _hook_ca(self, name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.ca_maps.append(out)
        return hook
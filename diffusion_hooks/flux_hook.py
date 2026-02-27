from diffusion_hooks.hook_base import AttentionHook


class Flux2KleinAttentionHook(AttentionHook):
    """
    Hook per catturare self‑attention e cross‑attention nel transformer di FLUX.2 Klein.
    Klein ha una struttura simile a FLUX.1 ma con layer di attenzione organizzati diversamente.
    """

    def __init__(self, target_blocks=None):
        super().__init__()
        self.target_blocks = target_blocks  # eventuale filtro

    def register(self, transformer):
        """
        Registra hook su tutti i moduli di attenzione.
        Cerca moduli con 'attn' nel nome e distingue self/cross in base a pattern.
        """
        for name, module in transformer.named_modules():
            # Verifica se è un modulo di attenzione (ha i pesi q,k,v)
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                if 'cross_attn' in name or 'context_attn' in name:
                    self.hooks.append(module.register_forward_hook(self._hook_ca(name)))
                elif 'attn' in name and not any(x in name for x in ['cross', 'context']):
                    self.hooks.append(module.register_forward_hook(self._hook_sa(name)))
                # Altri casi li ignoriamo (potrebbero essere attenzione mista)

        # Se non abbiamo trovato nulla, stampa i nomi per debug
        if len(self.hooks) == 0:
            print("ATTENZIONE: Nessun modulo di attenzione trovato. Nomi moduli disponibili:")
            for i, (name, _) in enumerate(transformer.named_modules()):
                if i < 50:  # primi 50 per non invadere
                    print(f"  {name}")

        print(f"Registrati {len(self.hooks)} hook su attention layers di FLUX.2 Klein")

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
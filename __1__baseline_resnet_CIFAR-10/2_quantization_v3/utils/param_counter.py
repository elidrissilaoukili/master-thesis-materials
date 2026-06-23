import torch
import torch.nn as nn

def count_params(model, model_label=""):
    """
    Consistent parameter counting for FP32, dynamic PTQ, and FX-quantized models.

    - nn.Parameter objects: deduplicated by id(p)
    - FX quantized .weight() callables: deduplicated by (module_path, 'weight') key
    - Avoids data_ptr() aliasing bugs with views/slices
    """
    total, nonzero = 0, 0
    seen_param_ids   = set()
    seen_module_keys = set()

    for mod_name, module in model.named_modules():
        # Standard parameters (FP32, bias terms, dynamic PTQ)
        for param_name, p in module.named_parameters(recurse=False):
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            n        = p.numel()
            total   += n
            nonzero += int((p != 0).sum().item())

        # FX quantized weight callables
        if hasattr(module, 'weight') and callable(module.weight):
            key = (mod_name, 'weight')
            if key in seen_module_keys:
                continue
            seen_module_keys.add(key)
            try:
                w = module.weight()
                if isinstance(w, torch.Tensor) and w.numel() > 0:
                    total   += w.numel()
                    nonzero += int((w != 0).sum().item())
            except Exception:
                pass

    if model_label:
        print(f"count_params [{model_label}]: "
              f"total={total:,}  nonzero={nonzero:,}  "
              f"sparsity={1 - nonzero/max(total,1):.4f}")

    return {"params_total": int(total), "params_nonzero": int(nonzero)}
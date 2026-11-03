import torch
import numpy as np

# Register numpy scalar types as safe globals (numpy 1.x and 2.x paths)
safe_globals = []
try:
    import numpy.core.multiarray
    safe_globals.append(numpy.core.multiarray.scalar)
except (AttributeError, ImportError):
    pass
try:
    import numpy._core.multiarray
    safe_globals.append(numpy._core.multiarray.scalar)
except (AttributeError, ImportError):
    pass

if safe_globals:
    try:
        torch.serialization.add_safe_globals(safe_globals)
    except AttributeError:
        pass

ef_path = "models/r2plus1d_18_32_2_pretrained.pt"
print(f"Loading {ef_path}")

try:
    state_dict = torch.load(ef_path, map_location='cpu', weights_only=False)
    print("Successfully loaded EF weights")
except Exception as e:
    print(f"Failed to load EF weights: {e}")

seg_path = "models/deeplabv3_resnet50_random.pt"
print(f"Loading {seg_path}")
try:
    state_dict = torch.load(seg_path, map_location='cpu', weights_only=False)
    print("Successfully loaded Seg weights")
except Exception as e:
    print(f"Failed to load Seg weights: {e}")

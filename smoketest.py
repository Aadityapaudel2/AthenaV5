# quick_cuda_check.py
import os, sys, platform

print("python:", sys.executable)
print("python_version:", sys.version.split()[0])
print("platform:", platform.platform())

try:
    import torch
    print("\n[torch]")
    print("torch_version:", torch.__version__)
    print("torch_cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version_build:", torch.version.cuda)
    if torch.cuda.is_available():
        print("cuda_device_count:", torch.cuda.device_count())
        i = 0
        print("cuda_device_0_name:", torch.cuda.get_device_name(i))
        print("cuda_device_0_capability:", torch.cuda.get_device_capability(i))
        print("cuda_device_0_total_mem_GB:", round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2))
        print("torch_cudnn_version:", torch.backends.cudnn.version())
    else:
        print("torch_cudnn_version:", torch.backends.cudnn.version())
except Exception as e:
    print("\n[torch] import failed:", repr(e))

print("\n[env]")
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    import torchvision
    print("\n[torchvision]")
    print("torchvision_version:", torchvision.__version__)
    # minimal op that often fails if cuda/torchvision mismatch
    from torchvision.ops import nms
    print("torchvision_ops_nms_import: ok")
except Exception as e:
    print("\n[torchvision] import/ops failed:", repr(e))

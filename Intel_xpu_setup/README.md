
docker run -it --rm     --device /dev/dri     -v /dev/dri/by-path:/dev/dri/by-path     --ipc=host     intel/intel-extension-for-pytorch:2.7.10-xpu

```
import torch
import intel_extension_for_pytorch as ipex

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if XPU (Intel GPU) is available
print("Is XPU available?", torch.xpu.is_available())

# Check XPU device count
if torch.xpu.is_available():
    print("Number of XPUs available:", torch.xpu.device_count())
    print("XPU Device name:", torch.xpu.get_device_name(0))
else:
    print("XPU not available.")

```

```
python test.py
[W504 04:41:13.404657067 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:186 (function operator())
PyTorch version: 2.7.0+xpu
Is XPU available? True
Number of XPUs available: 1
XPU Device name: Intel(R) Arc(TM) Graphics
```

NPU: (not supported yet.)
```
lspci | grep -i npu
lsusb | grep -i myriad
00:0b.0 Processing accelerators: Intel Corporation Meteor Lake NPU (rev 04)
```



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

Test code for XPU:
```
import torch
import numpy as np
import time

# Synthetic test data
N = 4000
all_mu = np.random.rand(N).astype(np.float32)
S = np.random.rand(N, N).astype(np.float32)
S = (S + S.T) / 2  # make symmetric positive semi-definite
S += N * np.eye(N, dtype=np.float32)

mu = torch.tensor(all_mu, dtype=torch.float32)
sigma = torch.tensor(S, dtype=torch.float32)

# Projection onto capped simplex (box constraints [0, 0.1], sum-to-one)
def project_capped_simplex(w, cap=0.1):
    w_clamped = torch.clamp(w, 0.0, cap)
    return w_clamped / (w_clamped.sum() + 1e-12)


def optimize_portfolio(mu, sigma, device):
    mu = mu.to(device)
    sigma = sigma.to(device)
    raw_w = torch.nn.Parameter(torch.full((N,), 1.0 / N, device=device))
    optimizer = torch.optim.Adam([raw_w], lr=1e-4)

    start_time = time.time()

    for step in range(1000):
        optimizer.zero_grad()
        w = project_capped_simplex(raw_w)
        #port_ret = torch.dot(w.to(device), mu) # Explicitly move w to the device.
        port_ret = (w * mu).sum()
        #port_ret = torch.dot(w, mu)
        sigma_w = torch.matmul(sigma, w)
        port_var = (w * sigma_w).sum()
        port_risk = torch.sqrt(port_var + 1e-12)
        one_sigma = port_ret - port_risk
        loss = -one_sigma
        loss.backward()
        optimizer.step()

    torch.xpu.synchronize() if device.type == 'xpu' else None  # Corrected sync
    end_time = time.time()

    return end_time - start_time


def main():
    device = torch.device("xpu")

    if torch.xpu.is_available():
        print("Intel GPU (XPU) is available.")
    else:
        print("Intel GPU (XPU) is NOT available.")
        return

    print(f"Running optimization on {device.type.upper()}...")
    elapsed = optimize_portfolio(mu, sigma, device)
    print(f"{device.type.upper()} time: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()
```


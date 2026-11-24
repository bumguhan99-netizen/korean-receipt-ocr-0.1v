import importlib.metadata
import platform
import subprocess
import sys
import torch
from common import write_json


def main():
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={k:importlib.metadata.version(k) for k in ('torch','torchvision','transformers','huggingface-hub','Pillow')},
                torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
    if info['cuda_available']:
        info.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                    supported_arches=torch.cuda.get_arch_list(), bf16_supported=torch.cuda.is_bf16_supported())
        # Verify actual kernels execute; availability/driver detection alone is insufficient.
        a = torch.randn(64,64,device='cuda',dtype=torch.bfloat16)
        b = torch.nn.functional.scaled_dot_product_attention(a[None,None],a[None,None],a[None,None])
        assert torch.isfinite(b).all().item()
        torch.cuda.synchronize()
        info['cuda_sdpa_bf16_kernel_check'] = 'passed'
    write_json('outputs/glm_environment.json', info)
    print({k:v for k,v in info.items() if k!='nvidia_smi'})


if __name__ == '__main__':
    main()

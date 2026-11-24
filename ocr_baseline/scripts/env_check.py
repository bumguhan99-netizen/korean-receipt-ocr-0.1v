import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from common import write_json


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=30)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error))


def main():
    import paddle
    data = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version,
                executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
                python_installations=capture(['py', '-0p']),
                hardware=capture(['powershell', '-NoProfile', '-Command',
                    'Get-CimInstance Win32_Processor | Select-Object Name | ConvertTo-Json; '
                    'Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json']),
                nvidia=capture(['nvidia-smi']), cuda_compiled=paddle.is_compiled_with_cuda(),
                packages={k:importlib.metadata.version(k) for k in ('paddlepaddle', 'paddleocr', 'paddlex', 'huggingface-hub', 'Pillow')})
    write_json('outputs/environment.json', data)
    print(data['packages'])
    print('CUDA compiled:', data['cuda_compiled'])


if __name__ == '__main__':
    main()

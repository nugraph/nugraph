import os
from pynvml.smi import nvidia_smi

def configure_device(cpu: bool = False) -> tuple[str, str | list[int]]:
    if not cpu:
        try:
            nvsmi = nvidia_smi.getInstance()
            info = nvsmi.DeviceQuery('index,memory.free')['gpu']
            info.sort(key=lambda m: m['fb_memory_usage']['free'], reverse=True)
            return 'auto', [int(info[0]['minor_number'])]
        except:
            pass
    return 'cpu', 'auto'
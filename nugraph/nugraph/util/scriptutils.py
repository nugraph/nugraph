"""Utility functions for scripts and notebooks"""
import os
import sys
import platform
from pynvml.smi import nvidia_smi

NUGRAPH_ENV = {
    "sneezy": {
        "NUGRAPH_DIR": "/data/home/$USER/nugraph",
        "NUGRAPH_DATA": "/share/lazy/nugraph",
        "NUGRAPH_LOG": "/data/home/$USER/logs",
    },
    "heimdall": {
        "NUGRAPH_DIR": "/raid/$USER/nugraph",
        "NUGRAPH_DATA": "/share/lazy/nugraph",
        "NUGRAPH_LOG": "/raid/$USER/logs",
    },
}

def setup_env(verbose: bool = True) -> None:
    """Configure NuGraph environment automatically"""
    env_vars = ("NUGRAPH_DIR", "NUGRAPH_DATA", "NUGRAPH_LOG")
    nugraph_env = {var: os.getenv(var) for var in env_vars}

    host = platform.node().lower()
    if not all(nugraph_env.values()) and host not in NUGRAPH_ENV:
        print(f"Could not detect nugraph environment on host \"{host}\".")
        print(("Please export the NUGRAPH_DIR, NUGRAPH_DATA and NUGRAPH_LOG "
               "environment variables before running, or add your current "
               "host to the NUGRAPH_ENV dictionary in "
               "nugraph/util/scriptutils.py"))
        exit(1)

    if verbose:
        print("\nconfiguring nugraph environment:")
    for var in env_vars:
        if nugraph_env[var]:
            if verbose:
                print(f"  {var:14}(set from env var)  - {nugraph_env[var]}")
        else:
            os.environ[var] = os.path.expandvars(NUGRAPH_ENV[host][var])
            if verbose:
                print(f"  {var:14}(set from host)     - {os.getenv(var)}")
    if verbose:
        print()

def configure_device(cpu: bool = False) -> tuple[str, str | list[int]]:
    if not cpu:
        try:
            # query GPU information using pynvml
            nvsmi = nvidia_smi.getInstance()
            info = nvsmi.DeviceQuery('index,memory.free')['gpu']

            # if there aren't multiple GPUs, don't do anything
            if len(info) < 2:
                return 'auto', 'auto'

            # if there are multiple GPUs, select the one with the most memory
            info.sort(key=lambda m: m['fb_memory_usage']['free'], reverse=True)
            return 'auto', [int(info[0]['minor_number'])]
        except:
            pass
    return 'cpu', 'auto'


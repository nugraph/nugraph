"""Utility functions for scripts and notebooks"""
import os
import platform

import torch.cuda

NUGRAPH_ENV = {
    "sneezy": {
        "NUGRAPH_DIR": "/data/home/$USER/nugraph",
        "NUGRAPH_DATA": "/data/home/hewesje/data",
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

def configure_device(device: int = None) -> tuple[str, str | list[int]]:
    """
    Convert GPU device index into a set of PyTorch Lightning arguments

    Args:
        device: index of GPU device to use
    """

    # if no device passed, run on CPU
    if device is None:
        return "cpu", "auto"

    # if a device was requested but there are none available, raise an error
    if not torch.cuda.is_available():
        raise RuntimeError((f"Device {device} requested but CUDA is not "
                             "available in the current environment."))

    # return GPU device as single element list
    return "gpu", [device]

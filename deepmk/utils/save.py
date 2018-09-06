import os
import shutil
import torch

__all__ = ["save"]

def save(obj, fpath):
    torch.save(obj, fpath+".temp")
    try:
        os.rename(fpath+".temp", fpath)
    except OSError:
        if os.path.isfile(fpath):
            os.remove(fpath)
        if os.path.isdir(fpath):
            shutil.rmtree(fpath)
        os.rename(fpath+".temp", fpath)

import os
import sys

# This line needs to be added since some terminals will not recognize the current directory
os.chdir("/home/wentinn/workspace/KF_RNN/")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch

from infrastructure.records import recarray


if __name__ == "__main__":
    recarr = recarray((), dtype=[("x", int), ("y", object)])
    recarr[()] = (1212, torch.randn((5, 5)))
    print(recarr.y[()] is recarr[()].y)        # This line does not break
    print(type(recarr[()].y))    # This line breaks





import sys
import pkg_resources
import torch

def check_environment():
    print("Python version:", sys.version)
    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    print("\nInstalled packages:")
    for package in pkg_resources.working_set:
        print(f"{package.key} {package.version}")

if __name__ == "__main__":
    check_environment() 
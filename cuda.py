import torch

def check_cuda_available():
    """
    Check if CUDA is available on the device.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()

if __name__ == "__main__":
    cuda_available = check_cuda_available()
    print("CUDA is available:", cuda_available)

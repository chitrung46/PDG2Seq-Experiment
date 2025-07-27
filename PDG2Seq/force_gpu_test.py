import torch
import time
import os

def force_gpu_usage():
    """Force heavy GPU computation để test GPU usage"""
    print("=== Testing GPU Usage ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
    
    # Create large tensors để force GPU usage
    size = 3000
    print(f"Creating {size}x{size} tensors on GPU...")
    
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    print("Starting intensive GPU computation...")
    print("This should trigger GPU usage to increase...")
    
    start_time = time.time()
    
    # Loop with heavy matrix operations
    for i in range(100):
        # Heavy matrix multiplication
        c = torch.mm(a, b)
        
        # More operations to keep GPU busy
        c = torch.relu(c)
        c = torch.sin(c)
        c = torch.cos(c)
        c = torch.exp(c * 0.01)  # Prevent overflow
        c = torch.mm(c, a)
        
        # Periodically update one of the matrices
        if i % 10 == 0:
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            print(f"Iteration {i+1}/100 - GPU should be working hard!")
            
        # Force synchronization every few iterations
        if i % 20 == 0:
            torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Computation completed in {end_time - start_time:.2f} seconds")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print("GPU usage should have increased during this test!")

if __name__ == "__main__":
    force_gpu_usage()
    
    # Keep the process alive for a bit
    print("\nKeeping process alive for 30 seconds to monitor GPU usage...")
    time.sleep(30)
    print("Test completed!")

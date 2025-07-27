#!/usr/bin/env python3
"""
Script để force GPU usage trong training
"""
import os
import sys
import torch
import torch.nn as nn
import subprocess
import time

# Change to working directory
os.chdir("/kaggle/working/PDG2Seq-Experiment/PDG2Seq")

print("=== GPU Training with Heavy Computation ===")

# Test 1: Force GPU usage first
print("Step 1: Testing basic GPU computation...")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create heavy computation
    print("Creating heavy GPU computation...")
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)
    
    for i in range(20):
        c = torch.mm(a, b)
        c = torch.relu(c)
        c = torch.sin(c)
        if i % 5 == 0:
            print(f"Heavy computation iteration {i+1}/20")
    
    torch.cuda.synchronize()
    print(f"GPU Memory after heavy computation: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    
    # Clear cache
    del a, b, c
    torch.cuda.empty_cache()
    
    print("Basic GPU test completed. GPU should show activity!")
    
    # Wait a bit
    time.sleep(5)
    
    print("\nStep 2: Running training with forced GPU computation...")
    
    # Run training
    cmd = [
        sys.executable, "run_hypergraph.py",
        "--dataset", "NYC-Bike",
        "--model", "PDG2Seq_HyperGCN", 
        "--mode", "train",
        "--device", "cuda:0",
        "--epochs", "10",
        "--batch_size", "32"  # Larger batch size
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Start training process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    # Monitor output
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
        if process.poll() is not None:
            break
    
    process.wait()
    print(f"Training completed with return code: {process.returncode}")
    
else:
    print("CUDA not available!")

print("Test completed!")

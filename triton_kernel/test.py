import os
import sys
sys.path.append(os.path.abspath('.'))  # Add current directory

import torch
import torch.nn as nn
import torch.nn.functional as F
from int8_matmul import triton_linear_int8
from triton_kernel.conv2d import triton_conv2d_int8
import time

def test_matmul_forward():
    # Create small tensors for testing
    batch_size, in_features, out_features = 4, 8, 2
    
    # Create random input and weights
    input_int8 = torch.randint(-128, 127, (batch_size, in_features), 
                              dtype=torch.int8, device='cuda')
    weight_int8 = torch.randint(-128, 127, (out_features, in_features), 
                               dtype=torch.int8, device='cuda')
    bias_int32 = torch.randint(-1000, 1000, (out_features,), 
                              dtype=torch.int32, device='cuda')
    
    # PyTorch reference implementation
    pytorch_output = torch.nn.functional.linear(
        input_int8.float(),
        weight_int8.float(),
        bias_int32.float()
    )
    
    # Our Triton implementation
    triton_output = triton_linear_int8(input_int8, weight_int8, bias_int32).float()
    
    # Check results
    max_diff = (pytorch_output - triton_output).abs().max().item()
    print(f"Forward max difference: {max_diff}")
    print(f"Forward results match: {torch.allclose(pytorch_output, triton_output, rtol=1e-3, atol=1e-3)}")


def test_conv2d_forward():
    # Create small tensors for testing
    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3
    stride, padding = (1, 1), (1, 1)
    
    # Create random input and weights
    input_int8 = torch.randint(-128, 127, (batch_size, in_channels, height, width), 
                              dtype=torch.int32, device='cuda')
    weight_int8 = torch.randint(-128, 127, (out_channels, in_channels, kernel_size, kernel_size), 
                               dtype=torch.int32, device='cuda')
    bias_int32 = torch.randint(-1000, 1000, (out_channels,), 
                              dtype=torch.int32, device='cuda')
    
    # PyTorch reference implementation
    pytorch_output = F.conv2d(
        input_int8.float(),
        weight_int8.float(),
        bias_int32.float(),
        stride=stride,
        padding=padding
    )
    
    # Our Triton implementation
    triton_output = triton_conv2d_int8(input_int8, weight_int8, bias_int32, 
                                      stride=stride, padding=padding).float()
    
    # Check results
    max_diff = (pytorch_output - triton_output).abs().max().item()
    print(f"Conv2D forward max difference: {max_diff}")
    print(f"Conv2D forward results match: {torch.allclose(pytorch_output, triton_output, rtol=1e-3, atol=1e-3)}")

    # Performance test
    warmup = 10
    repeats = 100
    
    # Warmup
    for _ in range(warmup):
        _ = F.conv2d(input_int8.float(), weight_int8.float(), bias_int32.float(), 
                     stride=stride, padding=padding)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        _ = F.conv2d(input_int8.float(), weight_int8.float(), bias_int32.float(), 
                    stride=stride, padding=padding)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - t0) / repeats
    
    # Warmup
    for _ in range(warmup):
        _ = triton_conv2d_int8(input_int8, weight_int8, bias_int32, 
                             stride=stride, padding=padding)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        _ = triton_conv2d_int8(input_int8, weight_int8, bias_int32, 
                             stride=stride, padding=padding)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / repeats
    
    print(f"PyTorch time: {pytorch_time*1000:.3f} ms")
    print(f"Triton time: {triton_time*1000:.3f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")

if __name__ == "__main__":
    print("Testing matmul forward implementation...")
    test_matmul_forward()
    
    print("\nTesting conv2d forward implementation...")
    test_conv2d_forward()
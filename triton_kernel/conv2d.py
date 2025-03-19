import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def _conv2d_int8_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_channels, in_height, in_width, 
    out_channels, out_height, out_width,
    kernel_size, stride, padding,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_IC: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)  # batch_idx * out_height + h_out
    pid_n = tl.program_id(axis=1)  # out_channels
    pid_l = tl.program_id(axis=2)  # out_width
    
    # Compute batch and spatial indices
    batch_idx = pid_m // out_height
    h_out = pid_m % out_height
    w_out = pid_l
    
    # Compute input starting position
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding
    
    # Calculate output channel block
    oc_start = pid_n * BLOCK_SIZE_N
    oc_offsets = tl.arange(0, BLOCK_SIZE_N)
    oc_mask = oc_start + oc_offsets < out_channels
    ocs = oc_start + oc_offsets
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.int32)
    
    # Load bias if available
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + ocs, mask=oc_mask, other=0)
        acc += bias
    
    # Loop over input channels in blocks
    for ic_block_idx in range((in_channels + BLOCK_SIZE_IC - 1) // BLOCK_SIZE_IC):
        ic_start = ic_block_idx * BLOCK_SIZE_IC
        ic_end = min(ic_start + BLOCK_SIZE_IC, in_channels)
        
        # Iterate over input channels and kernel dimensions
        for ic in range(ic_start, ic_end):
            for kh in range(kernel_size):
                h_in = h_in_start + kh
                if (0 <= h_in) & (h_in < in_height):
                    for kw in range(kernel_size):
                        w_in = w_in_start + kw
                        if (0 <= w_in) & (w_in < in_width):
                            # Load input value (shared across output channels)
                            in_idx = ((batch_idx * in_channels + ic) * in_height + h_in) * in_width + w_in
                            in_val = tl.load(input_ptr + in_idx).to(tl.int32)
                            
                            # Load weight values (for multiple output channels)
                            weight_idx = ((ocs * in_channels + ic) * kernel_size + kh) * kernel_size + kw
                            weight_val = tl.load(weight_ptr + weight_idx, mask=oc_mask, other=0).to(tl.int32)
                            
                            # Accumulate products for multiple output channels
                            acc += in_val * weight_val
    
    # Write output
    out_idx = ((batch_idx * out_channels + ocs) * out_height + h_out) * out_width + w_out
    tl.store(output_ptr + out_idx, acc, mask=oc_mask)

def triton_conv2d_int8(input_tensor, weight_tensor, bias_tensor=None,
                               stride=(1,1), padding=(0,0)):
    """
    Compute int8 convolution using optimized triton kernel
    
    Args:
        input_tensor: input tensor of shape (N, C, H, W)
        weight_tensor: weight tensor of shape (K, C, R, S)
        bias_tensor: optional bias tensor of shape (K,)
        stride: tuple of (stride_h, stride_w)
        padding: tuple of (padding_h, padding_w)
    Returns:
        output tensor of shape (N, K, H', W')
    """
    assert input_tensor.is_cuda and weight_tensor.is_cuda
    if bias_tensor is not None:
        assert bias_tensor.is_cuda and bias_tensor.dtype == torch.int32
    
    # Extract dimensions
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_size, _ = weight_tensor.shape
    
    # Compute output dimensions
    out_height = (in_height + 2 * padding[0] - kernel_size) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_size) // stride[1] + 1
    
    # Allocate output
    output = torch.empty((batch_size, out_channels, out_height, out_width),
                        device=input_tensor.device, dtype=torch.int32)
    
    # Determine optimal block size based on dimensions
    if out_channels >= 256:
        block_size_n = 32  # Process 32 output channels at once
    else:
        block_size_n = 16
        
    if in_channels >= 256:
        block_size_ic = 32  # Process input channels in blocks of 32
    else:
        block_size_ic = 16
    
    # Configure grid
    grid = (batch_size * out_height, (out_channels + block_size_n - 1) // block_size_n, out_width)
    
    # Launch kernel
    _conv2d_int8_kernel[grid](
        input_tensor, weight_tensor,
        bias_tensor, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride[0], padding[0],
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_K=1, BLOCK_SIZE_IC=block_size_ic,
    )
    
    return output

# Add this to your if __name__ == '__main__' block to test with your specific dimensions:
def run_benchmarks():
    device = 'cuda'
    
    # Test case 1
    N1, C1, H1, W1 = 10, 512, 4, 4
    K1, R1, S1 = 512, 3, 3
    input1 = torch.randint(-128, 127, (N1, C1, H1, W1), dtype=torch.int32, device=device)
    weight1 = torch.randint(-128, 127, (K1, C1, R1, S1), dtype=torch.int32, device=device)
    bias1 = torch.randint(-100, 100, (K1,), dtype=torch.int32, device=device)
    
    # Test case 2
    N2, C2, H2, W2 = 10, 256, 10, 10
    K2, R2, S2 = 256, 3, 3
    input2 = torch.randint(-128, 127, (N2, C2, H2, W2), dtype=torch.int32, device=device)
    weight2 = torch.randint(-128, 127, (K2, C2, R2, S2), dtype=torch.int32, device=device)
    bias2 = torch.randint(-100, 100, (K2,), dtype=torch.int32, device=device)

    # Benchmark setup
    import time
    
    def benchmark(fn, *args, warmup=5, repeat=10, **kwargs):
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(repeat):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        
        return (end - start) / repeat
    
    print("=== Test Case 1: 10x512x4x4 input, 512x512x3x3 weight ===")
    orig_time = benchmark(torch.nn.functional.conv2d, 
                        input1.float(), weight1.float(), bias1.float() if bias1 is not None else None, 
                        stride=(1,1), padding=(1,1))
    opt_time = benchmark(triton_conv2d_int8, input1, weight1, bias1, stride=(1,1), padding=(1,1))
    print(f"Original: {orig_time*1000:.3f} ms")
    print(f"Optimized: {opt_time*1000:.3f} ms")
    print(f"Speedup: {orig_time/opt_time:.2f}x")

    print("\n=== Test Case 2: 10x256x10x10 input, 256x256x3x3 weight ===")
    orig_time = benchmark(torch.nn.functional.conv2d, 
                        input2.float(), weight2.float(), bias2.float() if bias2 is not None else None, 
                        stride=(1,1), padding=(1,1))
    opt_time = benchmark(triton_conv2d_int8, input2, weight2, bias2, stride=(1,1), padding=(1,1))
    print(f"Original: {orig_time*1000:.3f} ms")
    print(f"Optimized: {opt_time*1000:.3f} ms")
    print(f"Speedup: {orig_time/opt_time:.2f}x")

import triton
import triton.language as tl
import torch

@triton.jit
def int8_linear_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Strides
    input_stride_batch, input_stride_feat,
    weight_stride_out, weight_stride_in,
    output_stride_batch, output_stride_feat,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block start indices
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Create offsets for the blocks
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Iterate over k-dimension in blocks
    for k in range(0, tl.cdiv(in_features, BLOCK_K)):
        k_start = k * BLOCK_K
        
        # Load input block - KEEP AS INT8
        a_ptrs = input_ptr + offs_m[:, None] * input_stride_batch + (offs_k[None, :] + k_start) * input_stride_feat
        a_mask = (offs_k[None, :] + k_start) < in_features
        a = tl.load(a_ptrs, mask=a_mask, other=0)  # Don't convert to int32 here
        
        # Load weight block - KEEP AS INT8
        b_ptrs = weight_ptr + offs_n[:, None] * weight_stride_out + (offs_k[None, :] + k_start) * weight_stride_in
        b_mask = (offs_k[None, :] + k_start) < in_features
        b = tl.load(b_ptrs, mask=b_mask, other=0)  # Don't convert to int32 here
        
        # Matrix multiply using int32 accumulation - specify out_dtype as int32
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.int32)
    
    # Add bias if provided
    if bias_ptr:
        bias_ptrs = bias_ptr + offs_n
        bias_mask = offs_n < out_features
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0)
        acc = acc + bias[None, :]
    
    # Write output with masking
    out_ptrs = output_ptr + offs_m[:, None] * output_stride_batch + offs_n[None, :] * output_stride_feat
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=out_mask)

def triton_linear_int8(input, weight, bias=None):
    """
    Int8 linear operation implementation using Triton
    
    Args:
        input: int8 tensor of shape [batch_size, in_features]
        weight: int8 tensor of shape [out_features, in_features]
        bias: int32 tensor of shape [out_features] or None
        
    Returns:
        int32 tensor of shape [batch_size, out_features]
    """
    batch_size, in_features = input.shape
    out_features, in_features_w = weight.shape
    assert in_features == in_features_w, "Input and weight feature dimensions must match"
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), device=input.device, dtype=torch.int32)
    
    # Block sizes (can be tuned for specific hardware)
    BLOCK_M = 16  # batch dimension
    BLOCK_N = 16  # output features dimension  
    BLOCK_K = 128  # input features dimension
    
    # Compute grid dimensions
    grid = (triton.cdiv(batch_size, BLOCK_M), triton.cdiv(out_features, BLOCK_N))
    
    # Launch kernel
    int8_linear_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else 0,
        output_ptr=output,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        input_stride_batch=input.stride(0),
        input_stride_feat=input.stride(1),
        weight_stride_out=weight.stride(0),
        weight_stride_in=weight.stride(1),
        output_stride_batch=output.stride(0),
        output_stride_feat=output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output
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
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Batch and output channel indices
    batch_idx = pid // (out_channels * out_height * out_width)
    tmp = pid % (out_channels * out_height * out_width)
    oc = tmp // (out_height * out_width)
    h_out = (tmp // out_width) % out_height
    w_out = tmp % out_width
    
    # Compute input indices
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding
    
    # Initialize accumulator
    acc = 0
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + oc).to(tl.int32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(in_channels):
        for kh in range(kernel_size):
            h_in = h_in_start + kh
            if (0 <= h_in) & (h_in < in_height):
                for kw in range(kernel_size):
                    w_in = w_in_start + kw
                    if (0 <= w_in) & (w_in < in_width):
                        # Load input value
                        in_idx = ((batch_idx * in_channels + ic) * in_height + h_in) * in_width + w_in
                        in_val = tl.load(input_ptr + in_idx)
                        
                        # Load weight value
                        weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw
                        weight_val = tl.load(weight_ptr + weight_idx)
                        
                        # Accumulate product
                        acc += in_val * weight_val
    
    # Write output
    out_idx = ((batch_idx * out_channels + oc) * out_height + h_out) * out_width + w_out
    tl.store(output_ptr + out_idx, acc)

def triton_conv2d_int8(input_tensor, weight_tensor, bias_tensor=None,
                      stride=(1,1), padding=(0,0)):
    """
    Compute int8 convolution using triton
    
    Args:
        input_tensor: input tensor of shape (N, C, H, W) with dtype int8
        weight_tensor: weight tensor of shape (K, C, R, S) with dtype int8
        bias_tensor: optional bias tensor of shape (K,) with dtype int32
        stride: tuple of (stride_h, stride_w)
        padding: tuple of (padding_h, padding_w)
    Returns:
        output tensor of shape (N, K, H', W') with dtype int32
    """
    assert input_tensor.is_cuda and weight_tensor.is_cuda
    # assert input_tensor.dtype == torch.int8 and weight_tensor.dtype == torch.int8
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
    
    # Configure grid and block sizes
    grid = (batch_size * out_channels * out_height * out_width,)
    
    # Launch kernel
    _conv2d_int8_kernel[grid](
        input_tensor, weight_tensor,
        bias_tensor, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride[0], padding[0],
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16,
    )
    
    return output

# Example Usage and Testing
if __name__ == '__main__':
    # Example usage and testing
    device = 'cuda'
    N, C, H, W = 1, 3, 32, 32
    K, R, S = 8, 3, 3
    stride = (2, 2)
    padding = (1, 1)

    input_tensor = torch.randint(-128, 127, (N, C, H, W), dtype=torch.int32, device=device)
    weight_tensor = torch.randint(-128, 127, (K, C, R, S), dtype=torch.int32, device=device)
    bias_tensor = torch.randint(-100, 100, (K,), dtype=torch.int32, device=device)

    # PyTorch's conv2d needs float input, convert int8 to float32
    input_float = input_tensor.float()
    weight_float = weight_tensor.float()

    output_torch = F.conv2d(input_float, weight_float, bias_tensor.float(), stride=stride, padding=padding).int()  # Convert to int after conv

    output_triton = triton_conv2d_int8(input_tensor, weight_tensor, bias_tensor, stride=stride, padding=padding)


    # Compare the results
    print("Max diff:", torch.max(torch.abs(output_torch - output_triton)))
    print("Mean diff:", torch.mean(torch.abs(output_torch - output_triton).float()))  # Convert to float for accurate mean calculation
    print("Triton Output:\n", output_triton)
    print("Pytorch Output:\n", output_torch)


    #Check if they match closely (within a tolerance)
    tolerance = 1e-5
    all_close = torch.allclose(output_torch.float(), output_triton.float(), rtol=tolerance, atol=tolerance)
    print("All close:", all_close)

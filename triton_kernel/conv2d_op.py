

import torch
import triton
import triton.language as tl
import pytest

dtype = tl.int32


def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_fwd_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, 
    N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (P * Q)
    npq_residual = gemm_i % (P * Q)
    p = npq_residual // Q
    q = npq_residual % Q
    k = gemm_j

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c = gemm_k // (R * S)
        crs_residual = gemm_k % (R * S)
        r = crs_residual // S
        s = crs_residual % S
        
        # triton broadcast rules is same as numpy
        # p: [BLOCK_SIZE_M], p[:, None]: [BLOCK_SIZE_M, 1]
        # r: [BLOCK_SIZE_K], r[None, :]: [1, BLOCK_SIZE_K]
        # h: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        h = p[:, None] * str_h + r[None, :] * dil_h - pad_h
        # q: [BLOCK_SIZE_M], q[:, None]: [BLOCK_SIZE_M, 1]
        # s: [BLOCK_SIZE_K], s[None, :]: [1, BLOCK_SIZE_K]
        # w: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        w = q[:, None] * str_w + s[None, :] * dil_w - pad_w

        mask_input = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_weight = (r[:, None] < R) & (s[:, None] < S) & (c[:, None] < C)

        # n: [BLOCK_SIZE_M], n[:, None]: [BLOCK_SIZE_M, 1]
        # c: [BLOCK_SIZE_K], c[None, :]: [1, BLOCK_SIZE_K]
        # h: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # w: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # offs_input: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        offs_input = n[:, None] * C * H * W + c[None, :] * H * W + h * W + w
        # k: [BLOCK_SIZE_N], k[None, :]: [1, BLOCK_SIZE_N]
        # c: [BLOCK_SIZE_K], c[:, None]: [BLOCK_SIZE_K, 1]
        # r: [BLOCK_SIZE_K], r[:, None]: [BLOCK_SIZE_K, 1]
        # s: [BLOCK_SIZE_K], s[:, None]: [BLOCK_SIZE_K, 1]
        # offs_weight: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        offs_weight = k[None, :] * C * R * S + c[:, None] * R * S + r[:, None] * S + s[:, None]

        input_ptrs = input_ptr + offs_input
        weight_ptrs = weight_ptr + offs_weight
        
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        acc = tl.dot(input_data, weight_data, acc)

    if bias_ptr is not None:
        # k: [BLOCK_SIZE_N], k[None, :]: [1, BLOCK_SIZE_N]
        # offs_bias: [1, BLOCK_SIZE_N]
        offs_bias = k[None, :]
        bias_ptrs = bias_ptr + offs_bias
        bias_data = tl.load(bias_ptrs)
        acc = acc + bias_data

    # has three store methods, [GEMM_M, GEMM_N], [N, P, Q, K], [N, K, P, Q]

    # 1. store to output [N*P*Q, K] [GEMM_M, GEMM_N]
    # # gemm_i: [BLOCK_SIZE_M], gemm_i[:, None]: [BLOCK_SIZE_M, 1]
    # # gemm_j: [BLOCK_SIZE_N], gemm_j[None, :]: [1, BLOCK_SIZE_N]
    # # offs_output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # offs_output = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    # mask_output = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)

    # output_ptrs = output_ptr + offs_output
    # tl.store(output_ptrs, acc, mask=mask_output)

    # # 2. store to output [N, P, Q, K]
    # # n: [BLOCK_SIZE_M], n[:, None]: [BLOCK_SIZE_M, 1]
    # # k: [BLOCK_SIZE_N], k[None, :]: [1, BLOCK_SIZE_N]
    # # p: [BLOCK_SIZE_M], p[:, None]: [BLOCK_SIZE_M, 1]
    # # q: [BLOCK_SIZE_M], q[:, None]: [BLOCK_SIZE_M, 1]
    # # offs_output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # offs_npqk = n[:, None] * P * Q * K + p[:, None] * Q * K + q[:, None] * K + k[None, :]
    # mask_npqk = (n[:, None] < N) & (p[:, None] < P) & (q[:, None] < Q) & (k[None, :] < K)
    # output_ptrs = output_ptr + offs_npqk
    # tl.store(output_ptrs, acc, mask=mask_npqk)

    # 3. store to output [N, K, P, Q]
    # n: [BLOCK_SIZE_M], n[:, None]: [BLOCK_SIZE_M, 1]
    # k: [BLOCK_SIZE_N], k[None, :]: [1, BLOCK_SIZE_N]
    # p: [BLOCK_SIZE_M], p[:, None]: [BLOCK_SIZE_M, 1]
    # q: [BLOCK_SIZE_M], q[:, None]: [BLOCK_SIZE_M, 1]
    # offs_output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_nkpq = n[:, None] * K * P * Q + k[None, :] * P * Q + p[:, None] * Q + q[:, None]
    mask_nkpq = (n[:, None] < N) & (k[None, :] < K) & (p[:, None] < P) & (q[:, None] < Q)

    output_ptrs = output_ptr + offs_nkpq
    tl.store(output_ptrs, acc, mask=mask_nkpq)
        

def _implicit_gemm_conv2d_fwd(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride, padding, dilation):
    N, C, H, W = input.shape
    K, C, R, S = weight.shape
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    
    # # 1. store to output [N*P*Q, K] [GEMM_M, GEMM_N]
    # output = torch.zeros((GEMM_M, GEMM_N), dtype=input.dtype, device=input.device) # [GEMM_M, GEMM_N] [N*P*Q, K]

    # # 2. store to output [N, P, Q, K]
    # output = torch.zeros((N, P, Q, K), dtype=input.dtype, device=input.device) # [N, P, Q, K]

    # 3. store to output [N, K, P, Q]
    output = torch.zeros((N, K, P, Q), dtype=input.dtype, device=input.device) # [N, K, P, Q]
    
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    debug = False
    if debug:
        pgm = _implicit_gemm_conv2d_fwd_kernel[grid](
            output, input, weight, bias, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
        ttir = pgm.asm['ttir']
        ttgir = pgm.asm['ttgir']
        llir = pgm.asm['llir']
        ptx = pgm.asm['ptx']
        print(f'ttir: {ttir}')
        print(f'ttgir: {ttgir}')
        print(f'llir: {llir}')
        print(f'ptx: {ptx}')
    else:
        _implicit_gemm_conv2d_fwd_kernel[grid](
            output, input, weight, bias, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )

    # # 1. store to output [N*P*Q, K] [GEMM_M, GEMM_N]
    # output = output.view(N, P, Q, K).permute(0, 3, 1, 2).contiguous() # [N*P*Q, K] -> [N, K, P, Q]

    # # 2. store to output [N, P, Q, K]
    # output = output.permute(0, 3, 1, 2).contiguous() # [N, P, Q, K] -> [N, K, P, Q]

    # 3. store to output [N, K, P, Q]
    return output


class _triton_conv2d_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                stride: int = (1, 1),
                padding: int = (0, 0),
                dilation: int = (1, 1)
    ):
        output = _implicit_gemm_conv2d_fwd(input, weight, bias, stride, padding, dilation)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return output
    

triton_conv2d = _triton_conv2d_func.apply
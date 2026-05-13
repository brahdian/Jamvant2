import torch
import triton
import triton.language as tl

# --- Triton Parallel Scan Kernel ---
# This kernel parallelizes the SSM recurrence: h_t = a_t * h_{t-1} + b_t * x_t
@triton.jit
def _mamba_scan_kernel(
    X, A_BAR, B_BAR, OUT,
    B, L, D, S,
    stride_xb, stride_xl, stride_xd,
    stride_ab, stride_al, stride_ad, stride_as,
    stride_bb, stride_bl, stride_bd, stride_bs,
    stride_ob, stride_ol, stride_od,
    BLOCK_D: tl.constexpr,
):
    # Parallelize over Batch and Dimension
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets for the current (batch, dim)
    x_ptr = X + pid_b * stride_xb + pid_d * stride_xd
    a_ptr = A_BAR + pid_b * stride_ab + pid_d * stride_ad
    b_ptr = B_BAR + pid_b * stride_bb + pid_d * stride_bd
    out_ptr = OUT + pid_b * stride_ob + pid_d * stride_od

    # State h is (S,) for each (batch, dim)
    # For simplicity in this PoC kernel, we assume d_state (S) is small and we reduce it
    h = tl.zeros((1,), dtype=tl.float32)

    for t in range(L):
        xt = tl.load(x_ptr + t * stride_xl)
        at = tl.load(a_ptr + t * stride_al) # Simplified A is scalar for this kernel example
        bt = tl.load(b_ptr + t * stride_bl)

        h = at * h + bt * xt
        tl.store(out_ptr + t * stride_ol, h)

class MambaScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_bar, b_bar):
        # x: (B, L, D)
        # a_bar: (B, L, D) - simplified for kernel
        # b_bar: (B, L, D) - simplified for kernel
        B, L, D = x.shape
        out = torch.empty_like(x)

        grid = (B, D)
        _mamba_scan_kernel[grid](
            x, a_bar, b_bar, out,
            B, L, D, 1,
            x.stride(0), x.stride(1), x.stride(2),
            a_bar.stride(0), a_bar.stride(1), a_bar.stride(2), 0,
            b_bar.stride(0), b_bar.stride(1), b_bar.stride(2), 0,
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=1
        )
        ctx.save_for_backward(x, a_bar, b_bar, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Simplification: return zeros for backward in this PoC
        return torch.zeros_like(ctx.saved_tensors[0]), torch.zeros_like(ctx.saved_tensors[1]), torch.zeros_like(ctx.saved_tensors[2])

def triton_mamba_scan(x, a_bar, b_bar):
    return MambaScanFunction.apply(x, a_bar, b_bar)

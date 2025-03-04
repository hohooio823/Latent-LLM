import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.rope import LigerRopeFunction
from torch.nn import CrossEntropyLoss
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction

class LigerRMSNorm(nn.Module):
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones"
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon, self.offset, self.casting_mode = (
            eps,
            offset,
            casting_mode,
        )

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"
    

class LigerSwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):

        return self.down_proj(
            LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )
        
# class FeedForward(nn.Module):
#     def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
#         super().__init__()
#         if hidden_dim is None:
#             hidden_dim = 4 * dim
#             hidden_dim = int(2 * hidden_dim / 3)
#             hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
#         self.w1 = nn.Linear(dim, hidden_dim, bias=False)
#         self.w2 = nn.Linear(hidden_dim, dim, bias=False)
#         self.w3 = nn.Linear(dim, hidden_dim, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))




def liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """

    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)




class LigerCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerCrossEntropyLoss, self).__init__(*args, **kwargs)
        assert (self.label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}"
        assert self.reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {self.reduction}"

    def forward(self, _input, target):
        return LigerCrossEntropyFunction.apply(
            _input, target, self.ignore_index, self.label_smoothing, self.reduction
        )

class LigerLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, bias=False, init_fn="ones"):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.bias = nn.Parameter(
            torch.randn(hidden_size) if bias else torch.zeros(hidden_size)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return LigerLayerNormFunction.apply(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"
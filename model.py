import torch
from torch import nn
from torch.nn import functional as F

from utils import Args

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

    

class GQA(nn.Module):
    """
    A grouped query attention module
    """
    def __init__(self, n_heads = 8, n_kv_heads = 2, n_embd = 128, block_size = 32, device: str = 'cpu') -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_embd = n_embd
        self.block_size = block_size
        self.device = device

        assert self.n_embd % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

        self.head_dim = self.n_embd // self.n_heads

        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                            .view(1, 1, block_size, block_size))
        
    def forward(self, x: torch.Tensor):
        """
        x: input tensor (batch_size, seqlen, n_embd)
        """
        batch_size, seqlen, _ = x.size()

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(batch_size, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.n_kv_heads, self.head_dim)

        # repeat key and value 
        n_repeats = self.n_heads // self.n_kv_heads
        xk = torch.repeat_interleave(xk, repeats=n_repeats, dim=2).transpose(1,2) 
        xv = torch.repeat_interleave(xv, repeats=n_repeats, dim=2).transpose(1,2)

        xq = xq.transpose(1,2)

        attn = (xq @ xk.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = attn.masked_fill(self.mask[:, :, :seqlen, :seqlen] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, xv)  # (bs, n_local_heads, slen, head_dim)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)
        return self.o_proj(output)
    

class MoeLayer(nn.Module):
    def __init__(self,
        num_experts: int,
        num_experts_per_tok: int,
        n_embd: int):
        
        super().__init__()
        assert num_experts > 0

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.experts = nn.ModuleList(
            [nn.Linear(n_embd, n_embd, bias=False) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, inputs: torch.Tensor):
        B, T, C = inputs.shape
        inputs = inputs.view(B*T, C)

        gate_logits = self.gate(inputs)

        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)

        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results.view(B, T, C)
    

class TransformerBlock(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_embd = args.n_embd
        self.attention = GQA(args.n_heads, args.n_kv_heads, args.n_embd, args.block_size, device=args.device)
        self.feed_forward = MoeLayer(
           args.num_experts,
           args.num_experts_per_tok,
           args.n_embd,
        )

        self.attention_norm = RMSNorm(args.n_embd, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.n_embd, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class MOE(nn.Module):
    """
    This class represents the whole MOE model
    """
    def __init__(self, args: Args) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.device = args.device
        self.block_size = args.block_size
        self.n_embd = args.n_embd
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.pos_embeddings = nn.Embedding(args.block_size, args.n_embd)
        self.layers = nn.ModuleDict({str(i): TransformerBlock(args=args) for i in range(args.n_layers)})
        self.norm = RMSNorm(args.n_embd, args.norm_eps)
        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        _, seqlen = input_ids.shape

        h = self.tok_embeddings(input_ids)

        pos = torch.arange(0, seqlen, dtype=torch.long, device=self.device).unsqueeze(0) # shape (1, t)
        pos_emb = self.pos_embeddings(pos)
        h = h + pos_emb

        for layer in self.layers.values():
            h = layer(h)   

        out = self.output(self.norm(h))
        return out
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample: bool = False, top_k = None):
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":

    args = Args.get_args()

    a = torch.ones(32, 6, args.n_embd)

    #gqa = GQA(args.n_heads, args.n_kv_heads, args.n_embd, args.block_size, args.device)

    model = MOE(args)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    out = model(input_ids)
    print(out.shape)

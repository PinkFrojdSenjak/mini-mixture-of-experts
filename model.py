import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional
import math
import tiktoken
from utils import Args


class PositionalEncoding(nn.Module):

    def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(max_len, 1, n_embd)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [B, T, C]
        """
        x = x.transpose(0, 1) # (T, B, C)
        x = x + self.pe[:x.size(0)]
        x =  self.dropout(x)
        return x.transpose(0, 1) # (B, T, C)
    

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
    

class CausalSelfAttention(nn.Module):

    def __init__(self, n_head = 8, n_embd = 128, block_size = 32, pos_dropout: float = 0.1) -> None:
        """
        n_head: number of heads in the multihead attention
        n_embd: Dimension of the embeddings. Head dimension will be equal to n_embd // n_head
        block_size: Number of tokens in the block, a.k.a. context length
        """
        super().__init__()
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.pos_encoder = PositionalEncoding(n_embd, pos_dropout)

        assert self.n_embd % self.n_head == 0

        # Query, Key, Value
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)

        # Output
        self.wo = nn.Linear(n_embd, n_embd)

        # Dropouts
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)

        # Causal mask, to avoid attending to future context
        # buffers are non-trainable parameters
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                            .view(1, 1, block_size, block_size))
        

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, block_size, n_embd)
        """
            
        B, T, C = x.size()

        # Query, Key, Value
        q = self.wq(x) # (B, T, C)
        q = self.pos_encoder(q)
        q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) # (B, n_head, T, C // n_head)

        k = self.wk(x) # (B, T, C)
        k = self.pos_encoder(k)
        k = k.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) # (B, n_head, T, C // n_head)

        v = self.wv(x) # (B, T, C)
        v = v.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) # (B, n_head, T, C // n_head)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Context
        y = attn @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # Output
        y = self.wo(y)
        y = self.resid_drop(y)

        return y
    

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(x)
    
    

class MoeLayer(nn.Module):
    def __init__(self,
        num_experts: int,
        num_experts_per_tok: int,
        n_embd: int,
        ff_hidden_dim: int,
    ):
        
        super().__init__()
        assert num_experts > 0

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.experts = nn.ModuleList(
            [FeedForward(n_embd, ff_hidden_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])

        gate_logits = self.gate(inputs_squashed)

        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)
    

class TransformerBlock(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.attention = CausalSelfAttention(args.n_head, args.n_embd, args.block_size)
        self.moe_layer = MoeLayer(
           args.num_experts,
           args.num_experts_per_tok,
           args.n_embd,
           args.ff_hidden_dim
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
        r = self.moe_layer.forward(self.ffn_norm(h))
        out = h + r
        return out



class PositionalEncoding(nn.Module):

    def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(max_len, 1, n_embd)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [B, T, C]
        """
        x = x.transpose(0, 1) # (T, B, C)
        x = x + self.pe[:x.size(0)]
        x =  self.dropout(x)
        return x.transpose(0, 1) # (B, T, C)
    

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
    

class CausalSelfAttention(nn.Module):

    def __init__(self, n_head = 8, n_embd = 128, block_size = 32, pos_dropout: float = 0.1) -> None:
        """
        n_head: number of heads in the multihead attention
        n_embd: Dimension of the embeddings. Head dimension will be equal to n_embd // n_head
        block_size: Number of tokens in the block, a.k.a. context length
        """
        super().__init__()
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.pos_encoder = PositionalEncoding(n_embd, pos_dropout)

        assert self.n_embd % self.n_head == 0

        # Query, Key, Value
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)

        # Output
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

        # Dropouts
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)

        # Causal mask, to avoid attending to future context
        # buffers are non-trainable parameters
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                            .view(1, 1, block_size, block_size))
        

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, block_size, n_embd)
        """
            
        B, T, C = x.size()

        # Query, Key, Value
        q = self.wq(x) # (B, T, C)
        q = self.pos_encoder(q)
        q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) # (B, n_head, T, C // n_head)

        k = self.wk(x) # (B, T, C)
        k = self.pos_encoder(k)
        k = k.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) # (B, n_head, T, C // n_head)

        v = self.wv(x) # (B, T, C)
        v = v.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) # (B, n_head, T, C // n_head)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Context
        y = attn @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # Output
        y = self.wo(y)
        y = self.resid_drop(y)

        return y
    

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(x)
    
    

class MoeLayer(nn.Module):
    def __init__(self,
        num_experts: int,
        num_experts_per_tok: int,
        n_embd: int,
        ff_hidden_dim: int,
    ):
        
        super().__init__()
        assert num_experts > 0

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.experts = nn.ModuleList(
            [nn.Linear(n_embd, n_embd, bias=False) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])

        gate_logits = self.gate(inputs_squashed)

        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)
    

class TransformerBlock(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.attention = CausalSelfAttention(args.n_head, args.n_embd, args.block_size)
        self.feed_forward = MoeLayer(
           args.num_experts,
           args.num_experts_per_tok,
           args.n_embd,
           args.ff_hidden_dim
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
        self.layers = nn.ModuleDict({str(i): TransformerBlock(args=args) for i in range(args.n_layers)})
        self.norm = RMSNorm(args.n_embd, args.norm_eps)
        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape

        h = self.tok_embeddings(input_ids)

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
    

    tokenizer = tiktoken.get_encoding("cl100k_base")
    args = Args(
            dataset_path='data/tiny_shakespeare.txt',
            batch_size=16,
            learning_rate = 1e-3,
            max_iters = 5000,
            eval_iters=100,
            test_size=0.1,
            n_head=8,
            n_embd=128,
            block_size=32,
            ff_hidden_dim=256,
            num_experts=8,
            num_experts_per_tok=2,
            norm_eps=1e-6,
            vocab_size=tokenizer.n_vocab,
            device = 'cpu',
            n_layers=4
        )


    model = MOE(args)

    out = model(torch.ones(16,32, dtype=torch.int32))

    

    x = torch.tensor(tokenizer.encode('Today I will '), dtype = torch.long).to(args.device)
    x = x.expand(1, x.shape[0])
    
    
    y = model.generate(x, max_new_tokens=10, do_sample=True, top_k=40)

    print(tokenizer.decode(list(y[0])))

    pass
from dataclasses import dataclass

@dataclass
class Args:
    dataset_path: str
    batch_size: int
    learning_rate: float
    max_iters: int
    eval_iters: int
    eval_interval: int
    test_size: float
    n_head: int
    n_embd: int
    block_size: int
    ff_hidden_dim: int
    num_experts: int
    num_experts_per_tok: int
    norm_eps: int
    vocab_size: int
    device: str
    n_layers: int
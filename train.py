import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import tiktoken

from model import MOE
from utils import Args
import time


tokenizer = tiktoken.get_encoding("cl100k_base")

args = Args(
            dataset_path='data/tiny_shakespeare.txt',
            batch_size=16,
            learning_rate = 1e-3,
            max_iters = 5000,
            eval_iters=200,
            eval_interval=100,
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


with open(args.dataset_path, encoding='utf-8') as f:
    s = f.read()


data = tokenizer.encode(s)
data = torch.tensor(data, dtype=torch.long)

n = int((1 - args.test_size) * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    assert split == 'train' or split == 'val'
    
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.block_size] for i in ix])
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix])
    x, y = x.to(args.device), y.to(args.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = loss_fn(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = MOE(args)
m = model.to(args.device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')



optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
loss_fn = F.cross_entropy

start_time = time.time()
for iter in range(args.max_iters):

    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        end_time = time.time()
        print(f'Elapsed time: {end_time - start_time}')
        start_time = end_time

    xb, yb = get_batch('train')

    logits= model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    yb = yb.view(B*T)
    loss = loss_fn(logits, yb)
    

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
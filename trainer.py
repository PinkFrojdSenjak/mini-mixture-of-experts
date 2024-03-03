import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import MOE
from utils import Args, Tokenizer
import time
import os 
import wandb


args = Args.get_args()
tokenizer = Tokenizer(args)

'''if args.use_wandb:
    wandb.init(entity = 'pavlepadjin', project = 'mini-mixture-of-experts')
    wandb.watch_called = False
    
    wconfig = wandb.config          
    wconfig.batch_size = args.batch_size
    wconfig.learning_rate = args.learning_rate
    wconfig.test_size = args.test_size
    wconfig.n_head = args.n_head
    wconfig.n_embd = args.n_embd
    wconfig.block_size = args.block_size
    wconfig.num_experts = args.num_experts
    wconfig.num_experts_per_tok = args.num_experts_per_tok
    wconfig.vocab_size = args.vocab_size
    wconfig.device = args.device'''



def train(model: nn.Module, args: Args):

    with open(args.dataset_path, encoding='utf-8') as f:
        text = f.read()

    data = tokenizer.encode(text)
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


    start_iter = 0
    end_iter = args.max_iters

    if args.load_checkpoint:
        l = os.listdir(args.models_dir)
        
        nums = [int(name[6:-3]) for name in l if name.startswith('model')] # hack to extract the iteration counts for models
        if nums:
            start_iter = max(nums)
            end_iter = args.max_iters + start_iter

            weights = torch.load(f"{args.models_dir}/model-{start_iter}.pt")
            model.load_state_dict(weights)
            print(f'Loaded model checkpoint at iteration {start_iter}...')
        else:
            print('No checkpoints to load, training the model from scratch')


    start_time = time.time()
    for iter in range(end_iter):

        if (iter+1) % args.eval_interval == 0 or iter == end_iter - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            end_time = time.time()
            print(f'Elapsed time: {end_time - start_time}')
            start_time = end_time

        if (iter+1) % args.model_save_interval == 0 or iter == end_iter - 1:
            path = os.path.join(args.models_dir, f"model-{iter+1}.pt" )
            torch.save(model.state_dict(), path)

        xb, yb = get_batch('train')

        logits= model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        yb = yb.view(B*T)
        loss = loss_fn(logits, yb)
        

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

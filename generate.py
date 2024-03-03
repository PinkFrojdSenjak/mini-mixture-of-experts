import torch 

from model import MOE
from utils import Args, Tokenizer, get_latest_checkoint


def generate(prompt = '', args: Args = None, max_new_tokens = 200, temperature: float = 1.0, top_k: int = 40):
        
    if args is None: args = Args.get_default_args()

    tokenizer = Tokenizer(args)
    model = MOE(args)    
    model, _ = get_latest_checkoint(model, args)

    if prompt == '': prompt = '\n' # as close to empty prompt as possible
    
    input_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    input_tokens = input_tokens.unsqueeze(0).to(args.device)

    toks = model.generate(input_tokens, max_new_tokens = 200, temperature = 30)

    s = tokenizer.decode([int(el) for el in toks[0]])
    return s

if __name__ == "__main__":
    print(generate('ROMEO:', temperature=0.1))
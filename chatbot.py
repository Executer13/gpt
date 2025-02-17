import torch
import torch.nn as nn
from torch.nn import functional as F 
import mmap
import random
import pickle 
import argparse

parser = argparse.ArgumentParser(description = 'This is a deomnstration Program')
parser.add_argument('-bs',type=str, required = True, help = 'Batch Size' )
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'batch size:{args.batch_size}')
print(device)



block_size= 64 
batch_size = args.batch_size
learning_rate = 3e-4
max_iters = 200
eval_iters = 250
dropout = 0.2
n_embd = 384
n_layer = 8
n_head = 8
dropout  = 0.2


chars = ""
with open('openwebtext/vocab.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

print(chars)
print(len(chars))
vocab_size = len(chars)


class FeedForward(nn.Module):
     def __init__(self, n_embd):
          super().__init__()
          self.net = nn.Sequential(
               nn.Linear(n_embd, 4 * n_embd),
               nn.ReLU(),
               nn.Linear(4 * n_embd, n_embd),
               nn.Dropout(dropout),
          )

     def forward(self, x):
          return self.net(x)
          

class Head (nn.Module):
     def __init__(self,  head_size ):
          super().__init__()
          self.key = nn.Linear(n_embd, head_size, bias = False)
          self.query = nn.Linear(n_embd, head_size, bias = False)
          self.value = nn.Linear(n_embd, head_size, bias =False)
          self.register_buffer('trl', torch.tril(torch.ones(block_size, block_size)))

          self.dropout = nn.Dropout(dropout)

     def forward(self, x):
          B,T,C = x.shape
          k = self.key(x)
          q = self.query(x)
          wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
          wei = wei.masked_fill(self.trl[:T, :T] == 0, float('-inf'))
          wei = F.softmax(wei, dim = -1)

          v = self.value(x)
          out = wei @ v

          return out
     

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size): 
          super().__init__()
          self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
          self.proj = nn.Linear(head_size * num_heads, n_embd, bias = False)
          self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.concat([head(x) for head in self.heads], dim =-1) #(B,T,F )---> ([h1, h1, h1, h2, h2, h2, h3, h3, h3, h4, h4, h4])
        out = self.dropout(self.proj(out))
        return out
          


class Block(nn.Module):
     def __init__(self, n_embd, n_head):
          super().__init__()
          head_size = n_embd // n_head
          self.sa = MultiHeadAttention(n_head, head_size)
          
          self.ffwd  = FeedForward(n_embd)
          self.ln1 = nn.LayerNorm(n_embd)
          self.ln2 = nn.LayerNorm(n_embd)

     def forward(self, x):
          y = self.sa(x)
          x = self.ln1(x+y)
          y = self.ffwd(x)
          x = self.ln2(x+y)
          
          return x


class GPTLanguageModel(nn.Module):
       def __init__(self, vocab_size):
           super().__init__()
           self.token_embeddings = nn.Embedding(vocab_size, n_embd)
           self.position_embeddings_table = nn.Embedding(block_size, n_embd)
           self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

           self.ln_f = nn.LayerNorm(n_embd)
           self.lm_head = nn.Linear(n_embd, vocab_size)
           self.apply(self._init_weights)

       def _init_weights(self, module):
           if isinstance(module, nn.Linear):
               torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
               if module.bias is not None:
                   torch.nn.init.zeros_(module.bias)
           elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean = 0.0, std =  0.02)
                   
       def forward(self, index, targets=None):
           B, T = index.shape
           token_emb = self.token_embeddings(index)
           pos_emb = self.position_embeddings_table(torch.arange(T, device=device))
           x = token_emb + pos_emb
           x = self.blocks(x)
           x = self.ln_f(x)
           logits = self.lm_head(x)


           # logits has shape (B, T, C) where C == vocab_size
           if targets is None:
               return logits, None

           B, T, C = logits.shape
           logits = logits.view(B * T, C)
           targets = targets.view(B * T)
           loss = F.cross_entropy(logits, targets)
           return logits, loss      

       def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop the context to block_size if it is longer.
            index_cond = index if index.shape[1] <= block_size else index[:, -block_size:]
            logits, _ = self(index_cond)
            # Focus on predictions for the last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index


model = GPTLanguageModel(vocab_size)
print('loading learnable parameters')
with open('model-01.pkl', 'rb') as f:
    pickle.load(f)
print('model loaded')
m = model.to(device)


while True: 
     prompt = input("Prompt: \n")
     context = torch.tensor(encode(prompt), dtype =torch.long, device=device)
     generated_chars = decode(m.generate(context.unsqueeze(0),max_new_tokens= 150)[0].tolist())
     print(f'Completion: \n{generated_chars}')
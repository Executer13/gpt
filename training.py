#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn import functional as F 
import mmap
import random
import pickle 
import argparse

from chatbot import GPTLanguageModel
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


# In[75]:


chars = ""
with open('openwebtext/vocab.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

print(chars)
print(len(chars))
vocab_size = len(chars)


# In[36]:


string_to_int = { ch:i for i,ch in enumerate(chars)}
int_to_string = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])



# In[77]:


def get_random_chunk(split):
    filename = "openwebtext/train_split.txt" if split == 'train' else 'openwebtext/val_split.txt'

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access= mmap.ACCESS_READ) as mm:

            file_size = len(mm)
            start_pos = random.randint(0,(file_size) - block_size*batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size* batch_size-1  )
            decoded_block = block.decode('utf-8', error= 'ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype = torch.long)
    return data 


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device) , y.to(device)
    return x,y
    
x,y = get_batch('train')

print('inputs:')
print(x)
print('targrets:')
print(y)
    



model = GPTLanguageModel(vocab_size)
# m = model.to(device)
# print('Loading Learnable Parameters')
# with open('model-01.pkl', 'rb') as f:
#     pickle.load(f)
# print('Model Loaded')
m= model.to(device)

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)


# In[71]:


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

    


# In[80]:


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model,f)
print('model saved')


# In[81]:


context = torch.zeros((1,1) , dtype= torch.long, device = device)
generated_chars = decode(m.generate(context, max_new_tokens = 5000)[0].tolist())


print(generated_chars)



# In[33]:


x = torch.tensor([10], dtype = torch.float32)
y = F.tanh(x)

print(y)


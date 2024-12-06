import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention



class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab:int, n_embed:int , n_tokens:int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):

        # (Batch_size, seq_len) -> (Batch_zize, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x 
    
    class CLIPLayer(nn.Module):

        def __init__(self, n_head:int, n_embed:int):
            super().__init__()

            self.layernorm_1 = nn.LayerNorm(n_embed)
            self.attention = SelfAttention(n_head, n_embed)
            self.layernorm_2 = nn.LayerNorm(n_head)
            self.linear_1 = nn.Linear(n_embed, 4*n_embed)
            self.linear_2 = nn.Linear(4*n_embed, n_embed)



        def forward(self, x: torch.Tensor) -> torch.Tensor:

            # (batch_size, seq_len, Dim)

            residue = x

            # SELF ATTENTION

            x= self.layernorm_1(x)

            x = self.attention(x, causal_mask = True)
            x += residue

            # FEEDFORWARD LAYER
            residue = x
            x = self.layernorm_2(x)

            x = self.linear_1(x)
        
            x = x * torch.sigmoid(1.702 * x)  #QuickGELU activation function

            x = self.linear_2(x)

            x += residue

            return x
        


# encode the prompt to be eble to feed it to the U-net model
# clip is basically very similar to the envoder layer of the transformer

class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49488, 768,77)

        self.Layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # (Batch_size, Seq_len) -> (batch_size, Seq_len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size, seq_len, dim)
        output = self.layernorm(state)


        return output
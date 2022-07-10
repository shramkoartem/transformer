import math
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor
import torch.functional as F



class Transformer(nn.Module):
    """
    Encoder-Decoder architecture
    """
    def __init__(
            self, encoder: nn.Module,
            decoder: nn.Module,
            generator: nn.Module,
            num_embeddings: int,
            hidden_dim: int = 512 
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.input_embedder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_dim
        )
        self.output_embedder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_dim
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        args:
            - inputs: (Tensor) book ids
        returns:
            - outputs: (Tensor) logits of next book
        """
        memory = self.encoder(inputs)
        hidden_state = self.decoder(inputs, memory)
        outputs = self.generator(hidden_state)
        return outputs
    
    def encode(self, inputs: Tensor) -> Tensor:
        """
        """
        embeddings = self.input_embedder(inputs)
        embeddings = self.encoder(embeddings)
        return embeddings

    def decode(self, inputs: Tensor) -> Tensor:
        """
        """
        embeddings = self.output_embedder(inputs)
        embeddings = self.decoder(embeddings)
        return embeddings

      
class Encoder(nn.Module):
    """
    Generates vector embeddings of supplied 
    books in readers context
    """
    def __init__(
            self,
            layer: nn.Module,
            n_layers: int,
        ):
        """
        Creates book id embedding layer.
        Stacks encoder layers.
        """
        super().__init__()
        
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        args:
            - inputs: (Tensor) book ids
        returns:
            - embeddings: (Tensor) book embeddings
        """
        for layer in self.layers:
            x = layer(x)
        return x
      
    class EncoderLayer(nn.Module):
    """
    Consists of two sub layers:
    - Multi-head self attention
    - Feed forward network
    
    After each sublayer the outputs
    are added to initial inputs and
    layer normalization is applied
    """
    def __init__(
            self,
            attention_layer: nn.Module,
            ffn_layer: nn.Module,
            layer_norm: nn.Module
        ):
        super().__init__()

        self.attention_layer = attention_layer
        self.ffn_layer = ffn_layer
        self.layer_norm = layer_norm

    def forward(self, x: Tensor) -> Tensor:
        """
        args:
            - inputs: Tensor
        returns:
            - outputs: Tensor
        """
        sub_x = self.attention_layer(x)
        x = self.layer_norm(x + sub_x)
        sub_x = self.ffn_layer(x)
        x = self.layer_norm(x + sub_x)
        return x


class Attention(ABC):

    @staticmethod
    def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        QKV attention mechanism
        y = softmax(q*k.T/ dim^0.5)*v
        """
        score = q @ k.transpose(-2, -1)
        scaled_score = score / math.sqrt(k.size(-1))
        attention = scaled_score.softmax(dim=-1)
        y = attention @ v
        return y, attention

class SelfAttention(nn.Module, Attention):
    """
    QKV self attention
    """
    def __init__(self, dim_x: int, dim_h: int):
        super().__init__()

        self.W_q = nn.Linear(dim_x, dim_h)
        self.W_k = nn.Linear(dim_x, dim_h)
        self.W_v = nn.Linear(dim_x, dim_h)

    def forward(self, x: Tensor) -> Tensor:
        """
        args:
            x: Tensor BxSxI
        returns:
            y: Tensor BxSxH
        """
        y, att = self.attention(
            q=self.W_q(x),
            k=self.W_k(x),
            v=self.W_v(x)
        )
        return y, att

class MixedAttention(nn.Module, Attention):
    """
    QKV self attention
    """
    def __init__(self, dim_x: int, dim_h: int):
        super().__init__()

        self.W_q = nn.Linear(dim_x, dim_h)
        self.W_k = nn.Linear(dim_x, dim_h)
        self.W_v = nn.Linear(dim_x, dim_h)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        args:
            x: Tensor BxSxI
        returns:
            y: Tensor BxSxH
        """
        y = self.attention(
            q=self.W_q(memory),
            k=self.W_k(memory),
            v=self.W_v(x)
        )
        return y


class MultiHeadSelfAttention(nn.Module):
    """
    QKV self attention with H heads
    """
    def __init__(self, attention_layer: nn.Module, dim_x:int, h: int):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(attention_layer) for _ in range(h)
        ])
        self.drop = nn.Dropout(p=0.1)
        self.dense = nn.Linear(dim_x, dim_x)

    def forward(self, x: Tensor) -> Tensor:
        """
        args:
            x: Tensor BxSxI
        returns:
            y: Tensor BxSxI
        """
        x = torch.cat(
            [layer(x)[0] for layer in self.layers],
            dim=-1
        )
        x = self.dense(self.drop(x))
        return x


class MultiHeadMixedAttention(nn.Module):
    """
    QKV mixed attention with H heads
    """
    def __init__(self, attention_layer: nn.Module, dim_x:int, h: int):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(attention_layer) for _ in range(h)
        ])
        self.drop = nn.Dropout(p=0.1)
        self.dense = nn.Linear(dim_x, dim_x)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        args:
            x: Tensor BxSxI
        returns:
            y: Tensor BxSxI
        """
        x = torch.cat(
            [layer(x, memory)[0] for layer in self.layers],
            dim=-1
        )
        x = self.dense(self.drop(x))
        return x

      
class FeedForwardNetwork(nn.Module):
    """
    """
    def __init__(self, dim_x: int, dim_h: int):
        super().__init__()

        self.dense = nn.Linear(dim_x, dim_h)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(dim_h, dim_x)

    def forward(self, x: Tensor) -> Tensor:
        """
        """
        x = self.dense(x).relu()
        x = self.drop(x)
        x = self.out(x)
        return x
      
class LayerNorm(nn.Module):
    """
    Layer normalization
    y = (x - E[x])/(sqrt(Var[x] + eta))*gamma + beta
    """
    def __init__(self, shape: Tuple[int], eps: float =1e-6):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        y = self.gamma*(x-mean)/(var+self.eps) + self.beta
        return y

class Decoder(nn.Module):
    """
    Stack of decoder layers
    """
    def __init__(self, layer, n_layers: int):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(layer) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
      
class DecoderLayer(nn.Module):
    """
    Mixed QKV multi head attention
    and feed forward network
    """
    def __init__(
            self,
            self_attention: nn.Module,
            mixed_attention: nn.Module,
            ffn_layer: nn.Module,
            layer_norm: nn.Module
        ):
        super().__init__()

        self.self_attention = self_attention
        self.mixed_attention = mixed_attention
        self.ffn_layer = ffn_layer
        self.layer_norm = layer_norm

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        args:
            - x: Tensor BxSxI inputs
            - memory: Tensor BxSxI encoder outputs
        returns:
            - y: Tensor BxSxI 
        """
        # 1. Self attention
        sub_x = self.self_attention(x)
        x = self.layer_norm(x + sub_x)

        # 2. Mixed attention
        sub_x = self.mixed_attention(x, memory)
        x = self.layer_norm(x + sub_x)

        # 3. Feed forward network
        sub_x = self.ffn_layer(x)
        x = self.layer_norm(x + sub_x)
        return x
      
      
encoder = Encoder(
    EncoderLayer(
        attention_layer=MultiHeadSelfAttention(
            SelfAttention(dim_x=512, dim_h=64),
            dim_x=512,
            h=8
        ),
        ffn_layer=FeedForwardNetwork(512, 512),
        layer_norm=LayerNorm(shape=(512, 512))
    ), n_layers=3
)
    

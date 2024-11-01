import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm, BatchNorm1d, Dropout, Linear
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class Intra_modal_atten(nn.Module):
    def __init__(self, d_model=768, nhead=8, dropout=0.1, layer_norm_eps=1e-5, num_seq=15):
        super(Intra_modal_atten, self).__init__()
                
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        # self.projection = nn.Conv2d(d_model, emb_dim, kernel_size=1) # 768 -> 128
        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positions = nn.Parameter(torch.randn(num_seq, d_model))

    def forward(self, x):
        # Apply the multihead attention layer to attend to all other epochs
        # The input sequence length is equal to the number of epochs (num_epochs)
        b, s, e = x.shape # [8, 10, 768]
        # x = x.permute(2, 0, 1) # ([768, 8, 10])
        # print("after permutation", x.shape)
        # x = self.projection(x) # [128, 8, 10]
        # print("after projection", x.shape)
        # x = x.permute(1, 2, 0) # [8, 10, 128]
        # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        # print("cls_tokens", cls_tokens.shape)
        # print("x", x.shape)
        # x = torch.cat([cls_tokens, x], dim=1)
        # print("self.positions", self.positions.shape)
        x += self.positions
        # print("after positional embedding", x.shape)
        x = x.permute(1, 0, 2) # need to check
        x_attended, average_weights = self.multihead_attn(x, x, x)  # query, key, value are all the same, use only cls tokens (?)
        # print("average_weights", average_weights.shape)
        out = x + self.dropout(x_attended)
        out = self.norm(out)
        
        return out, average_weights

class Feed_forward(nn.Module): 
    def __init__(self, d_model=64, dropout=0.1, dim_feedforward=512,
                 layer_norm_eps=1e-5,
                 device=None, dtype=None) -> None:

        super(Feed_forward, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:        
        src = x
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm(out)
        return out

class Seq_Cross_Transformer_Network(nn.Module):
    def __init__(self, model, d_model=128, dim_feedforward=512):
        # Epoch_cross_transformer is pretrained Vistion transformer
        super(Seq_Cross_Transformer_Network, self).__init__()
        
        # original trained model
        self.epoch = model
        # do the intra modal attention between these epochs
        self.seq_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, layer_norm_eps=1e-5, num_seq=15)
        # feed forward network
        self.ff_net = Feed_forward(d_model=d_model, dropout=0.1, dim_feedforward=dim_feedforward)               
        # Last layer for classification
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(d_model,5)) 
        
        
    def forward(self, images: Tensor, num_seq = 15):
        
        # print("images.shape ", images.shape)
        epochs = []
        for i in range(num_seq):
            epoch = self.epoch(images[:,i,:,:,:])
            epochs.append(epoch)
            
        seq = torch.stack(epochs, dim=1)
        # print("seq.shape ", seq.shape)
        seq, weights = self.seq_atten(seq)
        # print(seq.shape)
        seq = self.ff_net(seq)
        # print(seq.shape)

        outputs = []
        for i in range(num_seq):
            outputs.append(self.mlp(seq[i,:,:])) # changed here

        return weights, outputs
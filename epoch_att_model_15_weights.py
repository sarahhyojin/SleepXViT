import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm, BatchNorm1d, Dropout, Linear

class Intra_modal_atten(nn.Module):
    def __init__(self, d_model=1024, nhead=8, dropout=0.1, layer_norm_eps=1e-5):
        super(Intra_modal_atten, self).__init__()
                
        self.multihead_attn = nn.MultiheadAttention(1024, 8)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        
    def forward(self, x):
        # Apply the multihead attention layer to attend to all other epochs
        # x: [sequence_length, batch_size, d_model]
        # The input sequence length is equal to the number of epochs (num_epochs)
        x = x.permute(1, 0, 2) # !!!!!!
        # print("input shape", x.shape)
        x_attended, average_weights = self.multihead_attn(x, x, x)  # query, key, value are all the same
        out = x + self.dropout(x_attended)
        out = self.norm(out)
        
        return out, average_weights

class Feed_forward(nn.Module): 
    def __init__(self, d_model=64, dropout=0.1,dim_feedforward=512,
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
    def __init__(self, model, d_model = 128, dim_feedforward=512):
        # Epoch_cross_transformer is pretrained Vistion transformer
        super(Seq_Cross_Transformer_Network, self).__init__()
        
        self.epoch_1 = model
        self.epoch_2 = model
        self.epoch_3 = model
        self.epoch_4 = model
        self.epoch_5 = model
        self.epoch_6 = model
        self.epoch_7 = model
        self.epoch_8 = model
        self.epoch_9 = model
        self.epoch_10 = model
        self.epoch_11 = model
        self.epoch_12 = model
        self.epoch_13 = model
        self.epoch_14 = model
        self.epoch_15 = model
        
        # do the intra modal attention between these epochs
        self.seq_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, layer_norm_eps=1e-5)
        # feed forward network
        self.ff_net = Feed_forward(d_model=d_model, dropout=0.1, dim_feedforward=dim_feedforward)
                                                
        # Last layer for classification
        self.mlp_1    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))  ##################
        self.mlp_2    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_3    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_4    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_5    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))   

        self.mlp_6    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))  ##################
        self.mlp_7    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_8    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_9    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_10   = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))

        self.mlp_11   = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))  ##################
        self.mlp_12   = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_13   = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_14   = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_15   = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        
    def forward(self, images: Tensor, num_seg = 5):
        
        # print("images.shape ", images.shape)
        epoch_1 = self.epoch_1(images[:,0,:,:,:])
        epoch_2 = self.epoch_2(images[:,1,:,:,:])
        epoch_3 = self.epoch_3(images[:,2,:,:,:])
        epoch_4 = self.epoch_4(images[:,3,:,:,:])
        epoch_5 = self.epoch_5(images[:,4,:,:,:])
        
        # print("epoch_1.shape ", epoch_1.shape)
        epoch_6 = self.epoch_6(images[:,5,:,:,:])
        epoch_7 = self.epoch_7(images[:,6,:,:,:])
        epoch_8 = self.epoch_8(images[:,7,:,:,:])
        epoch_9 = self.epoch_9(images[:,8,:,:,:])
        epoch_10 = self.epoch_10(images[:,9,:,:,:])
        
        epoch_11 = self.epoch_11(images[:,10,:,:,:])
        epoch_12 = self.epoch_12(images[:,11,:,:,:])
        epoch_13 = self.epoch_13(images[:,12,:,:,:])
        epoch_14 = self.epoch_14(images[:,13,:,:,:])
        epoch_15 = self.epoch_15(images[:,14,:,:,:])
        
        seq = torch.stack((epoch_1, epoch_2, epoch_3, epoch_4, epoch_5,
                          epoch_6, epoch_7, epoch_8, epoch_9, epoch_10,
                          epoch_11, epoch_12, epoch_13, epoch_14, epoch_15), dim=1)

        # seq =  torch.cat([epoch_1, epoch_2,epoch_3,epoch_4,epoch_5], dim=0)
        #                  epoch_6, epoch_7,epoch_8,epoch_9,epoch_10,
        #                  epoch_11, epoch_12,epoch_13,epoch_14,epoch_15], dim=1)
        # print("seq.shape ", seq.shape)
        seq, weights = self.seq_atten(seq)
        # print("seq shape", seq.shape)
        # print("weight shape", weights.shape)
        seq = self.ff_net(seq)
        # print(seq.shape)
        out_1 = self.mlp_1(seq[0,:,:])
        out_2 = self.mlp_2(seq[1,:,:])
        out_3 = self.mlp_3(seq[2,:,:])
        out_4 = self.mlp_4(seq[3,:,:])
        out_5 = self.mlp_5(seq[4,:,:])
        #
        out_6 = self.mlp_6(seq[5,:,:])
        out_7 = self.mlp_7(seq[6,:,:])
        out_8 = self.mlp_8(seq[7,:,:])
        out_9 = self.mlp_9(seq[8,:,:])
        out_10 = self.mlp_10(seq[9,:,:])
        # #
        out_11 = self.mlp_11(seq[10,:,:])
        out_12 = self.mlp_12(seq[11,:,:])
        out_13 = self.mlp_13(seq[12,:,:])
        out_14 = self.mlp_14(seq[13,:,:])
        out_15 = self.mlp_15(seq[14,:,:])

        # print(out_1.shape)
        # print(out_1[0], out_2[0], out_3[0], out_4[0], out_5[0])
        return weights, [out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8,out_9,out_10,out_11,out_12,out_13,out_14,out_15]
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Text(nn.Module):
    
    def __init__(self, args):
        super(LSTM_Text,self).__init__()
        self.args = args
        self.bs = args.batch_size
        
        V = args.embed_num
        D = args.d_embed

        self.embed = nn.Embedding(V, D)

        if args.use_pretrain: # assign pretrained vectors to the embedding matrix
            self.embed.weight.data[args.vectors != 0] = args.vectors[args.vectors != 0] 

        Hd = args.Hd
        self.lstm = nn.LSTM(D, Hd, 1, batch_first=True, bidirectional=False)

        self.dropout = nn.Dropout(args.dropout)

        H = args.Hd//2

        self.mlp = nn.Linear(Hd, H)  #regularization trick
        #self.mlp = nn.Linear(Hd, args.class_num)

        # Attention layer
        A = 20
        self.att = nn.Linear(Hd, A)
        self.att2 = nn.Linear(A, 1, bias=False)
        self.use_att = args.use_att

    def forward(self, x):
        
        x = self.embed(x) # (N,W,D)
        
        hs,_ = self.lstm(x) # (N,W,Hd)

        if self.use_att:
            has = F.tanh(self.att(hs)) # (N,W,Had)
            at = F.softmax(self.att2(has),dim=1)
            h = (at*hs).sum(dim=1)
        else:
            h = hs.mean(1).squeeze()

        h = self.dropout(h) 
        logit = self.mlp(h) 
        return logit

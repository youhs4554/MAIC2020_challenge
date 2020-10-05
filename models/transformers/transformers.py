import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel_MTL(nn.Module):
    def __init__(self, n_cls, *args, **kwargs):
        super().__init__()
        if n_cls == 2:
            n_cls = 1  # sigmoid -> bce on single node

        transformer = nn.Transformer(*args, **kwargs)

        d_model = transformer.d_model
        self.embedding = nn.Linear(100, d_model, bias=False)
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.dense = nn.Linear(d_model, 100)
        self.logits = nn.Linear(d_model, n_cls)
        self.d_model = d_model
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _EncodeInputsAndGetLogits(self, src):
        # zero-pad at the end of src
        src = F.pad(src, (0, 100), value=0.0)

        # each vector represents samples for 1 sec
        src = src.view(-1, 21, 100).transpose(0, 1)  # (S+1,N,E)
        src = self.embedding(src)

        encoder_out = self.encoder(src)

        # last encoder output -> classification token
        memory, cls_token = torch.split(
            encoder_out, (len(encoder_out)-1, 1), dim=0)

        cls_token = cls_token.view(-1, self.d_model)

        # classification task
        logits_out = self.logits(cls_token)
        logits_out = torch.sigmoid(logits_out)

        return memory, logits_out

    def forward(self, src, tgt=None):
        memory, logits_out = self._EncodeInputsAndGetLogits(src)

        if tgt is None:
            return logits_out

        # decoder inputs
        tgt = tgt.view(-1, 60,
                       100).transpose(0, 1)  # (T,N,E)
        tgt = self.embedding(tgt)

        # reconstruction task
        decoder_out = self.decoder(tgt, memory)
        decoder_out = self.dense(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)  # major-axis -> batch

        return logits_out, decoder_out


class TransformerModel(nn.Module):

    def __init__(self, nout, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp*20, nout)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp) # 이미 임베딩 되어 있다고 가정
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.view(-1, 2000)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def transformers(nout=1, ninp=100, nhead=10, nhid=256, nlayers=2, dropout=0.2):
    return TransformerModel(nout, ninp, nhead, nhid, nlayers, dropout)


if __name__ == "__main__":
    # from models.transformers import transformer
    from torch import nn
    import torch
    # X_data : [N_samples, 4(external)+2000(signal)] => [N_sample, 2004]
    # y_data : [N_samples, 6000(future)+1(label)] => [N_sample, 6001]

    # transformer_model = nn.Transformer(
    #     d_model=100, nhead=10, num_encoder_layers=12)
    # transformer_model.eval()
    # src = torch.rand((20, 1, 100))
    # tgt = torch.rand((60, 1, 100))
    # out = transformer_model(src, tgt)
    # print(out)
    X = torch.rand((20, 1, 100))
    print(X.size())

    model = transformers()
    out = model(X)
    print(out.size())

    linearmodel = nn.Linear(100, 1)
    out2 = linearmodel(X)
    print(out2.size())

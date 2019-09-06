# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncRNN(nn.Module):
    def __init__(self, vsz, embed_dim, hidden_dim, n_layers, use_birnn, dout):
        super(EncRNN, self).__init__()
        self.embed = nn.Embedding(vsz, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           bidirectional=use_birnn)
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs):
        embs = self.dropout(self.embed(inputs))
        enc_outs, hidden = self.rnn(embs)
        return self.dropout(enc_outs), hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out*enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out*energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class DecRNN(nn.Module):
    def __init__(self, vsz, embed_dim, hidden_dim, n_layers, use_birnn, 
                 dout, attn, tied):
        super(DecRNN, self).__init__()
        hidden_dim = hidden_dim*2 if use_birnn else hidden_dim

        self.embed = nn.Embedding(vsz, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim , n_layers)

        self.w = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn = Attention(hidden_dim, attn)

        self.out_projection = nn.Linear(hidden_dim, vsz)
        if tied: 
            if embed_dim != hidden_dim:
                raise ValueError(
                    f"when using the tied flag, embed-dim:{embed_dim} \
                    must be equal to hidden-dim:{hidden_dim}")
            self.out_projection.weight = self.embed.weight
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs):
        inputs = inputs.unsqueeze(0)
        embs = self.dropout(self.embed(inputs))
        dec_out, hidden = self.rnn(embs, hidden)

        attn_weights = self.attn(dec_out, enc_outs).transpose(1, 0)
        enc_outs = enc_outs.transpose(1, 0)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs)
        cats = self.w(torch.cat((dec_out, context.transpose(1, 0)), dim=2))
        pred = self.out_projection(cats.tanh().squeeze(0))
        return pred, hidden


class Seq2seqAttn(nn.Module):
    def __init__(self, args, fields, device):
        super().__init__()
        self.src_field, self.tgt_field = fields
        self.src_vsz = len(self.src_field[1].vocab.itos)
        self.tgt_vsz = len(self.tgt_field[1].vocab.itos)
        self.encoder = EncRNN(self.src_vsz, args.embed_dim, args.hidden_dim, 
                              args.n_layers, args.bidirectional, args.dropout)
        self.decoder = DecRNN(self.tgt_vsz, args.embed_dim, args.hidden_dim, 
                              args.n_layers, args.bidirectional, args.dropout,
                              args.attn, args.tied)
        self.device = device
        self.n_layers = args.n_layers
        self.hidden_dim = args.hidden_dim
        self.use_birnn = args.bidirectional

    def forward(self, srcs, tgts=None, maxlen=100, tf_ratio=0.0):
        slen, bsz = srcs.size()
        tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        tf_ratio = tf_ratio if isinstance(tgts, torch.Tensor) else 0.0
       
        enc_outs, hidden = self.encoder(srcs)

        dec_inputs = torch.ones_like(srcs[0]) * 2 # <eos> is mapped to id=2
        outs = []

        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, bsz, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)

        for i in range(tlen):
            preds, hidden = self.decoder(dec_inputs, hidden, enc_outs)
            outs.append(preds)
            use_tf = random.random() < tf_ratio
            dec_inputs = tgts[i] if use_tf else preds.max(1)[1]
        return torch.stack(outs)

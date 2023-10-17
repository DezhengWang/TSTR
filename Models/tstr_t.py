import torch
import torch.nn as nn
from LayersStatic.encoder import Encoder, EncoderLayer
from LayersStatic.decoder import Decoder, DecoderLayer
from LayersStatic.attention import AttentionLayer, StaticAttention
from utils.embed import DataEmbeddingStatic, TempEmbedding


class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, embed='fixed', freq='h', activation='gelu',
                 output_attention=False, mix=True, args=None):
        super(Model, self).__init__()
        self.args = args
        self.pred_len = out_len
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbeddingStatic(enc_in+1, d_model, dropout)
        self.dec_embedding = DataEmbeddingStatic(dec_in, d_model, dropout)
        self.enc_embedding_temp = TempEmbedding()
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(StaticAttention(False, dropout=dropout, atts=output_attention,
                                                   n_size=(self.args.seq_len,self.args.seq_len),
                                                   n_heads=n_heads),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(StaticAttention(True, dropout=dropout, atts=False,
                                                   n_size=(self.args.label_len+self.args.pred_horizon,
                                                           self.args.label_len+self.args.pred_horizon),
                                                   n_heads=n_heads),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(StaticAttention(False, dropout=dropout, atts=False,
                                                   n_size=(self.args.label_len+self.args.pred_horizon,
                                                           self.args.seq_len),
                                                   n_heads=n_heads),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        temp_enc = self.enc_embedding_temp(x_mark_enc)
        x_enc = torch.cat([x_enc, temp_enc], dim=2)

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
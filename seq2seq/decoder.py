import torch.nn as nn
import config
import torch
import torch.nn.functional as F
from seq2seq.attention import Attention


class Decoder(nn.Module):
    def __init__(self, decoder_num_embedding, decoder_embedding_dim, bidirectional=False, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.decoder_num_embedding = decoder_num_embedding
        self.decoder_embedding_dim = decoder_embedding_dim
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        self.embeddings = nn.Embedding(num_embeddings=self.decoder_num_embedding,
                                       embedding_dim=self.decoder_embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        pass

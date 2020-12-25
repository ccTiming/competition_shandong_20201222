import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, encoder_num_embeddings, encoder_embedding_dim,
                 hidden_size, num_layers, bidirectional=False, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.encoder_num_embeddings = encoder_num_embeddings
        self.encoder_embedding_dim = encoder_embedding_dim
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embeddings = nn.Embedding(
            num_embeddings=self.encoder_num_embeddings,
            embedding_dim=self.encoder_embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.gru = nn.GRU(
            input_size=self.encoder_embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, x):
        """
        :param x: [batch_size,seq_len]
        :return:  out:[batch_size,seq_len,hidden_size]
                  hidden:[num_layers,batch_size,hidden_size]
        """
        # embeddings:[batch_size,seq_len,embedding_dim]
        embeddings = self.embeddings(x)
        dropout_embeddings = self.dropout(embeddings)

        out, hidden = self.gru(dropout_embeddings)
        return out, hidden

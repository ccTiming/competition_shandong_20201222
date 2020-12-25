import torch.nn as nn
from seq2seq.encoder import Encoder
from seq2seq.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_num_embeddings, encoder_embedding_dim, encoder_hidden_size, encoder_num_layers,
                 decoder_num_embedding, decoder_embedding_dim, decoder_num_layers, decoder_hidden_size
                 ):
        super(Seq2Seq, self).__init__()

        self.encoder_num_embeddings = encoder_num_embeddings
        self.encoder_embedding_dim = encoder_embedding_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers

        self.decoder_num_embedding = decoder_num_embedding
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = Encoder(self.encoder_num_embeddings, self.encoder_embedding_dim,
                               self.encoder_hidden_size, self.encoder_num_layers)
        self.decoder = Decoder(self.decoder_num_embedding, self.decoder_embedding_dim, self.decoder_num_layers,
                               self.decoder_hidden_size, self.encoder_hidden_size)

    def forward(self, inputs, target):
        encoder_outputs, encoder_hidden_state = self.encoder(inputs)
        decoder_outputs, decoder_hidden_state = self.decoder(target, encoder_outputs, encoder_hidden_state)
        return decoder_outputs, decoder_hidden_state

from seq2seq.decoder import Decoder
from seq2seq.encoder import Encoder
import torch


def test_encoder():
    inputs = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    encoder = Encoder(encoder_num_embeddings=1000, encoder_embedding_dim=128, dropout_rate=0.1,
                      hidden_size=128, num_layers=2, bidirectional=False)
    output, hidden = encoder(inputs)
    print(encoder)
    print(output.size(), hidden.size())


def test_decoder():
    decoder = Decoder(100, 128, 128, 2, 100, 128)
    print(decoder)


if __name__ == '__main__':
    test_encoder()

import torch.nn as nn
import config
import torch
import torch.nn.functional as F
from seq2seq.attention import Attention


class Decoder(nn.Module):
    def __init__(self, decoder_num_embeddings, decoder_embedding_dim, decoder_hidden_size, decoder_num_layers,
                 encoder_num_embeddings, encoder_hidden_size):
        super(Decoder, self).__init__()

        self.decoder_num_embeddings = decoder_num_embeddings
        self.encoder_num_embeddings = encoder_num_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=decoder_num_embeddings,
            embedding_dim=decoder_embedding_dim,
        )

        self.gru = nn.GRU(
            input_size=decoder_embedding_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            batch_first=True
        )

        self.dense1 = nn.Linear(decoder_hidden_size, decoder_num_embeddings, bias=False)

        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        self.dense2 = nn.Linear(decoder_hidden_size + encoder_hidden_size,
                                decoder_hidden_size, bias=False)

    def forward(self, encoder_hidden, encoder_outputs, target):
        decoder_hidden = encoder_hidden
        batch_size = target.size(0)
        # 设置开始[SOS]为 2
        decoder_input = torch.LongTensor(
            torch.ones([batch_size, 1], dtype=torch.int64) * 2).to(config.device)
        # max_len 设置为 128
        decoder_outputs = torch.zeros(
            [batch_size, 128 + 1, self.decoder_num_embeddings]).to(config.device)
        for t in range(128 + 1):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[:, t, :] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        # print("decoder_input_size:", decoder_input.size())
        decoder_input_embedded = self.embedding(decoder_input)
        # print("decoder_input__embedded_size:", decoder_input_embedded.size())
        out, decoder_hidden = self.gru(decoder_input_embedded, decoder_hidden, )

        # 添加 attention [batch_size,input_seq_len]*[batch_size,input_seq_len,input_hidden_size]
        attention_weight = self.attention(decoder_hidden, encoder_outputs).unsqueeze(1)
        context_vector = attention_weight.bmm(encoder_outputs)  # [batch_size,1,input_hidden_size]

        concat = torch.cat([out, context_vector], dim=-1)  # [batch_size,1,decoder_hidden_size+encoder_hidden_size]
        concat = concat.squeeze(1)
        out = torch.tanh(self.wa(concat))
        # attention 结束
        # out = out.squeeze(1)
        output = F.log_softmax(self.fc(out), dim=-1)
        return output, decoder_hidden

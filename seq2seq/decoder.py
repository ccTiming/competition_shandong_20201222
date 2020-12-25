import torch.nn as nn
import config
import torch
import torch.nn.functional as F
from seq2seq.attention import Attention


class Decoder(nn.Module):
    def __init__(self, decoder_num_embedding, decoder_embedding_dim,
                 decoder_num_layers, decoder_hidden_size,
                 encoder_hidden_size,
                 bidirectional=False, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.decoder_num_embedding = decoder_num_embedding
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        self.embeddings = nn.Embedding(num_embeddings=self.decoder_num_embedding,
                                       embedding_dim=self.decoder_embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.gru = nn.GRU(input_size=self.decoder_embedding_dim,
                          num_layers=self.decoder_num_layers,
                          hidden_size=self.decoder_hidden_size,
                          bidirectional=self.bidirectional,
                          batch_first=True)
        self.attention = Attention(encoder_hidden_size=self.encoder_hidden_size)
        self.dense1 = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size,
                                self.decoder_hidden_size,
                                bias=False)
        self.dense2 = nn.Linear(self.decoder_hidden_size, self.decoder_num_embedding)

    def forward(self, target, encoder_hidden_state, encoder_outputs):
        """
        :param target: list, sequence of sentence
        :param encoder_hidden_state: [num_layers*bidirectional,batch_size,encoder_hidden_size]
        :param encoder_outputs: [batch_size,seq_len,encoder_hidden_size]
        :return:
        """
        decoder_hidden_state = encoder_hidden_state
        batch_size, seq_len = encoder_outputs.size()[:2]

        # 设置句子开始的第一个字符的 id 为 0
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * 0).to(config.device)
        decoder_outputs = torch.zeros([batch_size, 15, self.decoder_num_embedding]).to(config.device)
        # 设置句子输出 15 个字符,包括开始字符长度为 16
        for t in range(15):
            # decoder_output_t: [batch_size,1,decoder_num_embeddings]
            decoder_output_t, decoder_hidden_state = self.forward_step(decoder_input, decoder_hidden_state,
                                                                       encoder_outputs)
            decoder_outputs[:, t, :] = decoder_output_t
            _, index = torch.topk(decoder_output_t, 1)
            decoder_input = index

        return decoder_outputs, decoder_hidden_state

    def forward_step(self, decoder_input, decoder_hidden_state, encoder_outputs):
        """
        :param decoder_input: [batch_size,1]
        :param decoder_hidden_state: [num_layers,batch_size,decoder_hidden_size]
        :param encoder_outputs: [batch_size, seq_len,encoder_hidden_size]
        :return: out:[batch_size,1,decoder_num_embeddings]

        """
        # embeddings:[batch_size,1,embedding_dim]
        embeddings = self.embeddings(decoder_input)
        embeddings_dropout = self.dropout(embeddings)

        # out:[batch_size,1,decoder_hidden_state]
        # hidden:[num_layers*bidirectional,batch_size,decoder_hidden_state]
        out, hidden = self.gru(embeddings_dropout, decoder_hidden_state)

        # 添加 attention #
        # attention_weight:[batch_size,1,seq_len]
        attention_weight = self.attention(hidden, encoder_outputs).unsqueeze(1)
        # context_vector:[batch_size,1,encoder_hidden_size]
        context_vector = attention_weight.bmm(encoder_outputs)
        # concat:[batch_size,1,decoder_hidden_state+encoder_hidden_state]
        concat = torch.cat([out, context_vector], dim=-1)
        concat = concat.squeeze(1)
        out = torch.tanh(self.dense1(concat))
        # attention 结束 #

        out = F.log_softmax(self.dense2(out), dim=-1)
        return out, hidden

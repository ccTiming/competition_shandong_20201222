import torch.nn as nn
import torch
import torch.nn.functional as F
import config


class BiLSTMAttention(nn.Module):
    def __init__(self, num_embeddings=508, embedding_dim=30, hidden_size=16, num_layers=1):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        self.dense = nn.Linear(hidden_size * 2, 508)

        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def forward(self, inputs):
        """
        :param inputs: [batch_size,seq_len]
        :return:
        """
        embedding = self.embedding(inputs)  # [batch_size,seq_len,embedding_dim]
        output, hidden = self.lstm(embedding)  # [batch_size,seq_len,hidden_size*2]

        # 添加 attention
        u = torch.tanh(torch.matmul(output, self.w))  # [batch_size,seq_len,hidden_size*2]
        attention = torch.matmul(u, self.u)  # [batch_size,seq_len,1]
        score = F.softmax(attention, dim=-1)  # [batch_size,seq_len,1]
        output = output * score  # [batch_size,seq_len,hidden_size*2]
        # attention 结束
        # print(output.size())
        output = torch.sum(output, dim=1)  # [batch_size,hidden_size*2]
        # print(output.size(), hidden.size())
        output = self.dense(output)  # [batch_size,508]

        return output, hidden

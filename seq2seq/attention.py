import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, method='general'):
        super(Attention, self).__init__()
        assert method in ['dot', 'general', 'concat'], 'this method has not been implemented'
        self.method = method
        if self.method == 'general':
            self.dense_general = nn.Linear(encoder_hidden_size, encoder_hidden_size, bias=False)
        if self.method == 'concat':
            self.dense1_cat = nn.Linear(2 * encoder_hidden_size, encoder_hidden_size, bias=False)
            self.dense2_cat = nn.Linear(encoder_hidden_size, 1)

    def forward(self, hidden_state, encoder_output):
        """
        :param hidden_state: [num_layers*directional,batch_size,encoder_hidden_size]
        :param encoder_output: [batch_size,seq_len,encoder_hidden_size]
        :return:
        """
        if self.method == 'dot':
            hidden_state = hidden_state[-1, :, :]  # [1,batch_size,encoder_hidden_size]
            hidden_state = hidden_state.permute(1, 2, 0)  # [batch_size,encoder_hidden_size,1]
            # attention: [batch_size,seq_len,1]-->[batch_size,seq_len]
            attention = encoder_output.bmm(hidden_state).squeeze(-1)
            attention_weight = F.softmax(attention)
        elif self.method == 'general':
            encoder_output = self.dense_general(encoder_output)  # [batch_size,seq_len,encoder_hidden_size]
            # hidden_state:[1,batch_size,encoder_hidden_size]-->[batch_size,encoder_hidden_size,1]
            hidden_state = hidden_state[-1, :, :].permute(1, 2, 0)
            attention = encoder_output.bmm(hidden_state).squeeze(-1)
            attention_weight = F.softmax(attention)  # [batch_size,seq_len]
        elif self.method == 'concat':
            batch_size, seq_len = encoder_output.size()[:2]
            hidden_state = hidden_state[-1, :, :].squeeze(0)  # [batch_size,encoder_hidden_size]
            hidden_state.repeat(1, seq_len, 1)  # [batch_size,seq_len,encoder_hidden_size]
            concat = torch.cat([encoder_output, hidden_state], dim=-1)  # [batch_size,seq_len,2*encoder_hidden_size]
            attention = self.dense2_cat(F.tanh(self.dense1_cat(concat.view((batch_size * seq_len, -1))))).squeeze(-1)
            attention_weight = F.softmax(attention.view((batch_size, seq_len)))
        else:
            attention_weight = None
        return attention_weight

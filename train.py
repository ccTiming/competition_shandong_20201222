from seq2seq import Seq2Seq
import config
from torch.optim import Adam
from dataset import disease_data_loader
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

s2s = Seq2Seq(
    encoder_num_embeddings=config.encoder_num_embeddings,
    encoder_embedding_dim=config.encoder_embedding_dim,
    encoder_hidden_size=config.encoder_hidden_size,
    encoder_num_layers=config.encoder_num_layers,
    decoder_num_embeddings=config.decoder_num_embeddings,
    decoder_embedding_dim=config.decoder_embedding_dim,
    decoder_hidden_size=config.decoder_hidden_size,
    decoder_num_layers=config.decoder_num_layers
).to(config.device)

optimizer = Adam(s2s.parameters(), lr=0.001)


def train(epoch):
    bar = tqdm(enumerate(disease_data_loader), total=len(disease_data_loader), desc='train')
    total_loss = []
    for idx, data in bar:
        x = data.get('input')
        y = data.get('output')
        single_x = x[:, :, 0].to(config.device)
        single_y = y[:, :, 0].to(config.device)
        optimizer.zero_grad()
        output, _ = s2s(single_x, single_y)
        value, indices = torch.topk(output, k=1, dim=-1)
        indices = indices.squeeze(-1)
        loss = F.mse_loss(indices.float(), single_y.float())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    # print(epoch, sum(total_loss) / len(bar))
    x = torch.LongTensor(
        [[78, 51, 77, 113, 79, 72, 91, 76, 81, 87, 51, 88, 97, 55, 84, 50, 80, 62, 82, 83, 81, 57, 82, 72, 65, 91, 85,
          82, 66, 47, 88]])
    output, _ = s2s(x, None)
    value, indices = torch.topk(output, k=1, dim=-1)
    indices = indices.flatten().tolist()
    # print(len(indices))
    print('loss', mean_squared_error(indices, [68, 102, 112, 107, 87, 78, 99, 49, 101, 75, 92, 62, 60, 99]))


if __name__ == '__main__':
    for i in range(8):
        train(i)
        # break
"""
78,
51,
77,
113,
79,
72,
91,
76,
81,
87,
51,
88,
97,
55,
84,
50,
80,
62,
82,
83,
81,
57,
82,
72,
65,
91,
85,
82,
66,
47,
88,
"""

from lstm.bi_lstm_attention import BiLSTMAttention
import config
from torch.optim import Adam
from dataset import disease_data_loader
from tqdm import tqdm
import torch.nn as nn
import torch

bi_lstm = BiLSTMAttention().to(config.device)
optimizer = Adam(bi_lstm.parameters(), lr=0.001)


def train(epoch):
    bar = tqdm(enumerate(disease_data_loader), total=len(disease_data_loader), desc='train')
    for idx, data in bar:
        x = data.get('input')
        y = data.get('output')
        single_x = x[:, :, 0].to(config.device)
        single_y = y[:, :, 0].to(config.device)

        optimizer.zero_grad()
        output = bi_lstm(single_x)
        output = nn.Softmax(dim=-1)(output)
        print(output)
        print(output.size())
        loss = nn.MSELoss()(output, single_y)
        loss.backward()
        optimizer.step()

        print(loss)


if __name__ == '__main__':
    for i in range(8):
        train(i)
        break

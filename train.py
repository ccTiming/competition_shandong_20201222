from seq2seq import Seq2Seq
import config
from torch.optim import Adam, SGD
from dataset import disease_train_data_loader, disease_test_data_loader
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

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

optimizer = Adam(s2s.parameters(), lr=0.000001)


def train(epoch):
    bar = tqdm(enumerate(disease_train_data_loader), total=len(disease_train_data_loader), desc='train')
    total_loss = []
    for idx, data in bar:
        x = data.get('input')[:, :1, :].squeeze(0)
        y = data.get('output')[:, :1, :].squeeze(0)

        optimizer.zero_grad()
        output, _ = s2s(x, y)
        value, indices = torch.topk(output, k=1, dim=-1)
        indices = indices.squeeze(-1)
        loss = nn.MSELoss()(indices.float(), y.float())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    print(epoch, sum(total_loss) / len(bar))


if __name__ == '__main__':
    for i in range(100):

        train(i)
    # bar = tqdm(enumerate(disease_test_data_loader), total=len(disease_test_data_loader), desc='predict')
    # for idx, data in bar:
    #     x = data.get('input').squeeze(0)
    #     output, _ = s2s(x, None)
    #     value1, indices1 = torch.topk(output, k=1, dim=-1)
    #     indices1 = indices1.squeeze(-1)
    #     print(indices1)

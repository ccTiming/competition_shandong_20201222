import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data2017_path = '健康医疗-数据/source_data/count_2017.csv'
train_data2018_path = '健康医疗-数据/source_data/count_2018.csv'
test_data2019_path = '健康医疗-数据/source_data/count_2019.csv'
test_data2020_path = '健康医疗-数据/source_data/count_2020.csv'

save_5_days_mean_path = '健康医疗-数据/5日平均/'
save_2_years_data = '健康医疗-数据/data/count_2017_2018.txt'
save_2_years_test_data = '健康医疗-数据/data/count_2019_2020.txt'

vocab_path = '健康医疗-数据/vocab'

# model 相关  522个词
decoder_num_embeddings = 728
encoder_num_embeddings = 728
decoder_embedding_dim = 128
encoder_embedding_dim = 128
decoder_hidden_size = 64
encoder_hidden_size = 64
decoder_num_layers = 1
encoder_num_layers = 1

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data2017_path = '健康医疗-数据/source_data/count_2017.csv'
train_data2018_path = '健康医疗-数据/source_data/count_2018.csv'

save_5_days_mean_path = '健康医疗-数据/5日平均/'
save_2_years_data = '健康医疗-数据/data/count_2017_2018.txt'

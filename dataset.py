from torch.utils.data import Dataset, DataLoader
import config
import pandas as pd
import torch
import numpy as np


class HealthTrainDataset(Dataset):
    def __init__(self, data_path=config.save_2_years_data):
        super(HealthTrainDataset, self).__init__()
        df = pd.read_csv(data_path, delimiter=',')
        iter_row = df[df['时间'] > 20171031].iterrows()
        self.inputs = []
        self.outputs = []
        tmp = []
        for idx, d in iter_row:
            # 去掉时间
            tmp.append(d[1:].tolist())
            if len(tmp) == 44:
                self.inputs.append(tmp[:30])
                self.outputs.append(tmp[30:])
                tmp = []

    def __getitem__(self, index):
        # diseases = ['上呼吸道感染', '上呼吸道疾病', '丘疹性荨麻疹', '中耳炎', '发热', '变应性鼻炎', '呕吐', '呼吸道感染', '咳嗽', '哮喘', '外伤证', '头晕', '头痛',
        #             '急性上呼吸道感染', '急性喉炎', '急性扁桃体炎', '手足口病', '支气管炎', '消化不良', '湿疹', '猩红热样红斑', '疱疹性咽峡炎', '皮炎', '皮疹', '结膜炎',
        #             '肺炎', '胃肠功能紊乱', '腹泻', '腹痛', '荨麻疹']
        x = torch.LongTensor(self.inputs[index]).permute(1, 0)
        y = torch.LongTensor(self.outputs[index]).permute(1, 0)
        return {'input': x, 'output': y}

    def __len__(self):
        return len(self.inputs)


class HealthPredictDataset(Dataset):
    def __init__(self, data_path=config.save_2_years_test_data):
        super(HealthPredictDataset, self).__init__()
        df = pd.read_csv(data_path, delimiter=',')
        self.inputs = []
        durations = [
            (20190130, 20190228),
            (20190502, 20190531),
            (20190802, 20190831),
            (20191101, 20191130),
            (20200302, 20200331),
            (20200601, 20200630),
        ]
        tmp = []
        for duration in durations:
            iter_row = df[df['时间'].between(duration[0], duration[1])].iterrows()
            for _, d in iter_row:
                tmp.append(d[1:].tolist())
            self.inputs.append(tmp)
            tmp = []

    def __getitem__(self, index):
        # diseases = ['上呼吸道感染', '上呼吸道疾病', '丘疹性荨麻疹', '中耳炎', '发热', '变应性鼻炎', '呕吐', '呼吸道感染', '咳嗽', '哮喘', '外伤证', '头晕', '头痛',
        #             '急性上呼吸道感染', '急性喉炎', '急性扁桃体炎', '手足口病', '支气管炎', '消化不良', '湿疹', '猩红热样红斑', '疱疹性咽峡炎', '皮炎', '皮疹', '结膜炎',
        #             '肺炎', '胃肠功能紊乱', '腹泻', '腹痛', '荨麻疹']
        x = torch.LongTensor(self.inputs[index]).permute(1, 0)
        return {'input': x}

    def __len__(self):
        return len(self.inputs)


disease_train_data_loader = DataLoader(HealthTrainDataset(), shuffle=True, batch_size=1)
disease_test_data_loader = DataLoader(HealthPredictDataset(), shuffle=False, batch_size=1)

if __name__ == '__main__':
    # ht = HealthTrainDataset()
    # item = ht.__getitem__(0)
    # print(item['input'].size(), item['output'].size())

    hp = HealthPredictDataset()
    item = hp.__getitem__(-1)
    print(item['input'])

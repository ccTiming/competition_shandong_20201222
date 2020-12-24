from torch.utils.data import Dataset, DataLoader
import config
import pandas as pd
import torch


class HealthDataset(Dataset):
    def __init__(self, data_path=config.save_2_years_data):
        super(HealthDataset, self).__init__()
        df = pd.read_csv(data_path, delimiter=',')
        iter_row = df[df['时间'] > 20171031].iterrows()
        self.inputs = []
        self.outputs = []
        tmp = []
        for idx, d in iter_row:
            # 去掉时间
            tmp.append(d[:-1].tolist())
            if len(tmp) == 45:
                self.inputs.append(tmp[:30])
                self.outputs.append(tmp[30:])
                tmp = []

    def __getitem__(self, item):
        # diseases = ['上呼吸道感染', '上呼吸道疾病', '丘疹性荨麻疹', '中耳炎', '发热', '变应性鼻炎', '呕吐', '呼吸道感染', '咳嗽', '哮喘', '外伤证', '头晕', '头痛',
        #             '急性上呼吸道感染', '急性喉炎', '急性扁桃体炎', '手足口病', '支气管炎', '消化不良', '湿疹', '猩红热样红斑', '疱疹性咽峡炎', '皮炎', '皮疹', '结膜炎',
        #             '肺炎', '胃肠功能紊乱', '腹泻', '腹痛', '荨麻疹']
        x = torch.LongTensor(self.transfer(self.inputs[item]))
        y = torch.LongTensor(self.transfer(self.outputs[item]))
        return {'input': x, 'output': y}

    @staticmethod
    def transfer(x):
        for i in range(len(x)):
            for j in range(len(x[0])):
                try:
                    x[i][j] = int(x[i][j])
                except:
                    # 值为'None'
                    x[i][j] = 9999
        return x

    def __len__(self):
        return len(self.inputs)


disease_data_loader = DataLoader(HealthDataset(), shuffle=True, batch_size=1)

if __name__ == '__main__':
    hd = HealthDataset()
    # print(hd.__getitem__(0))
    for i, data in enumerate(disease_data_loader):
        print(i, data)
        break

import pandas as pd
import config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import time
import datetime

plt.rcParams['font.family'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def bar_chart(data_path, save_path):
    df = pd.read_csv(data_path, delimiter=',')
    heads = df.columns.tolist()
    # print(df)
    # print(heads)
    # print(df)
    group_by_name = df.groupby(df['admin_illness_name'])

    # print(group_by_name.sum())
    count = [value[-1] for value in group_by_name.sum().values]
    name = group_by_name.sum().index.tolist()
    print(count, name)
    plt.bar(x=name, height=count)
    plt.xticks(rotation=270)
    plt.savefig(save_path)
    plt.show()


def line_chart(data_path, save_path):
    diseases = ['上呼吸道感染', '上呼吸道疾病', '丘疹性荨麻疹', '中耳炎', '发热', '变应性鼻炎', '呕吐', '呼吸道感染', '咳嗽', '哮喘', '外伤证', '头晕', '头痛',
                '急性上呼吸道感染', '急性喉炎', '急性扁桃体炎', '手足口病', '支气管炎', '消化不良', '湿疹', '猩红热样红斑', '疱疹性咽峡炎', '皮炎', '皮疹', '结膜炎', '肺炎',
                '胃肠功能紊乱', '腹泻', '腹痛', '荨麻疹']
    df = pd.read_csv(data_path, delimiter=',')
    total_data = {}
    dates = []
    counts = []
    labels = []
    _counts = []
    _dates = []
    for disease in diseases:
        for idx, (name, date, count) in df.iterrows():
            if name != disease:
                continue
            month = str(date)[4:6]
            day = str(date)[6:]

            if month not in labels:
                labels.append(month)
                counts.append(_counts)
                dates.append(_dates)
                _counts = []
                _dates = []
            _counts.append(count)
            _dates.append(day)

        counts.append(_counts)
        dates.append(_dates)
        counts = counts[1:]
        dates = dates[1:]

        total_data[disease] = list(zip(labels, dates, counts))

    with open("2017_total.json", "w")as f:
        f.write(json.dumps(total_data))

    for key, value in total_data.items()[:1]:
        for i in range(11):
            data = value[i:i + 2]
            plt.plot(value[1], value[2], label=value[0])
            plt.savefig()
    # plt.show()


def cal_days(date1, date2):
    date1 = time.strptime(date1, "%Y%m%d")
    date2 = time.strptime(date2, "%Y%m%d")
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return (date2 - date1).days


def is_leap_year(date):
    """
    :param date:   str, "%Y%m%d"
    :return:
    """
    year = int(date[:4])
    if year % 100 == 0:
        return year % 400 == 0
    else:
        return year % 4 == 0


def duration_mean(data_path1, data_path2, save_path):
    data_2017 = generate_data(data_path1)
    data_2018 = generate_data(data_path2)
    for key, value in data_2017.items():
        five_mean_2017 = [value[i:i + 5] for i in [*range(len(value))][::5]]
        five_mean_2017 = [sum(i) / len(i) for i in five_mean_2017]
        plt.plot([i * 5 for i in range(len(five_mean_2017))], five_mean_2017, label="2017")

        value = data_2018.get(key)
        five_mean_2018 = [value[i:i + 5] for i in [*range(len(value))][::5]]
        five_mean_2018 = [sum(i) / len(i) for i in five_mean_2018]
        plt.plot([i * 5 for i in range(len(five_mean_2018))], five_mean_2018, label="2018")

        plt.legend()

        plt.savefig(save_path + key + '.png')
        plt.show()


def generate_data(data_path, default_value=0):
    df = pd.read_csv(data_path, delimiter=',')
    counts = []
    labels = []
    _counts = []

    for idx, (name, date, count) in df.iterrows():
        # print(idx, name, date, count)
        date = str(date)
        date_base = date[:4] + "0101"
        if name not in labels:
            labels.append(name)
            counts.append(_counts)
            if is_leap_year(date):
                _counts = [default_value] * 366
            else:
                _counts = [default_value] * 365
        _counts[cal_days(date_base, date)] = count
    counts.append(_counts)
    data = dict(zip(labels, counts[1:]))
    return data


if __name__ == '__main__':
    # dua = cal_days("20180201", "20180202")
    # print(dua)
    print(is_leap_year('2100'))

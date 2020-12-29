import config
import pandas as pd
from datetime import datetime, timedelta
import time


class ProcessData:
    def __init__(self,
                 data_path1=config.train_data2017_path,
                 data_path2=config.train_data2018_path,
                 data_path3=config.test_data2019_path,
                 data_path4=config.test_data2020_path,
                 save_data1=config.save_2_years_data,
                 save_data2=config.save_2_years_test_data,
                 start_date1='20170101',
                 start_date2='20180101',
                 start_date3='20190101',
                 start_date4='20200101',
                 ):
        self.data_2017 = self.generate_data(data_path1, None)
        self.data_2018 = self.generate_data(data_path2, None)
        self.data_2019 = self.generate_data(data_path3, None)
        self.data_2020 = self.generate_data(data_path4, None)
        self.save_data1 = save_data1
        self.save_data2 = save_data2
        self.start_date1 = start_date1
        self.start_date2 = start_date2
        self.start_date3 = start_date3
        self.start_date4 = start_date4

    def process_data(self):
        numbers = []
        with open(self.save_data1, 'w')as f:
            keys = list(self.data_2017.keys())
            f.write('时间,' + ','.join(keys) + '\n')
            for i in range(len(self.data_2017[keys[0]])):
                f.write(self.get_date(i, self.start_date1))
                for key in keys:
                    value = self.data_2017[key][i]
                    value = str(value) if value else str(0)
                    f.write(',' + str(value))
                    numbers.append(self.data_2017[key][i])
                f.write('\n')
            for i in range(len(self.data_2017[keys[0]])):
                f.write(self.get_date(i, self.start_date2))
                for key in keys:
                    value = self.data_2018[key][i]
                    value = str(value) if value else str(0)
                    f.write(',' + str(value))
                    numbers.append(self.data_2018[key][i])
                f.write('\n')

        with open(self.save_data2, 'w')as f:
            keys = list(self.data_2019.keys())
            f.write('时间,' + ','.join(keys) + '\n')
            for i in range(len(self.data_2019[keys[0]])):
                f.write(self.get_date(i, self.start_date3))
                for key in keys:
                    value = self.data_2019[key][i]
                    value = str(value) if value else str(0)
                    f.write(',' + str(value))
                    numbers.append(self.data_2019[key][i])
                f.write('\n')
            for i in range(len(self.data_2019[keys[0]])):
                f.write(self.get_date(i, self.start_date4))
                for key in keys:
                    value = self.data_2020[key][i]
                    value = str(value) if value else str(0)
                    f.write(',' + str(value))
                    numbers.append(self.data_2020[key][i])
                f.write('\n')

        numbers = list(set([number if number else 0 for number in numbers]))
        with open(config.vocab_path, 'w')as f:
            for number in numbers:
                f.write(str(number) + '\n')

    @staticmethod
    def get_date(days, base_time):
        """
        :param days: int , 15
        :param base_time: str, '20170101'
        :return:
        """
        base_time = datetime.strptime(base_time, '%Y%m%d')
        date = (base_time + timedelta(days=days)).strftime("%Y%m%d")
        return date

    def generate_data(self, data_path, default_value=0):
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
                if self.is_leap_year(date):
                    _counts = [default_value] * 366
                else:
                    _counts = [default_value] * 365
            _counts[self.cal_days(date_base, date)] = count
        counts.append(_counts)
        data = dict(zip(labels, counts[1:]))
        return data

    @staticmethod
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

    @staticmethod
    def cal_days(date1, date2):
        date1 = time.strptime(date1, "%Y%m%d")
        date2 = time.strptime(date2, "%Y%m%d")
        date1 = datetime(date1[0], date1[1], date1[2])
        date2 = datetime(date2[0], date2[1], date2[2])
        return (date2 - date1).days


if __name__ == '__main__':
    process_data = ProcessData()
    process_data.process_data()

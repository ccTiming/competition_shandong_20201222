import config
import pandas as pd
from datetime import datetime, timedelta
import time


class ProcessData:
    def __init__(self, data_path1=config.train_data2017_path, data_path2=config.train_data2018_path):
        self.data_2017 = self.generate_data(data_path1, None)
        self.data_2018 = self.generate_data(data_path2, None)

    def process_data(self):
        with open(config.save_2_years_data, 'w')as f:
            keys = list(self.data_2017.keys())
            f.write(','.join(keys) + ',时间\n')
            for i in range(len(self.data_2017[keys[0]])):
                for key in keys:
                    f.write(str(self.data_2017[key][i]) + ',')
                f.write(self.get_date(i, '20170101') + '\n')
            for i in range(len(self.data_2017[keys[0]])):
                for key in keys:
                    f.write(str(self.data_2018[key][i]) + ',')
                f.write(self.get_date(i, '20180101') + '\n')

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

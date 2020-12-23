from visualise import bar_chart, line_chart, duration_mean
import config

if __name__ == '__main__':
    duration_mean(config.train_data2017_path, config.train_data2018_path, config.save_5_days_mean_path)

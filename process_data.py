from trainer import data, storage
import json
from datetime import datetime
import os

if __name__ == '__main__':
    storage.set_bucket(os.getenv('BUCKET'))
    data_directory = 'working_dir/data/%s' % datetime.now().strftime('%Y_%m_%d_%H.%M')
    data_config = json.load(open('data_config.json'))
    data.process_data(data_config, data_directory)
    print('Data written to: %s' % data_directory)

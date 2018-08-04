# NOTE: This is for running locally only. Please see bin/run_cloud for
# initializing a distributed cloud job.
import json
from trainer import train, storage
from datetime import datetime
import os

if __name__ == '__main__':
  storage.set_bucket(os.getenv('BUCKET'))
  hypes = json.load(open('hypes.json'))
  output_dir = 'working_dir/runs/%s' % (datetime.now().strftime('job_%Y_%m_%d__%H_%M'))
  train.execute(hypes, output_dir)

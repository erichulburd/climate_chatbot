# NOTE: This is for running locally only. Please see bin/run_cloud for
# initializing a distributed cloud job.
import json
from trainer import train, storage, data
from datetime import datetime
import os

if __name__ == '__main__':
  storage.set_bucket(os.getenv('BUCKET'))
  hypes = json.load(open('hypes.json'))
  metadata = data.load_metadata(hypes)
  job_directory = 'working_dir/runs/%s' % (datetime.now().strftime('job_%Y_%m_%d__%H_%M'))
  train.execute(hypes, metadata, job_directory)

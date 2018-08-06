import matplotlib
matplotlib.use('Agg')

import argparse
import os
import json
from datetime import datetime

from trainer import model, data, storage
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

def execute(hypes, metadata, job_directory):
  data_directory = 'working_dir/data/%s' % (hypes['data_directory'])
  hypes['data'] = json.loads(storage.get('%s/config.json' % data_directory).decode('utf-8'))

  storage.write(json.dumps(hypes, indent=2, sort_keys=True), "%s/hypes.json" % job_directory)

  estimator = model.build_estimator(
        hypes,
        metadata,
        job_directory
    )

  train_input_fn = model.get_input_fn(
    hypes, ModeKeys.TRAIN
  )
  train_steps = hypes['epochs'] * data.length(data_directory, ModeKeys.TRAIN)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=train_steps
  )

  eval_input_fn = model.get_input_fn(
    hypes, ModeKeys.EVAL
  )
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hypes['eval_steps'],
      throttle_secs=hypes['eval_throttle_seconds']
    )

  # Run the training job
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec,
      eval_spec
  )

if __name__ == '__main__':
    # NOTE: This is entry point for distributed cloud execution.
    # Use {root}/train.py for running estimator locally.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hypes_path',
        help='Path to hypes on GCS for this job run.',
        default='/hypes.json'
    )
    parser.add_argument(
        '--bucket_name',
        help='Name of GCS bucket',
        required=True
    )
    parser.add_argument(
        '--job_directory',
        help='Name of job directory under working_dir/runs.',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    storage.set_bucket(arguments['bucket_name'])

    hypes = {
        'cell_fn': 'LSTM'
    }
    hypes.update(json.loads(storage.get(arguments['hypes_path']).decode('utf-8')))

    job_directory = 'working_dir/runs/%s' % (arguments['job_directory'])
    metadata = data.load_metadata(hypes)
    execute(hypes, metadata, job_directory)

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
# from tensorflow.estimator import stop_if_no_decrease_hook

"""
def early_stop_hook(estimator, hypes):
  return tf.estimator.stop_if_no_decrease_hook(
    estimator,
    metric_name=hypes['early_stopping']['metric_name'],
    max_steps_without_decrease=hypes['early_stopping']['max_steps_without_decrease'],
    min_steps=hypes['early_stopping']['min_steps']
  )
"""

def serving_exporter():
    # TODO
    return saved_model_export_utils.make_export_strategy(
        model.serving_input_fn,
        default_output_alternative_key=None,
        exports_to_keep=1
    )

def execute(hypes, output_dir):
  data_directory = 'working_dir/data/%s' % (hypes['data_directory'])
  hypes['data'] = json.loads(storage.get('%s/config.json' % data_directory))

  # save answer metatokens
  storage.write(json.dumps(hypes, indent=2, sort_keys=True), "%s/hypes.json" % output_dir)

  estimator = model.build_estimator(
        'gs://%s/%s' % (storage.bucket_name, output_dir),
        hypes
    )

  train_input_fn = model.get_input_fn(
    data_directory, hypes, ModeKeys.TRAIN
  )
  train_steps = hypes['epochs'] * data.length(data_directory, ModeKeys.TRAIN)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=train_steps,
      hooks = [
          # early_stop_hook(estimator, hypes)
      ]
  )

  eval_input_fn = model.get_input_fn(
        data_directory, hypes, ModeKeys.EVAL
  )
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hypes['eval_steps'],
      throttle_secs=hypes['eval_throttle_seconds']
      # exporters=[
      #   serving_exporter
      # ]
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
        required=True
    )
    parser.add_argument(
        '--bucket_name',
        help='Name of GCS bucket',
        required=True
    )
    parser.add_argument(
        '--working_directory',
        help='Name of job directory under working_dir/runs.',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    storage.set_bucket(arguments['bucket_name'])

    hypes = json.loads(storage.get('%s/hypes.json' % arguments['hypes_path']))

    output_dir = 'working_dir/runs/%s' % (arguments['working_directory'])
    execute(hypes, output_dir)

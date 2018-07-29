# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import json

from trainer import model, data
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

def early_stop_hook(estimator, hypes):
  return tf.contrib.estimator.stop_if_no_decrease_hook(
    estimator,
    metric_name=hypes['early_stopping']['metric_name'],
    max_steps_without_decrease=hypes['early_stopping']['max_steps_without_decrease'],
    min_steps=hypes['early_stopping']['min_steps']
  )

def serving_exporter():
    return saved_model_export_utils.make_export_strategy(
        model.serving_input_fn,
        default_output_alternative_key=None,
        exports_to_keep=1
    )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--base_path',
      help='Base file path',
      default=os.getcwd()
  )
  parser.add_argument(
      '--output_dir',
      help='Path to write checkpoints and export models',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default=''
  )
  args = parser.parse_args()
  arguments = args.__dict__

  base_path = arguments['base_path']
  hypes = json.load(open('%s/hypes.json' % base_path))
  data_directory = '%s/working_dir/data/%s' % (base_path, hypes['data_directory'])
  hypes['data'] = json.load(open('%s/config.json' % data_directory))
  output_dir = '%s/working_dir/runs/%s' % (base_path, arguments['working_directory'])

  # save answer metatokens
  with open("%s/hypes.json" % output_dir, "w") as f:
      json.dump(hypes, f, indent=2, sort_keys=True)

  estimator = model.build_estimator(
        output_dir,
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
          early_stop_hook(estimator, hypes)
      ]
  )

  eval_input_fn = model.get_input_fn(
        data_directory, hypes, ModeKeys.EVAL
  )
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hypes['eval_steps'],
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

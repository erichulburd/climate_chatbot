import tensorflow as tf
import numpy as np

from trainer import model

class EarlyStopHook(tf.train.SessionRunHook):
  def __init__(self, hypes, loss_op):
    self.losses = []
    self.loss_op = loss_op
    self.max_eval_steps_without_decrease = hypes['early_stopping']['max_eval_steps_without_decrease']
    self.min_eval_steps = hypes['early_stopping']['min_eval_steps']

  def before_run(self, run_context):
    print('EarlyStopHook#before_run')
    return tf.train.SessionRunArgs({ 'loss': self.loss_op })

  def after_run(self, run_context, run_values):
    print('EarlyStopHook#after_run')
    print(run_values)
    self.losses.append(run_values.results['loss'])
    if len(self.losses) <= self.min_eval_steps:
      return

    compare_index = -self.max_eval_steps_without_decrease - 1
    if (self.losses[compare_index] < self.losses[-1]):
      run_context.request_stop()
      return

class SaveVariablesHook(tf.train.SessionRunHook):
  def __init__(self, layer, path):
    self.layer = layer
    self.path = path

  def after_run(self, run_context, run_values):
    print('SaveVariablesHook#after_run')
    print(self.path)
    model.save_variables(run_context.session, self.layer, self.path)


default_seeds = [
    "How are you?",
    "What did you eat for breakfast this morning?"
    "What is the most potent greenhouse gas?",
    "Are climate models overestimating the risks of greenhouse gases?"
]
class ExampleSeedsEval(tf.train.SessionRunHook):
  def __init__(self, hypes, metadata, inference_model, seeds = default_seeds):
    self.seeds = seeds
    self.hypes = hypes
    self.metadata = metadata
    self.inference_model = inference_model

  def after_run(self, run_context, run_values):
    print('ExampleSeedsEval#after_run')
    responses = model.infer(run_context.session, self.seeds, self.hypes, self.metadata, self.inference_model)
    for index, seed in enumerate(self.seeds):
      print('$ %s' % seed)
      for j, response in enumerate(responses[index]):
        print('  %i. %s' % (j + 1, response))

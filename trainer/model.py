from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorlayer as tl
import tensorflow as tf
from trainer import log
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys
from tensorlayer.layers import EmbeddingInputlayer, Seq2Seq, DenseLayer, retrieve_seq_length_op2

from trainer import data

tf.logging.set_verbosity(tf.logging.INFO)


def get_input_fn(data_directory, hypes, mode):
  metadata, trainX, trainY, _, _, validX, validY = data.load_data(data_directory)

  X = trainX; Y = trainY; batch_size = hypes['batch_size']
  if mode == ModeKeys.EVAL:
    X = validX; Y = validY; batch_size = hypes['eval_batch_size']

  features, labels = data.preprocess_data(metadata, X, Y, hypes, mode)
  dataset = tf.data.Dataset.from_tensor_slices((features, labels))

  # Shuffle, repeat, and batch the examples.
  dataset_length = len(X)
  return lambda: dataset.shuffle(dataset_length).repeat().batch(batch_size).make_one_shot_iterator().get_next()

###============= model
def _model(encode_seqs, decode_seqs, hypes, mode):
  # We add two here for start, end ids as well as unknown and pad.
  xvocab_size = hypes['data']['vocab_size'] + 4

  reuse = (mode != ModeKeys.TRAIN)
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      # for chatbot, you can use the same embedding layer,
      # for translation, you may want to use 2 seperated embedding layers
      with tf.variable_scope("embedding") as vs:
        net_encode = EmbeddingInputlayer(
            inputs = encode_seqs,
            vocabulary_size = xvocab_size,
            embedding_size = hypes['emb_dim'],
            name = 'seq_embedding')
        vs.reuse_variables()
        # tl.layers.set_name_reuse(True) # remove if TL version == 1.8.0+
        net_decode = EmbeddingInputlayer(
            inputs = decode_seqs,
            vocabulary_size = xvocab_size,
            embedding_size = hypes['emb_dim'],
            name = 'seq_embedding')
      net_rnn = Seq2Seq(net_encode, net_decode,
            cell_fn = tf.contrib.rnn.BasicLSTMCell,
            n_hidden = hypes['emb_dim'],
            initializer = tf.random_uniform_initializer(-0.1, 0.1),
            encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
            decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
            initial_state_encode = None,
            dropout = (hypes['dropout'] if mode == ModeKeys.TRAIN else None),
            n_layer = hypes['seq2seq']['n_layer'],
            return_seq_2d = True,
            name = 'seq2seq')
      net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
  return net_out, net_rnn

def _generate_cnn_model_fn(hypes):
  def _cnn_model_fn(features, labels, mode):
    net_out, _ = _model(features['encode_seqs'], features['decode_seqs'], hypes, mode)

    loss = tl.cost.cross_entropy_seq_with_mask(
      logits=net_out.outputs,
      target_seqs=labels['target_seqs'],
      input_mask=labels['target_mask'],
      return_details=False,
      name='cost'
    )

    minimize = loss
    perplexity = tf.exp(loss)

    if hypes['minimize'] == 'perplexity':
      minimize = perplexity

    if mode == ModeKeys.TRAIN:
      train_op = tf.train.AdamOptimizer(learning_rate=hypes['lr']).minimize(minimize, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == ModeKeys.EVAL:
      eval_metric_ops = {}
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    elif mode == ModeKeys.INFER:
      pass
  return _cnn_model_fn


def build_estimator(model_dir, hypes):
  return tf.estimator.Estimator(
      model_fn=_generate_cnn_model_fn(hypes),
      model_dir=model_dir,
      config=tf.estimator.RunConfig(save_checkpoints_secs=180))

def serving_input_fn():
  inputs = {'inputs': tf.placeholder(tf.float32, [None, None])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

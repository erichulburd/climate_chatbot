from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorlayer as tl
import tensorflow as tf
import numpy as np
from tensorflow.estimator import ModeKeys
from tensorlayer.layers import EmbeddingInputlayer, Seq2Seq, DenseLayer, retrieve_seq_length_op2
import sys

from trainer import data, storage
from trainer.hooks import EarlyStopHook, ExampleSeedsEval, SaveVariablesHook

tf.logging.set_verbosity(tf.logging.INFO)


def get_input_fn(hypes, mode):
  metadata, trainX, trainY, _, _, validX, validY = data.load_data(hypes)

  X = trainX; Y = trainY; batch_size = hypes['batch_size']
  if mode == ModeKeys.EVAL:
    X = validX; Y = validY; batch_size = hypes['eval_batch_size']

  features, labels = data.preprocess_data(metadata, X, Y, hypes, mode)
  dataset = tf.data.Dataset.from_tensor_slices((features, labels))

  # Shuffle, repeat, and batch the examples.
  dataset_length = len(X)
  return lambda: dataset.shuffle(dataset_length).repeat().batch(batch_size).make_one_shot_iterator().get_next()

def save_variables(sess, layer, path):
    values = sess.run(layer.all_params)
    storage.write(
      np.ndarray.dumps(np.array(values)),
      path
    )

###============= model
def _model(encode_seqs, decode_seqs, hypes, mode):
  # We add two here for start, end ids as well as unknown and pad.
  xvocab_size = hypes['data']['vocab_size'] + 4

  reuse = (mode != ModeKeys.TRAIN)
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      # for chatbot, you can use the same embedding layer
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

def initialize_inference_model(hypes):
  encode_seqs_placeholder = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
  decode_seqs_placeholder = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
  net_out, net_rnn = _model(encode_seqs_placeholder, decode_seqs_placeholder, hypes, ModeKeys.EVAL)
  y = tf.nn.softmax(net_out.outputs)
  return {
    'net_out': net_out,
    'net_rnn': net_rnn,
    'encode_seqs_placeholder': encode_seqs_placeholder,
    'decode_seqs_placeholder': decode_seqs_placeholder,
    'y': y
  }

def infer(session, inputs, hypes, metadata, inference_model, top=5):
  responses = []
  for inpt in inputs:
    responses.append(
      _infer_input(session, inference_model, inpt, metadata, top=top)
    )
  return responses

def _infer_input(sess, inference_model, inpt, metadata, top):
  net_rnn = inference_model['net_rnn']
  encode_seqs_placeholder = inference_model['encode_seqs_placeholder']
  decode_seqs_placeholder = inference_model['decode_seqs_placeholder']
  y = inference_model['y']

  w2idx = metadata['w2idx']
  idx2w = metadata['idx2w']
  start_id = data.get_start_id(metadata)
  end_id = data.get_end_id(metadata)
  w2idx.update({ 'start_id': start_id, 'end_id': end_id })
  idx2w = idx2w + ['start_id', 'end_id']

  seed_id = [(w2idx[w] if w in w2idx else w2idx['unk']) for w in inpt.split(" ")]
  candidates = []
  for _ in range(top):  # 1 Query --> 5 Reply
      # 1. encode, get state
      state = sess.run(net_rnn.final_state_encode,
                      {encode_seqs_placeholder: [seed_id]})
      # 2. decode, feed start_id, get first word
      #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
      o, state = sess.run([y, net_rnn.final_state_decode],
                      {net_rnn.initial_state_decode: state,
                      decode_seqs_placeholder: [[start_id]]})
      w_id = tl.nlp.sample_top(o[0], top_k=3)
      w = idx2w[w_id]
      # 3. decode, feed state iteratively
      sentence = [w]
      for _ in range(30): # max sentence length
          o, state = sess.run([y, net_rnn.final_state_decode],
                          {net_rnn.initial_state_decode: state,
                          decode_seqs_placeholder: [[w_id]]})
          w_id = tl.nlp.sample_top(o[0], top_k=2)
          if w_id == end_id:
            break
          try:
            w = idx2w[w_id]
            if w in metadata['climate_metatokens']:
                w = metadata['climate_metatokens'][w]
            sentence = sentence + [w]
          except IndexError:
            print('IndexError for %i (out of bounds %i).' % (w_id, len(idx2w)))
            raise IndexError
      candidates.append(' '.join(sentence))
  return candidates

def _generate_cnn_model_fn(hypes, metadata, job_directory):
  def _cnn_model_fn(features, labels, mode):
    net_out, net_rnn = _model(features['encode_seqs'], features['decode_seqs'], hypes, mode)

    loss = tl.cost.cross_entropy_seq_with_mask(
      logits=net_out.outputs,
      target_seqs=labels['target_seqs'],
      input_mask=labels['target_mask'],
      return_details=False,
      name='cost'
    )

    minimize = loss
    perplexity = tf.exp(loss)
    tf.summary.scalar('perplexity', perplexity)

    if hypes['minimize'] == 'perplexity':
      minimize = perplexity


    if mode == ModeKeys.TRAIN:
      if hypes['optimizer'] == 'GradientDescent':
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, net_out.all_params), hypes['max_grad_norm'])
        optimizer = tf.train.GradientDescentOptimizer(hypes['lr'])
        train_op = optimizer.apply_gradients(zip(grads, net_out.all_params), global_step=tf.train.get_global_step())
        # train_op = tf.train.GradientDescentOptimizer().minimize(minimize, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
      else:
        train_op = tf.train.AdamOptimizer(learning_rate=hypes['lr']).minimize(minimize, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == ModeKeys.EVAL:
      inference_model = initialize_inference_model(hypes)
      eval_metric_ops = {}
      return tf.estimator.EstimatorSpec(mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=[
          # tf.train.LoggingTensorHook({ 'perplexity': perplexity }),
          SaveVariablesHook(net_out, '%s/%s' % (job_directory, data.NET_OUT_FILENAME)),
          SaveVariablesHook(net_rnn, '%s/%s' % (job_directory, data.NET_RNN_FILENAME)),
          ExampleSeedsEval(hypes, metadata, inference_model),
          EarlyStopHook(hypes, loss)
        ]
      )

    elif mode == ModeKeys.PREDICT:
      # This is an artifact of the infer functions running multiple operations in succession.
      raise ValueError('Estimator does not support prediction.')

  return _cnn_model_fn


def build_estimator(hypes, metadata, job_directory):
  return tf.estimator.Estimator(
      model_fn=_generate_cnn_model_fn(hypes, metadata, job_directory),
      model_dir='gs://%s/%s' % (storage.bucket_name, job_directory),
      config=tf.estimator.RunConfig(save_checkpoints_secs=180))

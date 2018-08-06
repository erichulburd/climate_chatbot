import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import sys
from trainer import data, model, storage

storage.set_bucket(os.getenv('BUCKET'))

class ModelServer:

    def __init__(self, job_id):
        hypes = data.load_hypes('working_dir/runs/%s' % job_id)
        self.hypes = hypes
        metadata = data.load_metadata(hypes)
        self.metadata = metadata

        net_out_values = np.loads(
            storage.get('working_dir/runs/%s/%s' % (job_id, data.NET_OUT_FILENAME))
        )
        net_rnn_values = np.loads(
            storage.get('working_dir/runs/%s/%s' % (job_id, data.NET_RNN_FILENAME))
        )

        sess = tf.Session()
        self.sess = sess
        inference_model = model.initialize_inference_model(hypes)
        self.inference_model = inference_model
        tl.files.assign_params(sess, net_out_values, inference_model['net_out'])
        tl.files.assign_params(sess, net_rnn_values, inference_model['net_rnn'])

    def respond(self, inpt):
        return model.infer(self.sess, [inpt], self.hypes, self.metadata, self.inference_model)

    def close(self):
        self.sess.close()

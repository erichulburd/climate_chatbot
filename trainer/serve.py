import tensorflow as tf
import os
from trainer import data, model

hypes = data.load_hypes(os.getenv('JOB_DIRECTORY'))
metatokens = data.load_metatokens(hypes)
metadata = data.load_metadata(hypes)

def respond(inpt, top=5):
    sess = tf.Session()
    responses = model.infer(sess, [inpt], hypes, metadata, load_from_storage=True)
    sess.close()
    return responses

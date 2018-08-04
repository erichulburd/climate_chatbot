# To authenticate from local environment see: https://cloud.google.com/storage/docs/authentication
#


from google.cloud import storage as gcs
import json
import os
import sys

bucket_name = None
def set_bucket(name):
    module = sys.modules[__name__]
    setattr(module, 'bucket_name', name)

def get(filename):
    client = gcs.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(filename)
    return blob.download_as_string()

def write(content, filename, content_type='application/json'):
    client  = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(content, content_type=content_type)

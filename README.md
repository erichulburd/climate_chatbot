# Commands

## Processing Data

* Raw data should be saved to `gs://$BUCKET/data`.
* Run `BUCKET=... python process_data.py` to save the tokenized data.
  * This will save the tokenized, processed data in pickle and .npy format to timestamped directory in `gs://$BUCKET/working_dir/data`.
    * The data_config.py will also be copied to this directory.
* This data will be loaded when training the model according to the `data_directory` in hypes.json.

## Editing and Saving Hyper Parameters

* Edit hypes.json locally.
* Run `BUCKET=... python save_hypes.py --path some_path` to save hypes.json to GCS.
* When you train the model, hypes.json will be saved to the specific job working directory so you can evaluate hyper-parameters compared to model performance.

## Training the Model

### Locally

* Run `BUCKET=... python train.py`.
* This will use the hypes.json file in your project root directory.

### Distributed on Google Cloud

* Run `bin/run_cloud $gc_project_name $path_to_hypes`.

# References

## Implementation

This code base is essentially a combination of:

* Tensorlayer's [seq2seq-chatbot](https://github.com/tensorlayer/seq2seq-chatbot)
* Google Cloud's [Using Distributed TensorFlow with Cloud ML Engine and Cloud Datalab ](https://cloud.google.com/ml-engine/docs/tensorflow/distributed-tensorflow-mnist-cloud-datalab).

Additionally, this code base:

* includes training on climate science question and answers
* configurable hyper parameters

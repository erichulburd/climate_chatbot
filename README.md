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


# Run Notes

* `job_2018_08_05__20_54` demonstrated good first and second derivatives, but after 10k training steps, this was still above 8 loss.

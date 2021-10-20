# Two Stage Open Workflow

Server and client code for running wearable cognitive assistance applications
created using Open Workflow Editor, that use the two stage processor.

Object detection is performed using the TensorFlow Object Detection API.
Fine-grained image classification is performed using
[Fast MPN-COV](https://github.com/akindofyoga/fast-MPN-COV).

## Creating Application

1. Create an application using the web tool at
   https://cmusatyalab.github.io/OpenWorkflow/
2. Every processor that you add must be of type "TwoStageProcessor." You can
   import the file `stirling.pbfsm`, from this repository, into the web tool to
   see an example.
3. This code will be run in a Docker container. All classifiers and detectors
   should be accessed using a volume map. The `classifier_path` that you specify
   for each processor must contain the files `model_best.pth.tar` and
   `classes.txt` that resulted from training the Fast MPN-COV model.
4. The `detector_path` must contain `saved_model.pb` and `variables` that result
   from running
   ```
   python object_detection/exporter_main_v2.py --input_type image_tensor \
   --pipeline_config_path <PATH TO pipeline.config> --trained_checkpoint_dir \
   <Path to model_dir> --output_directory <Where to save model>
   ```
   It should also contain a `label_map.pbtxt` file (such as the one created by
   [this](https://github.com/cmusatyalab/tfrecord-scripts/blob/master/merge_tfrecords.py)
   script.
5. `detector_class_name` is the name of the class from the object detector that
   should be classified.
6. `conf_threshold` is the minimum confidence score we require from the object
   detector. Any bounding boxes with a lower confidence score will be ignored.

## Installation

1. Install the Docker container for TensorFlow object detection by following
   [these](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#docker-installation)
   instructions.
2. Run `pip install gabriel-server` from inside the container.
3. Commit the container.
4. Run the container with the object detectors and classifiers accessible in the
   directories that you specified in the web editor.
5. Run `cd /path/to/this/repository/server` and then
   `python3 server.py /path/to/your/app.pbfsm`

## Client

1. Add the line `gabrielHost="THE_SERVER_HOST"` to
   `android-client/local.properties`
2. Run the client with Android Studio

## Protobuf

`server/wca_state_machine_pb2.py` was copied from
 https://github.com/cmusatyalab/OpenWorkflow/blob/master/gabrieltool/statemachine/wca_state_machine_pb2.py

 `server/owf_pb2.py` gets generated when the Android client is run.

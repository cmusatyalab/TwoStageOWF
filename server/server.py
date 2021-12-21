import ast
import argparse
import json
import logging
import io
import os
from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util

from torchvision import transforms
import torch

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

import credentials
import mpncov
import owf_pb2
import wca_state_machine_pb2


SOURCE = 'owf_client'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 1

DETECTOR_ONES_SIZE = (1, 480, 640, 3)


ALWAYS = 'Always'
HAS_OBJECT_CLASS = 'HasObjectClass'
CLASS_NAME = 'class_name'
TWO_STAGE_PROCESSOR = 'TwoStageProcessor'
CLASSIFIER_PATH = 'classifier_path'
DETECTOR_PATH = 'detector_path'
DETECTOR_CLASS_NAME = 'detector_class_name'
CONF_THRESHOLD = 'conf_threshold'

LABELS_FILENAME = 'classes.txt'
CLASSIFIER_FILENAME = 'model_best.pth.tar'
LABEL_MAP_FILENAME = 'label_map.pbtxt'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_State = namedtuple('_State', ['always_transition', 'has_class_transitions', 'processors'])
_Classifier = namedtuple('_Classifier', ['model', 'labels'])
_Detector = namedtuple('_Detector', ['detector', 'category_index'])


def _result_wrapper_for_transition(transition):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)

    logger.info('sending %s', transition.instruction.audio)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.payload = transition.instruction.audio.encode()
    result_wrapper.results.append(result)

    if len(transition.instruction.image) > 0:
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = transition.instruction.image
        result_wrapper.results.append(result)

    if len(transition.instruction.video) > 0:
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.VIDEO
        result.payload = transition.instruction.video
        result_wrapper.results.append(result)

    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.step = transition.next_state
    to_client_extras.zoom_result = owf_pb2.ToClientExtras.ZoomResult.NO_CALL

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def _result_wrapper_for(step, zoom_result):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.step = step
    to_client_extras.zoom_result = zoom_result

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def _start_zoom(step):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.zoom_result = owf_pb2.ToClientExtras.ZoomResult.CALL_START

    zoom_info = owf_pb2.ZoomInfo()
    zoom_info.app_key = credentials.ANDROID_KEY
    zoom_info.app_secret = credentials.ANDROID_SECRETs
    zoom_info.meeting_number = credentials.MEETING_NUMBER
    zoom_info.meeting_password = credentials.MEETING_PASSWORD

    to_client_extras.zoom_info = zoom_info

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


class _StatesModels:
    def __init__(self, fsm_file_path):
        self._states = {}
        self._classifiers = {}
        self._object_detectors = {}

        self._classifier_representation = {
            'function': mpncov.MPNCOV,
            'iterNum': 5,
            'is_sqrt': True,
            'is_vec': True,
            'input_dim': 2048,
            'dimension_reduction': None,
        }

        pb_fsm = wca_state_machine_pb2.StateMachine()
        with open(fsm_file_path, 'rb') as f:
            pb_fsm.ParseFromString(f.read())

        for state in pb_fsm.states:
            for processor in state.processors:
                self._load_models(processor)

            assert (state.name not in self._states), 'duplicate state name'
            always_transition = None
            has_class_transitions = {}

            for transition in state.transitions:
                assert (len(transition.predicates) == 1), 'bad transition'

                predicate = transition.predicates[0]
                if predicate.callable_name == ALWAYS:
                    always_transition = transition
                    break

                assert predicate.callable_name == HAS_OBJECT_CLASS, (
                    'bad callable')
                callable_args = json.loads(predicate.callable_args)
                class_name = callable_args[CLASS_NAME]

                has_class_transitions[class_name] = transition

            self._states[state.name] = _State(
                always_transition=always_transition,
                has_class_transitions=has_class_transitions,
                processors=state.processors)

        self._start_state = self._states[pb_fsm.start_state]

    def _load_models(self, processor):
        assert processor.callable_name == TWO_STAGE_PROCESSOR, 'bad processor'

        callable_args = json.loads(processor.callable_args)
        classifier_dir = callable_args[CLASSIFIER_PATH]

        if classifier_dir not in self._classifiers:
            labels_file = open(os.path.join(classifier_dir, LABELS_FILENAME))
            labels = ast.literal_eval(labels_file.read())

            freezed_layer = 0
            model = mpncov.Newmodel(self._classifier_representation,
                                    len(labels), freezed_layer)
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            trained_model = torch.load(os.path.join(classifier_dir,
                                                    CLASSIFIER_FILENAME))
            model.load_state_dict(trained_model['state_dict'])
            model.eval()

            self._classifiers[classifier_dir] = _Classifier(
                model=model, labels=labels)

        detector_dir = callable_args[DETECTOR_PATH]

        if detector_dir not in self._object_detectors:
            detector = tf.saved_model.load(detector_dir)
            ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
            detector(ones)

            label_map_path = os.path.join(detector_dir, LABEL_MAP_FILENAME)
            label_map = label_map_util.load_labelmap(label_map_path)
            categories = label_map_util.convert_label_map_to_categories(
                label_map,
                max_num_classes=label_map_util.get_max_label_map_index(
                    label_map),
                use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            self._object_detectors[detector_dir] = _Detector(
                detector=detector, category_index=category_index)

    def get_classifier(self, path):
        return self._classifiers[path]

    def get_object_detector(self, path):
        return self._object_detectors[path]

    def get_state(self, name):
        return self._states[name]

    def get_start_state(self):
        return self._start_state


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, fsm_file_path):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize,
        ])
        self._states_models = _StatesModels(fsm_file_path)
        self._on_zoom_call = False

    def handle(self, input_frame):
        to_server_extras = cognitive_engine.unpack_extras(
            owf_pb2.ToServerExtras, input_frame)

        step = to_server_extras.step
        if step == '':
            state = self._states_models.get_start_state()
        elif (to_server_extras.zoom_status ==
              owf_pb2.ToServerExtras.ZoomStatus.START):
            if self._on_zoom_call:
                return _result_wrapper_for(
                    step, owf_pb2.ToClientExtras.ZoomResult.EXPERT_BUSY)

            return _start_zoom(step)
        else:
            state = self._states_models.get_state(step)

        if state.always_transition is not None:
            return _result_wrapper_for_transition(state.always_transition)

        if len(state.processors) == 0:
            return _result_wrapper_for(step)

        assert len(state.processors) == 1, 'wrong number of processors'
        processor = state.processors[0]
        callable_args = json.loads(processor.callable_args)
        detector_dir = callable_args[DETECTOR_PATH]
        detector = self._states_models.get_object_detector(detector_dir)

        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detector(np.expand_dims(img, 0))

        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        im_height, im_width = img.shape[:2]

        classifier_dir = callable_args[CLASSIFIER_PATH]
        classifier = self._states_models.get_classifier(classifier_dir)

        pil_img = Image.open(io.BytesIO(input_frame.payloads[0]))

        conf_threshold = float(callable_args[CONF_THRESHOLD])
        detector_class_name = callable_args[DETECTOR_CLASS_NAME]
        for score, box, class_id in zip(scores, boxes, classes):
            class_name = detector.category_index[class_id]['name']
            if (score < conf_threshold) or (class_name != detector_class_name):
                continue
            logger.debug('found object')

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/research/object_detection/utils/visualization_utils.py#L1232
            ymin, xmin, ymax, xmax = box

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/official/vision/detection/utils/object_detection/visualization_utils.py#L192
            (left, right, top, bottom) = (
                xmin * im_width, xmax * im_width,
                ymin * im_height, ymax * im_height)

            cropped_pil = pil_img.crop((left, top, right, bottom))
            transformed = self._transform(cropped_pil).cuda()

            output = classifier.model(transformed[None, ...])
            _, pred = output.topk(1, 1, True, True)
            classId = pred.t()

            label_name = classifier.labels[classId]
            logger.info('Found label: %s', label_name)
            transition = state.has_class_transitions.get(label_name)
            if transition is None:
                continue

            return _result_wrapper_for_transition(transition)

        return _result_wrapper_for(
            step, owf_pb2.ToClientExtras.ZoomResult.NO_CALL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fsm_file_path', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(args.fsm_file_path)

    local_engine.run(
        engine_factory, SOURCE, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()

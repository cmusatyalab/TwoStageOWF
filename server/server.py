import argparse
import json
import logging
from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_FSM = namedtuple('_FSM', ['states', 'start'])
_State = namedtuple('_State', ['always_transition', 'has_class_transitions'])
_Transition = namedtuple('_Transition', ['instruction', 'next_state'])


def _get_fsm(fsm_file_path):
    pb_fsm = wca_state_machine_pb2.StateMachine()
    with open(fsm_file_path, 'rb') as f:
        pb_fsm.ParseFromString(f.read())

    states = {}
    for state in pb_fsm.states:
        assert (state.name not in states), 'duplicate state name'
        always_transition = None
        has_class_transitions = {}

        for transition in state.transitions:
            assert (len(transition.predicates) == 1), 'bad transition'
            transition = _Transition(
                instruction=transition.instruction,
                next_state=transition.next_state)

            predicate = transition.predicates[0]
            if predicate.callable_name == ALWAYS:
                always_transition = transition
                break

            assert predicate.callable_name == HAS_OBJECT_CLASS, 'bad callable'
            callable_args = json.loads(predicate.callable_args)
            class_name = callable_args[CLASS_NAME]

            has_class_transitions[class_name] = transition

        states[state.name] = _State(
            always_transition=always_transition,
            has_class_transitions=has_class_transitions)

    return _FSM(states=states, start=states[pb_fsm.start_state])


def _result_wrapper_for_transition(transition):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)

    logger.info('sending %s', transition.instruction.audio)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.payload = transition.instruction.audio.encode()
    result_wrapper.results.append(result)

    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.step = transition.next_state

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, fsm):
        self._fsm = fsm

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self._object_detector = tf.saved_model.load(OBJECT_DETECTOR_PATH)
        ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
        self._object_detector(ones)

        representation = {
            'function': mpncov.MPNCOV,
            'iterNum': 5,
            'is_sqrt': True,
            'is_vec': True,
            'input_dim': 2048,
            'dimension_reduction': None,
        }
        num_classes = 8
        freezed_layer = 0
        self._model = mpncov.Newmodel(representation, num_classes, freezed_layer)
        self._model.features = torch.nn.DataParallel(self._model.features)
        self._model.cuda()
        trained_model = torch.load(CLASSIFIER_PATH)
        self._model.load_state_dict(trained_model['state_dict'])
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize,
        ])

    def handle(self, input_frame):
        to_server_extras = cognitive_engine.unpack_extras(
            owf_pb2.ToServerExtras, input_frame)

        step = to_server_extras.step
        if step == '':
            self._fsm.start_state

        state = self._fsm.states[step]
        if state.always_transition is not None:
            return _result_wrapper_for_transition(state.always_transition)


        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fsm_file_path', type=str)
    args = parser.parse_args()

    fsm = _get_fsm(args.fsm_file_path)

    def engine_factory():
        return InferenceEngine()

    local_engine.run(
        engine_factory, SOURCE, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: owf.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='owf.proto',
  package='owf',
  syntax='proto3',
  serialized_options=_b('\n\016edu.cmu.cs.owfB\006Protos'),
  serialized_pb=_b('\n\towf.proto\x12\x03owf\"\x83\x01\n\x0eToServerExtras\x12\x0c\n\x04step\x18\x01 \x01(\t\x12\x33\n\x0bzoom_status\x18\x02 \x01(\x0e\x32\x1e.owf.ToServerExtras.ZoomStatus\".\n\nZoomStatus\x12\x0b\n\x07NO_CALL\x10\x00\x12\t\n\x05START\x10\x01\x12\x08\n\x04STOP\x10\x02\"\xb1\x01\n\x0eToClientExtras\x12\x0c\n\x04step\x18\x01 \x01(\t\x12 \n\tzoom_info\x18\x02 \x01(\x0b\x32\r.owf.ZoomInfo\x12\x33\n\x0bzoom_result\x18\x03 \x01(\x0e\x32\x1e.owf.ToClientExtras.ZoomResult\":\n\nZoomResult\x12\x0b\n\x07NO_CALL\x10\x00\x12\x0e\n\nCALL_START\x10\x01\x12\x0f\n\x0b\x45XPERT_BUSY\x10\x02\"a\n\x08ZoomInfo\x12\x0f\n\x07\x61pp_key\x18\x01 \x01(\t\x12\x12\n\napp_secret\x18\x02 \x01(\t\x12\x16\n\x0emeeting_number\x18\x03 \x01(\t\x12\x18\n\x10meeting_password\x18\x04 \x01(\tB\x18\n\x0e\x65\x64u.cmu.cs.owfB\x06Protosb\x06proto3')
)



_TOSERVEREXTRAS_ZOOMSTATUS = _descriptor.EnumDescriptor(
  name='ZoomStatus',
  full_name='owf.ToServerExtras.ZoomStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NO_CALL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='START', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STOP', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=104,
  serialized_end=150,
)
_sym_db.RegisterEnumDescriptor(_TOSERVEREXTRAS_ZOOMSTATUS)

_TOCLIENTEXTRAS_ZOOMRESULT = _descriptor.EnumDescriptor(
  name='ZoomResult',
  full_name='owf.ToClientExtras.ZoomResult',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NO_CALL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CALL_START', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EXPERT_BUSY', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=272,
  serialized_end=330,
)
_sym_db.RegisterEnumDescriptor(_TOCLIENTEXTRAS_ZOOMRESULT)


_TOSERVEREXTRAS = _descriptor.Descriptor(
  name='ToServerExtras',
  full_name='owf.ToServerExtras',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='step', full_name='owf.ToServerExtras.step', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zoom_status', full_name='owf.ToServerExtras.zoom_status', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TOSERVEREXTRAS_ZOOMSTATUS,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=150,
)


_TOCLIENTEXTRAS = _descriptor.Descriptor(
  name='ToClientExtras',
  full_name='owf.ToClientExtras',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='step', full_name='owf.ToClientExtras.step', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zoom_info', full_name='owf.ToClientExtras.zoom_info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zoom_result', full_name='owf.ToClientExtras.zoom_result', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TOCLIENTEXTRAS_ZOOMRESULT,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=153,
  serialized_end=330,
)


_ZOOMINFO = _descriptor.Descriptor(
  name='ZoomInfo',
  full_name='owf.ZoomInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='app_key', full_name='owf.ZoomInfo.app_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='app_secret', full_name='owf.ZoomInfo.app_secret', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meeting_number', full_name='owf.ZoomInfo.meeting_number', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meeting_password', full_name='owf.ZoomInfo.meeting_password', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=332,
  serialized_end=429,
)

_TOSERVEREXTRAS.fields_by_name['zoom_status'].enum_type = _TOSERVEREXTRAS_ZOOMSTATUS
_TOSERVEREXTRAS_ZOOMSTATUS.containing_type = _TOSERVEREXTRAS
_TOCLIENTEXTRAS.fields_by_name['zoom_info'].message_type = _ZOOMINFO
_TOCLIENTEXTRAS.fields_by_name['zoom_result'].enum_type = _TOCLIENTEXTRAS_ZOOMRESULT
_TOCLIENTEXTRAS_ZOOMRESULT.containing_type = _TOCLIENTEXTRAS
DESCRIPTOR.message_types_by_name['ToServerExtras'] = _TOSERVEREXTRAS
DESCRIPTOR.message_types_by_name['ToClientExtras'] = _TOCLIENTEXTRAS
DESCRIPTOR.message_types_by_name['ZoomInfo'] = _ZOOMINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ToServerExtras = _reflection.GeneratedProtocolMessageType('ToServerExtras', (_message.Message,), {
  'DESCRIPTOR' : _TOSERVEREXTRAS,
  '__module__' : 'owf_pb2'
  # @@protoc_insertion_point(class_scope:owf.ToServerExtras)
  })
_sym_db.RegisterMessage(ToServerExtras)

ToClientExtras = _reflection.GeneratedProtocolMessageType('ToClientExtras', (_message.Message,), {
  'DESCRIPTOR' : _TOCLIENTEXTRAS,
  '__module__' : 'owf_pb2'
  # @@protoc_insertion_point(class_scope:owf.ToClientExtras)
  })
_sym_db.RegisterMessage(ToClientExtras)

ZoomInfo = _reflection.GeneratedProtocolMessageType('ZoomInfo', (_message.Message,), {
  'DESCRIPTOR' : _ZOOMINFO,
  '__module__' : 'owf_pb2'
  # @@protoc_insertion_point(class_scope:owf.ZoomInfo)
  })
_sym_db.RegisterMessage(ZoomInfo)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)

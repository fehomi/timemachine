# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='service.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rservice.proto\"F\n\x0e\x46orwardRequest\x12\x11\n\tinference\x18\x01 \x01(\x08\x12\x0e\n\x06system\x18\x02 \x01(\x0c\x12\x11\n\tprecision\x18\x03 \x01(\t\"0\n\x0c\x46orwardReply\x12\x0e\n\x06\x64u_dls\x18\x01 \x01(\x0c\x12\x10\n\x08\x65nergies\x18\x02 \x01(\x0c\")\n\x0f\x42\x61\x63kwardRequest\x12\x16\n\x0e\x61\x64joint_du_dls\x18\x01 \x01(\x0c\"\x1f\n\rBackwardReply\x12\x0e\n\x06\x64l_dps\x18\x01 \x01(\x0c\"\x18\n\nWorkerInfo\x12\n\n\x02ip\x18\x01 \x01(\t\"\x0e\n\x0c\x45mptyMessage2m\n\x06Worker\x12/\n\x0b\x46orwardMode\x12\x0f.ForwardRequest\x1a\r.ForwardReply\"\x00\x12\x32\n\x0c\x42\x61\x63kwardMode\x12\x10.BackwardRequest\x1a\x0e.BackwardReply\"\x00\x32>\n\x0cRegistration\x12.\n\x0eRegisterWorker\x12\x0b.WorkerInfo\x1a\r.EmptyMessage\"\x00\x62\x06proto3'
)




_FORWARDREQUEST = _descriptor.Descriptor(
  name='ForwardRequest',
  full_name='ForwardRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='inference', full_name='ForwardRequest.inference', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='system', full_name='ForwardRequest.system', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='precision', full_name='ForwardRequest.precision', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=17,
  serialized_end=87,
)


_FORWARDREPLY = _descriptor.Descriptor(
  name='ForwardReply',
  full_name='ForwardReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='du_dls', full_name='ForwardReply.du_dls', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='energies', full_name='ForwardReply.energies', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=89,
  serialized_end=137,
)


_BACKWARDREQUEST = _descriptor.Descriptor(
  name='BackwardRequest',
  full_name='BackwardRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='adjoint_du_dls', full_name='BackwardRequest.adjoint_du_dls', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=139,
  serialized_end=180,
)


_BACKWARDREPLY = _descriptor.Descriptor(
  name='BackwardReply',
  full_name='BackwardReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dl_dps', full_name='BackwardReply.dl_dps', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=182,
  serialized_end=213,
)


_WORKERINFO = _descriptor.Descriptor(
  name='WorkerInfo',
  full_name='WorkerInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ip', full_name='WorkerInfo.ip', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=215,
  serialized_end=239,
)


_EMPTYMESSAGE = _descriptor.Descriptor(
  name='EmptyMessage',
  full_name='EmptyMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
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
  serialized_start=241,
  serialized_end=255,
)

DESCRIPTOR.message_types_by_name['ForwardRequest'] = _FORWARDREQUEST
DESCRIPTOR.message_types_by_name['ForwardReply'] = _FORWARDREPLY
DESCRIPTOR.message_types_by_name['BackwardRequest'] = _BACKWARDREQUEST
DESCRIPTOR.message_types_by_name['BackwardReply'] = _BACKWARDREPLY
DESCRIPTOR.message_types_by_name['WorkerInfo'] = _WORKERINFO
DESCRIPTOR.message_types_by_name['EmptyMessage'] = _EMPTYMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ForwardRequest = _reflection.GeneratedProtocolMessageType('ForwardRequest', (_message.Message,), {
  'DESCRIPTOR' : _FORWARDREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:ForwardRequest)
  })
_sym_db.RegisterMessage(ForwardRequest)

ForwardReply = _reflection.GeneratedProtocolMessageType('ForwardReply', (_message.Message,), {
  'DESCRIPTOR' : _FORWARDREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:ForwardReply)
  })
_sym_db.RegisterMessage(ForwardReply)

BackwardRequest = _reflection.GeneratedProtocolMessageType('BackwardRequest', (_message.Message,), {
  'DESCRIPTOR' : _BACKWARDREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:BackwardRequest)
  })
_sym_db.RegisterMessage(BackwardRequest)

BackwardReply = _reflection.GeneratedProtocolMessageType('BackwardReply', (_message.Message,), {
  'DESCRIPTOR' : _BACKWARDREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:BackwardReply)
  })
_sym_db.RegisterMessage(BackwardReply)

WorkerInfo = _reflection.GeneratedProtocolMessageType('WorkerInfo', (_message.Message,), {
  'DESCRIPTOR' : _WORKERINFO,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:WorkerInfo)
  })
_sym_db.RegisterMessage(WorkerInfo)

EmptyMessage = _reflection.GeneratedProtocolMessageType('EmptyMessage', (_message.Message,), {
  'DESCRIPTOR' : _EMPTYMESSAGE,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:EmptyMessage)
  })
_sym_db.RegisterMessage(EmptyMessage)



_WORKER = _descriptor.ServiceDescriptor(
  name='Worker',
  full_name='Worker',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=257,
  serialized_end=366,
  methods=[
  _descriptor.MethodDescriptor(
    name='ForwardMode',
    full_name='Worker.ForwardMode',
    index=0,
    containing_service=None,
    input_type=_FORWARDREQUEST,
    output_type=_FORWARDREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='BackwardMode',
    full_name='Worker.BackwardMode',
    index=1,
    containing_service=None,
    input_type=_BACKWARDREQUEST,
    output_type=_BACKWARDREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_WORKER)

DESCRIPTOR.services_by_name['Worker'] = _WORKER


_REGISTRATION = _descriptor.ServiceDescriptor(
  name='Registration',
  full_name='Registration',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=368,
  serialized_end=430,
  methods=[
  _descriptor.MethodDescriptor(
    name='RegisterWorker',
    full_name='Registration.RegisterWorker',
    index=0,
    containing_service=None,
    input_type=_WORKERINFO,
    output_type=_EMPTYMESSAGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_REGISTRATION)

DESCRIPTOR.services_by_name['Registration'] = _REGISTRATION

# @@protoc_insertion_point(module_scope)

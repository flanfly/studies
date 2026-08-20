"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 6, 33, 5, '', 'PublicBookTickerBatchV3Api.proto')
_sym_db = _symbol_database.Default()
from . import PublicBookTickerV3Api_pb2 as PublicBookTickerV3Api__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n PublicBookTickerBatchV3Api.proto\x1a\x1bPublicBookTickerV3Api.proto"q\n\x1aPublicBookTickerBatchV3Api\x12%\n\x05items\x18\x01 \x03(\x0b2\x16.PublicBookTickerV3Api\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x1b\n\x13lastOrderCreateTime\x18\x03 \x01(\x03BC\n\x1ccom.mxc.push.common.protobufB\x1fPublicBookTickerBatchV3ApiProtoH\x01P\x01b\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'PublicBookTickerBatchV3Api_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'\n\x1ccom.mxc.push.common.protobufB\x1fPublicBookTickerBatchV3ApiProtoH\x01P\x01'
    _globals['_PUBLICBOOKTICKERBATCHV3API']._serialized_start = 65
    _globals['_PUBLICBOOKTICKERBATCHV3API']._serialized_end = 178
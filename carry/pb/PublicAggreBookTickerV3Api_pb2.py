"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 6, 33, 5, '', 'PublicAggreBookTickerV3Api.proto')
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n PublicAggreBookTickerV3Api.proto"\x98\x01\n\x1aPublicAggreBookTickerV3Api\x12\x10\n\x08bidPrice\x18\x01 \x01(\t\x12\x13\n\x0bbidQuantity\x18\x02 \x01(\t\x12\x10\n\x08askPrice\x18\x03 \x01(\t\x12\x13\n\x0baskQuantity\x18\x04 \x01(\t\x12\x0f\n\x07version\x18\x05 \x01(\t\x12\x1b\n\x13lastOrderCreateTime\x18\x06 \x01(\x03BC\n\x1ccom.mxc.push.common.protobufB\x1fPublicAggreBookTickerV3ApiProtoH\x01P\x01b\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'PublicAggreBookTickerV3Api_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'\n\x1ccom.mxc.push.common.protobufB\x1fPublicAggreBookTickerV3ApiProtoH\x01P\x01'
    _globals['_PUBLICAGGREBOOKTICKERV3API']._serialized_start = 37
    _globals['_PUBLICAGGREBOOKTICKERV3API']._serialized_end = 189
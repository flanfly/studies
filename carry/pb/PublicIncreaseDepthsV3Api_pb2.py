"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 6, 33, 5, '', 'PublicIncreaseDepthsV3Api.proto')
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fPublicIncreaseDepthsV3Api.proto"\x99\x01\n\x19PublicIncreaseDepthsV3Api\x12+\n\x04asks\x18\x01 \x03(\x0b2\x1d.PublicIncreaseDepthV3ApiItem\x12+\n\x04bids\x18\x02 \x03(\x0b2\x1d.PublicIncreaseDepthV3ApiItem\x12\x11\n\teventType\x18\x03 \x01(\t\x12\x0f\n\x07version\x18\x04 \x01(\t"?\n\x1cPublicIncreaseDepthV3ApiItem\x12\r\n\x05price\x18\x01 \x01(\t\x12\x10\n\x08quantity\x18\x02 \x01(\tBB\n\x1ccom.mxc.push.common.protobufB\x1ePublicIncreaseDepthsV3ApiProtoH\x01P\x01b\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'PublicIncreaseDepthsV3Api_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'\n\x1ccom.mxc.push.common.protobufB\x1ePublicIncreaseDepthsV3ApiProtoH\x01P\x01'
    _globals['_PUBLICINCREASEDEPTHSV3API']._serialized_start = 36
    _globals['_PUBLICINCREASEDEPTHSV3API']._serialized_end = 189
    _globals['_PUBLICINCREASEDEPTHV3APIITEM']._serialized_start = 191
    _globals['_PUBLICINCREASEDEPTHV3APIITEM']._serialized_end = 254
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 6, 33, 5, '', 'PrivateAccountV3Api.proto')
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19PrivateAccountV3Api.proto"\xba\x01\n\x13PrivateAccountV3Api\x12\x11\n\tvcoinName\x18\x01 \x01(\t\x12\x0e\n\x06coinId\x18\x02 \x01(\t\x12\x15\n\rbalanceAmount\x18\x03 \x01(\t\x12\x1b\n\x13balanceAmountChange\x18\x04 \x01(\t\x12\x14\n\x0cfrozenAmount\x18\x05 \x01(\t\x12\x1a\n\x12frozenAmountChange\x18\x06 \x01(\t\x12\x0c\n\x04type\x18\x07 \x01(\t\x12\x0c\n\x04time\x18\x08 \x01(\x03B<\n\x1ccom.mxc.push.common.protobufB\x18PrivateAccountV3ApiProtoH\x01P\x01b\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'PrivateAccountV3Api_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'\n\x1ccom.mxc.push.common.protobufB\x18PrivateAccountV3ApiProtoH\x01P\x01'
    _globals['_PRIVATEACCOUNTV3API']._serialized_start = 30
    _globals['_PRIVATEACCOUNTV3API']._serialized_end = 216
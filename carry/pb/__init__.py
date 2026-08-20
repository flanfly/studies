"""MEXC V3 push/websocket protobuf messages.

Re-exports every generated ``*_pb2`` module from this package and promotes
each module's message class(es) to package level, so both of these work:

    from pb import PushDataV3ApiWrapper_pb2   # the generated module
    from pb import PushDataV3ApiWrapper       # the message class itself

New modules compiled into this directory are picked up automatically.
"""

from importlib import import_module
from pathlib import Path

__all__: list[str] = []

for _module_file in sorted(Path(__file__).parent.glob("*_pb2.py")):
    _module_name = _module_file.stem
    _module = import_module(f"{__name__}.{_module_name}")

    globals()[_module_name] = _module
    __all__.append(_module_name)

    for _name, _obj in vars(_module).items():
        if (
            _name != "DESCRIPTOR"
            and isinstance(_obj, type)
            # generated protobuf classes carry the unqualified module name
            and getattr(_obj, "__module__", None) == _module_name
        ):
            globals()[_name] = _obj
            __all__.append(_name)

if "_module_file" in globals():
    del _module_file, _module_name, _module, _name, _obj

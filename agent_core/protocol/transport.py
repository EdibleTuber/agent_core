"""Message transport: encode/decode + registration of message dataclasses."""
import json
from dataclasses import asdict

# asyncio StreamReader default is 64 KiB, which long NDJSON lines (e.g. /research
# results aggregated into a single response) can exceed. 16 MiB matches what PAL
# needs and is comfortable for any agent built on the same primitives.
STREAM_BUFFER_LIMIT = 16 * 1024 * 1024

_MESSAGE_TYPES: dict[str, type] = {}


def register_message(cls: type) -> type:
    """Register a dataclass type with the protocol registry. Returns the class
    unchanged so it can be used as a decorator or called directly."""
    type_field = cls.__dataclass_fields__["type"].default  # type: ignore[index]
    _MESSAGE_TYPES[type_field] = cls
    return cls


def encode_message(msg) -> bytes:
    """Serialize a registered message to a newline-terminated JSON bytes line."""
    return json.dumps(asdict(msg), ensure_ascii=False).encode("utf-8") + b"\n"


def decode_message(data: bytes):
    """Deserialize a JSON bytes line into a message object.

    Raises ValueError for unknown message types.
    """
    obj = json.loads(data)
    msg_type = obj.get("type")
    cls = _MESSAGE_TYPES.get(msg_type)
    if cls is None:
        raise ValueError(f"Unknown message type: {msg_type!r}")
    obj.pop("type", None)
    return cls(**obj)

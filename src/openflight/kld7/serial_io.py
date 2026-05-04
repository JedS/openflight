"""Serial I/O helpers for the K-LD7.

The kld7 Python library's default `_read_packet` calls
`serial.read(length)` exactly once. At 12M USB Full Speed the FTDI
driver splits large packets across USB microframes, so
`serial.read(length)` often returns whatever bytes are available
rather than the full requested length. The library treats the short
read as an error ("Failed to read all of reply") and continues with
a truncated packet, which then fails to parse and cascades into
"KLD7Exception: Wrong length reply" / drops the rest of the stream.

The patch below replaces `_read_packet` with a retrying version that
loops until it has the full payload (or the underlying read returns
zero bytes, which means timeout / EOF).

Both the live tracker and offline capture scripts apply this same
patch on every connected radar, so the choice of code path doesn't
change the recovery behaviour.
"""

from __future__ import annotations

import struct
from typing import Any


def install_robust_read_packet(radar: Any) -> None:
    """Replace ``radar._read_packet`` with a short-read-tolerant version.

    Args:
        radar: A connected ``kld7.KLD7`` instance.
    """
    # Lazy import so this module is safe to import on machines where
    # the kld7 library is not installed (CI, dev laptops).
    from kld7 import KLD7Exception  # type: ignore[import-not-found]

    def _robust_read_packet(device: Any):
        if device._port is None:
            raise KLD7Exception("serial port has been closed")
        header = device._port.read(8)
        if len(header) == 0:
            raise KLD7Exception("Timeout waiting for reply")
        if len(header) != 8:
            raise KLD7Exception("Wrong length reply")
        reply, length = struct.unpack("<4sI", header)
        reply = reply.decode("ASCII")
        if length != 0:
            payload = b""
            remaining = length
            while remaining > 0:
                chunk = device._port.read(remaining)
                if not chunk:
                    # Underlying read timed out / returned nothing; the
                    # caller decides whether to retry or surface this.
                    break
                payload += chunk
                remaining -= len(chunk)
        else:
            payload = None
        return reply, payload

    radar._read_packet = lambda: _robust_read_packet(radar)

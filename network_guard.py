
"""Block outbound TCP connections from this Python process.

Localhost connections, including a local Ollama server, are allowed.
Use this as a development/test guard, not as a complete OS firewall.
"""

import ipaddress
import socket


_original_connect = socket.socket.connect


def guarded_connect(sock, address):
    if isinstance(address, tuple) and address:
        host = address[0]

        try:
            ip = ipaddress.ip_address(host)
            is_loopback = ip.is_loopback
        except ValueError:
            # Allow only the explicit local Ollama host.
            is_loopback = host.lower() == "localhost"

        if is_loopback:
            return _original_connect(sock, address)

    raise RuntimeError(
        f"Blocked outbound TCP connection: {address!r}"
    )


def enable_network_guard():
    socket.socket.connect = guarded_connect

"""Host identity for training-rank records on a filesystem shared across nodes."""

import os
import socket
from pathlib import Path


def host_identity():
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "network_namespace": os.readlink("/proc/self/ns/net"),
    }

"""Compatibility Ray entry; use ``python -m espnet2.bin.rl_train``."""

from espnet2.rl.cli import main as _main
from espnet2.rl.cli import parser, resolve_nodes  # noqa: F401


def main():
    _main(worker="ray")


if __name__ == "__main__":
    main()

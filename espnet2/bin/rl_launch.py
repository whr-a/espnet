"""Compatibility entry; use ``python -m espnet2.bin.rl_train``."""

from espnet2.rl.cli import launch_command, main, parser  # noqa: F401

if __name__ == "__main__":
    main()

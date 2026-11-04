# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Task data: one module per task writes that task's training manifest.

A task module defines ``add_arguments(parser)`` for its own command-line
options and ``run(args)``, which writes the manifest and the audit receipt
beside it. :data:`TASKS` maps a task name to its module, so a new task is one
module plus one line here, and the command line picks it up::

    python -m espnet2.rl.tasks speaker_id --wav-scp data/train/wav.scp \\
        --utt2spk data/train/utt2spk --output data/rl/speaker_id.jsonl

Every record a Kaldi preparer writes carries ``reward_task``: the name, in
:data:`espnet2.rl.rewards.REWARDS`, of the reward that scores it.
"""

import argparse
import importlib

#: Task name -> the module that prepares its manifest.
TASKS = {
    "asr_wer": "espnet2.rl.tasks.asr",
    "longform_wer": "espnet2.rl.tasks.longform",
    "speaker_id": "espnet2.rl.tasks.speaker",
    "multispeaker_cpwer": "espnet2.rl.tasks.multispeaker",
    "avqa": "espnet2.rl.tasks.avqa",
}


def parser():
    result = argparse.ArgumentParser(
        prog="python -m espnet2.rl.tasks", description=__doc__.split("\n\n")[0]
    )
    tasks = result.add_subparsers(dest="task", required=True, metavar="TASK")
    for name, path in TASKS.items():
        module = importlib.import_module(path)
        summary = module.__doc__.strip().splitlines()[0]
        task = tasks.add_parser(name, help=summary, description=summary)
        module.add_arguments(task)
        task.set_defaults(run=module.run)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    return args.run(args)

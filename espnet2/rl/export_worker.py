# Copyright 2025 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export in the inference interpreter, preserving native ESPnet tensors."""

import argparse
import importlib.util
import json
from pathlib import Path

from .config import ESPnetRLConfig
from .utils import resolve_callable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = ESPnetRLConfig(**json.loads(Path(args.config).read_text()))
    resolve_callable(config.exporter)(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

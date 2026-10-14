# SGLang RL dependency profile

This profile fixes 229 public dependency artifacts for CPython 3.12 on Linux
x86_64 with CUDA 13.0 Torch wheels. It includes the complete fixed SGLang core,
ESPnet base dependencies, Liger 0.8.4 and Joblib. Every artifact has a SHA256
hash. The profile was resolved against SGLang
`4c3f163fe54634f20ef846e1ec0c507b27ae2430` and ESPnet
`296fefe8096ee347569b74f911e0f8ce68a27c9e`.

A fresh isolated environment installed this lock and both source editables;
`pip check`, Python Engine import, all four ESPnet architecture registrations
and CPU audio numerics passed on 2026-10-14. This is CPU environment acceptance;
GPU kernels, model inference and RL training use their separate run evidence.
The environment uses Torch/Torchaudio 2.11.0+cu130, Transformers 5.12.1,
NumPy 2.3.5 and SGLang Kernel 0.4.5. Linux must satisfy the wheel platform tags.

Install through `tools/installers/install_sglang_rl_env.sh`. The script creates
a new venv and refuses existing paths. It installs the artifacts with
`--require-hashes --no-deps`, then installs the supplied ESPnet and fixed SGLang
sources as editables with `--no-deps`; editable installation does not re-resolve
runtime packages. Complete ESPnet base requirements remain present.

Only `antlr4-python3-runtime==4.9.3` is a source dependency: Hydra requires
4.9.*, whose public releases have no wheel. Its pure Python wheel is built
during installation. Other dependencies are binary artifacts. The documented
SGLang build option `SGLANG_BUILD_RUST_EXTS=none` is the script's default for the
validated Python Engine profile; grpc/multimodal Rust extensions are outside
this validation. Another explicit build mode needs its Rust toolchain.
Editable package build dependencies are installed in temporary
build isolation and do not change runtime package versions.

Full `espnet[speechlm]` SFT extras conflict with this profile on Transformers
and Tilelang and are not installed. The Miles distribution is installed in the
native library environment through `install_miles_rl.sh`; its complete
distributed runtime is not part of this rollout profile. The Ray 2.58.0 and
Safetensors 0.8.0 artifacts here satisfy those pinned Miles dependency bounds.

The script accepts another hash-locked `--requirements` file when preparing a
separately reviewed environment. Do not treat changing the file as validation
of another platform, SGLang revision or GPU configuration. The default lock
requires its tested SGLang revision; a different revision requires an explicitly
supplied and separately reviewed requirements file.

# Native ESPnet / Miles asynchronous environment

This independent CPython 3.12/Linux x86_64/CUDA13 profile retains the synchronous profile's Torch 2.11/Transformers 5.12.1 stack and adds real Miles router 0.3.2, Polars 1.42.1 and PyLaTeXEnc 2.10. It replaces xxhash 4.0.1 with Miles-compatible 3.7.1 in this new lock only. All 233 installed artifacts are hash-bound; the official router wheel requires glibc 2.39. The source pins and lock hashes are in profile.json.

Miles is an unchanged pinned source checkout exposed through config.miles_source/PYTHONPATH. Its source and full eager imports are verified. Its distribution is not installed, so unused Megatron/ring-flash/torchft/Kubernetes extras are not claimed installed. `pip check` covers every installed package including ESPnet and SGLang. This is the native ESPnet runtime environment, not a general all-backend Miles Docker replacement.

The limited SGLang backport retains original dispatch, adds the exact upstream V41 encoder and three helpers needed by Miles imports, and has actual upstream parity tests. It does not enable full V41 serving. Original synchronous profile/locks and running frozen environments remain separate.

Install with tools/installers/install_miles_async_env.sh. It refuses an existing venv and requires clean exact source pins. The installer writes source/profile/import/pipcheck evidence beside the venv. Add `--with-tests` for separately hash-locked pytest dependencies; GPU training is never started by installation.

# Native ESPnet/Miles RL dependency profile

This hash-locked CPython 3.12/Linux x86_64/CUDA 13 profile supports synchronous
and bounded asynchronous native RL. Use the single
[environment guide](../../../../espnet2/rl/ENVIRONMENT.md) and
`tools/installers/install_miles_async_env.sh`; the historical script name does
not restrict the execution mode.

`profile.json` records unchanged Miles and reviewed SGLang source pins, all 233
artifact bindings and the glibc 2.39 requirement. Miles is imported from source;
its unused model-backend distribution extras are not installed. The installer
creates a fresh venv and records imports, source identities and `pip check`.
`--with-tests` adds test dependencies without running tests. Installation does
not establish GPU execution, numerical equivalence, memory fit or quality gain.

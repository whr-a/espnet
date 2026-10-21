# Tools for setup

<div align="left"><img src="../doc/image/environment_structure.png" width="700"/></div>


See also https://espnet.github.io/espnet/installation.html

## Contents
```
installers/         # Installation scripts for extra tools
Makefile            # Makefile to make an environment for experiments in espnet
check_install.py    # To check the status of installation
extra_path.sh       # Setup script for environment variables for extra tools
setup_cuda_env.sh   # Setup script for environment variables for cuda
setup_anaconda.sh   # To generate activate_python.sh with Anaconda environment
setup_miniforge.sh  # To generate activate_python.sh with conda-forge environment
setup_python.sh     # To generate activate_python.sh with the specified python
setup_venv.sh       # To generate activate_python.sh with venv of your Python
```

## Install

```sh
# ./setup_miniforge.sh <conda-root|default="venv"> <env-name|default=root env> <python-version|default=latest python>
# e.g.
./setup_miniforge.sh miniconda espnet 3.12
# If the conda-root already exists at the path, it tries to create it in the conda
# ./setup_miniforge.sh /some/where/miniconda espnet 3.12

# make TH_VERSION=<torch-ver|default=latest torch>
# e.g.
make
```

For more details: https://espnet.github.io/espnet/installation.html#step-2-installation-espnet



## Check installation

```sh
bash -c ". ./activate_python.sh; . ./extra_path.sh; python3 check_install.py"
```


## Caution

ESPnet recommends preparing a new Python environment independent of your pre-existing Python environment because pre-installed modules in your existing Python environment sometimes conflict with the dependencies of ESPnet. If you still want to install ESPnet in your original Python, all issues that might happen are under your responsibility.

## Optional speech RL environment

For the pinned native ESPnet/Miles/SGLang profile, follow the
[from-scratch speech training and RL guide](../espnet2/rl/ENVIRONMENT.md).
It uses `installers/install_miles_async_env.sh` to create a new isolated
environment and verifies source pins, dependency hashes and imports. Keep it
separate from full supervised SpeechLM extras. The guide also supplies an
[shared architecture and training guide](../espnet2/rl/README.md) and the current
validation limits. This feature requires the reviewed RL checkout.

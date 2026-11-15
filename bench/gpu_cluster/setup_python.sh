#!/usr/bin/env bash
# The scqubits + torch environment (the desktop's scqubits-bench pins), at
# /projects/elwe6267/software/anaconda/envs/scqubits-bench. Run once on a compile node:
#   acompile          (then, on the node:)   bash /projects/elwe6267/kronfree-bench-gpu/setup_python.sh
set -euo pipefail
module load miniforge
P=/projects/elwe6267/software/anaconda/envs/scqubits-bench
[ -x "$P/bin/python" ] || conda create -y -p "$P" python=3.12
"$P/bin/python" -m pip install --upgrade pip
"$P/bin/python" -m pip install numpy==2.2.6 scipy==1.14.1 qutip==5.1.1 scqubits==4.3.1 sympy==1.14.0 torch==2.15.0
"$P/bin/python" -c 'import torch, scqubits; print("torch", torch.__version__, "CUDA", torch.version.cuda, "scqubits", scqubits.__version__)'

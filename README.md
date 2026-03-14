From scratch implementation of the ScionLight optimizer from [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529) because the reference implementation was not yet available and I wanted to try it out for a ≈1.6B param training run on some local hardware.

ScionLight can be thought of as an alternative formulation of Muon with better scaling rules and a neat trick for grad accumulation memory use.

**Make sure not to zero grads between steps! This optimizer accumulates momentum in grads.**

See the self-contained [./scionlight.py](./scionlight.py) file for API.

The official reference implementation is now available at [github:LIONS-EPFL/scion](https://github.com/LIONS-EPFL/scion).

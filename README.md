# Effective Sparsification of Neural Networks with Global Sparsity Constraint

## Requirements:

```
Pytorch 1.4
Python 3.7.7
CUDA Version 10.1
pyyaml 5.3.1
tensorboard 2.2.1
torchvision 0.5.0
tqdm 4.50.2
```
## Setup
1. Set up a virtualenv with python 3.7.7 with conda.
2. Install the required packages.
3. Create a data directory as a base for all datasets, e.g., ./data/ in the code directory/
## Demo
```bash
python main.py --config configs/resnet32-cifar100-pr0.1.yaml --multigpu 0 --data dataset/ --prune-rate 0.1 --lr 6e-3
python main.py --config configs/resnet32-cifar100-pr0.1.yaml --multigpu 0 --data dataset/ --prune-rate 0.05 --lr 6e-3
python main.py --config configs/resnet32-cifar100-pr0.1.yaml --multigpu 0 --data dataset/ --prune-rate 0.02 --lr 6e-3
```
## Implementation
1. The implementation of ProbMaskConv can be found at utils/conv_type.py ProbMaskConv.
2. The implementation of Projection can be found at utils/net_utils.py, constrainScoreByWhole and solve_v_total.
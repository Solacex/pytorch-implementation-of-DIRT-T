# Pytorch-implementation-of-DIRT-T

Pytorch implementation of [A DIRT-T Approach to Unsupervised Domain Adaptation (ICLR 2018).](https://arxiv.org/abs/1802.08735)


## Dependencies

    pytorch = 1.0.0 
    tsnecuda
    tensorboardX
    tensorboard

## To run

To run VADA:
    python train.py

To run VADA without VAT:
    python train.py --no_vat

To run DIRT-T:
    python dirtt_train.py


## Results
SVHN->MNIST|  Paper  | Our implementation
-|-|- 
Source-only     |  82.4   | 80.36
VADA-no_vat     |  83.1   | 89.80
VADA            |  94.5   | 93.00
VADA+DIRT-T     |  99.4   | 99.50




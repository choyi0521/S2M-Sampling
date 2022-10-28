# S2M Sampling

Official implementation of "Mining Multi-Label Samples from Single Positive Labels" (NeurIPS 2022). [[arxiv]](https://arxiv.org/abs/2206.05764)

## Prerequisites

We tested our algorithm in the following environment:

```bash
Anaconda
python 3.9.7
pytorch 1.10.1
numpy 1.20.3
torchvision 0.11.2
```

## Running S2M Sampling

We prepare scripts for executing S2M sampling algorithm. To run the algorithm, you need to locate the checkpoints directory in the root directory of the project as follows:
```bash
┌── configs
├── scripts
├── src
└── checkpoints
    ├── GAN_cifar7to3
    ├── GAN_celeba7to3
    ├── cGAN_cifar7to3
    └── cGAN_celeba7to3
```

If you run the scripts below, sampling results on cifar7to3 and celeba7to3 will be saved in the results directory.

```bash
sh scripts/cgan_cifar7to3.sh
sh scripts/gan_cifar7to3.sh
sh scripts/cgan_celeba7to3.sh
sh scripts/gan_celeba7to3.sh
```
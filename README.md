# HSKD

**Multi-Level Feature Supervision through Mixup-Driven Hierarchical Self-Knowledge Distillation**

Published in *IEEE Transactions on Circuits and Systems for Video Technology* (TCSVT), 2026.
DOI: [10.1109/TCSVT.2026.3716467](https://doi.org/10.1109/TCSVT.2026.3716467)

This repository provides the official PyTorch implementation of HSKD for image classification.

## Acknowledgement

This project is built upon the open-source codebase of **MixSKD: Self-Knowledge Distillation from Mixup for Image Recognition** (ECCV 2022, [winycg/Self-KD-Lib](https://github.com/winycg/Self-KD-Lib)).

We sincerely thank the MixSKD authors for releasing their code. The training pipeline, model zoo, data pipeline and the baseline implementations under `methods/` are inherited from that release; the HSKD training scheme, the mask feature reconstruction branch and the contrastive branch are added on top of it. If you use this repository, please also consider citing MixSKD.

## Implementation overview

HSKD performs self-knowledge distillation that is driven by mixup, where supervision is applied over multiple feature levels and is supplied by the model of the previous epoch. Following `hskd.py`, each training iteration proceeds as follows.

Each sample is loaded as two independently augmented views, `inputs1` and `inputs2`. `inputs1` is fed through mixup (`alpha = 0.4`) to produce a mixed input together with a mixed label pair and the mixing coefficient `lam`.

The network returns both the classifier logits and its intermediate features. Three supervisory signals are then combined:

- **Hierarchical self-knowledge distillation.** The model snapshot saved at the end of the previous epoch (`lastmodel.pth.tar`) acts as the teacher. Its logits on `inputs2` and on the mixup-permuted view supervise the current model (`hist_loss`), and a soft target built from the one-hot labels mixed with the teacher's probabilities supervises the auxiliary branch (`mix_kl`). The mixing ratio is annealed over training as `alpha_t = alpha_T * (epoch + 1) / epochs`.
- **Mask feature reconstruction.** A reconstruction wrapper consumes the multi-level features of the current network and predicts from a masked feature representation; its output is distilled against the current classifier logits, so that all feature levels are supervised jointly (`aux_loss`).
- **Contrastive learning.** A projection head maps the final feature to a 128-dimensional embedding. The projected features of the mixup-mixed teacher view are contrasted against the projected student features with a supervised contrastive loss (`cl_loss`, temperature 0.3).

The overall objective is

```python
loss = loss_cls + 0.5 * (aux_loss + mix_kl) + 0.2 * cl_loss + 0.5 * hist_loss
```

where `loss_cls` is the mixup cross-entropy term. During the first epoch no teacher is available yet, so only `0.5 * aux_loss` is used.

## Requirements

- Ubuntu 18.04 LTS
- Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)
- CUDA 11.1
- PyTorch 1.12 + torchvision 0.13
- `einops`, `scipy`, `numpy`

## Data preparation

`--data` should point to the dataset root.

For CIFAR-100 / CIFAR-10, download and extract the archive so that the directory layout is:

```
<data>/cifar-100-python/
```

CIFAR-100 is available [here](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).

For the ImageNet-style datasets (`imagenet`, `CUB200`, `Dogs`, `MIT67`, `Flower`, `Air`, `Cars`), the loader expects a standard `ImageFolder` layout:

```
<data>/train/<class_name>/*.jpg
<data>/val/<class_name>/*.jpg
```

Two augmented views of every training image are produced automatically, so no extra flag is needed.

## Training

The training entry point is **`hskd.py`**.

```bash
python hskd.py \
    --dataset CIFAR-100 \
    --data ./data \
    --arch CIFAR_ResNet18 \
    --method hskd \
    --epochs 200 \
    --batch_size 128 \
    --init_lr 0.1 \
    --weight-decay 5e-4 \
    --warmup-epoch 5 \
    --kl_T 4.0 \
    --dl_T 4.0 \
    --gpu-id 0 \
    --manual_seed 0
```

`--method hskd` selects the HSKD objective; other values fall back to the corresponding baseline inherited from MixSKD. The default hyper-parameters (`--epochs 200`, `--batch_size 128`, `--init_lr 0.1`, `--weight-decay 5e-4`, `--kl_T 4.0`, `--dl_T 4.0`, `--alpha-T 0.8`, `--milestones 100 150`) already match the CIFAR-100 setting, so the command above only makes the important ones explicit.

For the fine-grained datasets use **`hskd_fg.py`**, which shares the HSKD objective but defaults to a smaller batch size and a weaker weight decay:

```bash
python hskd_fg.py \
    --dataset CUB200 \
    --data /path/to/CUB200 \
    --arch resnet18_imagenet \
    --method hskd \
    --epochs 200 \
    --batch_size 32 \
    --weight-decay 1e-4 \
    --gpu-id 0
```

Checkpoints and logs are written to `--checkpoint-dir` (default `./checkpoint/`), in a sub-directory named after the dataset, architecture, method, augmentation, seed and trial. Each epoch stores the current snapshot, keeps `lastmodel.pth.tar` for the next epoch's teacher, and copies the best model to `<arch>_best.pth.tar`.

To evaluate a trained checkpoint, pass `--evaluate` together with `--eval-checkpoint`.

## Repository structure

```
hskd.py                 HSKD training entry point (CIFAR / ImageNet-style)
hskd_fg.py              HSKD training entry point (fine-grained datasets)
main.py                 baseline runner inherited from MixSKD
main_kd.py              offline knowledge distillation runner
main_dtskd.py           DTSKD runner
losses.py               supervised contrastive loss
wrapper.py              mask feature reconstruction branch
wrapper_mgd.py          reconstruction wrapper used by hskd.py
wrapper_cl.py           contrastive projection head
models/                 ResNet, WRN, VGG, DenseNet, PyramidNet, MobileNetV2,
                        ShuffleNetV2, ResNeXt backbones
methods/                baseline self-KD and regularization methods
dataloader/             two-view data pipeline and augmentation policies
util/                   losses, metrics and training loops
scripts/                launch scripts inherited from MixSKD
```

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@ARTICLE{11622549,
  author={Zhao, Lei and Ng, Wing W. Y. and Zhang, Jianjun},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Multi-Level Feature Supervision through Mixup-Driven Hierarchical Self-Knowledge Distillation},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Labeling;Learning (artificial intelligence);Accuracy;Training;Lead;Design methodology;Distortion;Visualization;Testing;Self-Knowledge Distillation;Mask Feature Reconstruction;Contrastive Learning;Image Classification},
  doi={10.1109/TCSVT.2026.3716467}
}
```

## Acknowledgement

This repository is a derivative of the MixSKD open-source release ([winycg/Self-KD-Lib](https://github.com/winycg/Self-KD-Lib)). Thanks to the MixSKD authors for making their code publicly available, and to the authors of the self-knowledge distillation and data augmentation methods whose implementations are collected in `methods/`.

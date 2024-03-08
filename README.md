# Fine-grained classification for aero-engine borescope images based on the fusion of local and global features

This is code for Fine-grained classification for aero-engine borescope images based on the fusion of local and global features.

## Requirements

- python 3.6.5
- PyTorch >= 1.2.0
- torchvision >= 0.4.0

### Step

```
conda create -n pytorch18 python=3.8
conda activate pytorch18
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
conda install matplotlib pandas scipy opencv
```

## Train

change `root_dir` in the `config.py` and run `python train.py` 

> You can change other `hyper-parameters` in the `config.py`

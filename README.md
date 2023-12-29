# Learning based Multi-modality Image and Video Compression

## Quick Start

1. Install python dependencies:

   ``` sh
   pip install -r requirements.txt
   ```

2. Download **checkpoints** and **test datasets** from [Google Drive](https://drive.google.com/drive/folders/1l3vwGhHTwUK3xYsyxBE4fAV-yVeInDLO?usp=sharing)

3. Set PATH to the dataset in `config.sh`

4. Run test command:

   ``` sh
   python train.py --config config.json -m train_rgb --test -p model.pth.tar
   ```

## Commands

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json -m train_rgb -i ir.pth.tar -r rgb.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_rgb.py --config config.json
CUDA_VISIBLE_DEVICES=0 python train_ir.py --config config.json
```
### Finetune
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json -m train_rgb -p pretrain.pth.tar -i ir.pth.tar --finetune
```

### Testing

``` sh
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json -m train_rgb --test -p model.pth.tar
```


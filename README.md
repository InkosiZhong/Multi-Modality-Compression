# Learning based Multi-modality Image and Video Compression
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

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json --test -p model.pth.tar
```


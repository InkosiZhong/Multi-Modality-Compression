# Multi-Device Mix-Compression
## Training Script
### Multi GPU
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --config config.json -m train_rgb -i ir.pth.tar -r rgb.pth.tar
```

### Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json -m train_rgb -i ir.pth.tar -r rgb.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_rgb.py --config config.json
CUDA_VISIBLE_DEVICES=0 python train_ir.py --config config.json
```
### Finetune
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json -m train_rgb -p pretrain.pth.tar --finetune
CUDA_VISIBLE_DEVICES=0 python train_rgb.py --config config.json --finetune
CUDA_VISIBLE_DEVICES=0 python train_ir.py --config config.json --finetune
```

## Testing Script
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json --test -p model.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_rgb.py --config config.json --test -p rgb.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_ir.py --config config.json --test -p ir.pth.tar
```


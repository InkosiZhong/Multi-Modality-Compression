# Multi-Device Mix-Compression
## Training Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json
CUDA_VISIBLE_DEVICES=0 python train_rgb.py --config examples/example/config.json
CUDA_VISIBLE_DEVICES=0 python train_ir.py --config examples/example/config.json
```

## Testing Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json --test -p 模型路径
CUDA_VISIBLE_DEVICES=0 python train_rgb.py --config examples/example/config.json --test -p 模型路径
CUDA_VISIBLE_DEVICES=0 python train_ir.py --config examples/example/config.json --test -p 模型路径
```


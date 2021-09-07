# Multi-Device Mix-Compression
## Training Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json --rgb
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json --ir
```

## Testing Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json --test -p 模型路径
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json --test -p 模型路径 --rgb
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json --test -p 模型路径 --ir
```


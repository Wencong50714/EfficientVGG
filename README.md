## Project Struct

```
EfficientVGG/ 
├── data.py     # 数据加载和预处理工具 
├── vgg.py      # VGG模型实现 
├── train.py    # 模型训练脚本 
├── record.py   # 运行对比试验结果脚本
├── utility.py  # 模型信息统计工具
├── prune.py    # 模型剪枝
```

## Usage

```bash

python train.py

python record.py

```

## Record

#### Prune

Platform: 
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- GPU: 2080 Ti


| Model Type | Accuracy | Latency (CPU) | Size | MACs | Params |
| --- | --- | --- | --- | --- | --- |
| Raw model | 90.70% | 1.48 ms | 100.47 MiB | 404.74 M | 26.34 M |
| Fine Grained Prune | 90.63% | 1.49 ms | 20.86 MiB | 404.74 M | 26.34 M |
| Channel Prune (prune_ratio=0.3) | 89.22% | 1.47 ms | 63.33 MiB | 204.10 M | 16.60 M |

## Reference

My Code is modified from https://hanlab.mit.edu/courses/2024-fall-65940
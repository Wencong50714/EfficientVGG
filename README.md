## Project Struct

```
EfficientVGG/ 
├── data.py     # 数据加载和预处理工具 
├── vgg.py      # VGG模型实现 
├── train.py    # 模型训练脚本 
├── utility.py  # 模型信息统计工具
├── prune.py    # 模型剪枝
```

## Usage

```bash

python train.py

```

## Record

#### Prune

Platform: 
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- GPU: 2080 Ti


| Model Type | Accuracy | Latency (CPU) | Size | MACs | Params |
| --- | --- | --- | --- | --- | --- |
| Raw VGG19 | 90.70% | 2.41 ms | 100.47 MiB | 4047.37 M | 26.34 M |
| SA Fine Grained Prune | 90.63% | 2.40 ms | 20.86 MiB | 4047.37 M | 26.34 M |
| Unifrom Fine Grained Prune | 90.43 | 2.41 ms | 23.84 MiB | 4047.37 M | 26.34 M |
| Channel Prune (prune_ratio=0.3) | 89.22% | 1.85 ms | 63.33 MiB | 2040.97 M | 16.60 M |

#### Quantization

k-means Quantization

| Model Type | Accuracy | Latency (CPU) | Size | MACs | Params |
| --- | --- | --- | --- | --- | --- |
| Raw VGG19     | 90.70% | 2.41 ms | 100.47 MiB| 4047.37 M | 26.34 M |
| 8-bit k-means | 91.16% | 2.33 ms | 25.12 MiB | 4047.37 M | 26.34 M |
| 4-bit k-means | 89.59% | 2.41 ms | 12.56 MiB | 4047.37 M | 26.34 M |
| 2-bit k-means | 70.79% | 2.44 ms | 6.28 MiB  | 4047.37 M | 26.34 M |

#### mix

| Metric | Raw VGG19 | Efficient VGG | Improvement |
| --- | --- | --- | --- |
| Accuracy | 90.70% | 89.29% | 0.98× |
| Latency (CPU) | 2.41 ms | 1.84 ms | 1.31× |
| Size | 100.47 MiB | 15.83 MiB | 6.35× |
| MACs | 4047.37 M | 2040.97 M | 1.98× |
| Params | 26.34 M | 16.60 M | 1.59× |

- EfficentVGG = SA fine grained prune + channel prune + 8-bit k-means quantinization

## Reference

My Code is modified from https://hanlab.mit.edu/courses/2024-fall-65940
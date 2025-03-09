## Project Struct

```
EfficientVGG/ 
├── data.py     # 数据加载和预处理工具 
├── vgg.py      # VGG模型实现 
├── train.py    # 模型训练脚本 
├── utility.py  # 模型信息统计工具
├── prune.py    # 模型剪枝
├── linear_quantization.py # 线性量化
├── mix.py      # 综合剪枝量化手段
```

## Usage

```bash

pip install -r requiments.txt

python train.py

python prune.py

python linear_quantization.py

python mix.py

```

## Record

Platform: 
- GPU: 2080 Ti

#### Prune

| Model Type | Accuracy | Latency (CPU) | Size | MACs | Params |
| --- | --- | --- | --- | --- | --- |
| Raw VGG19 | 91.28% | 3.14 ms | 76.43 MiB | 3984.39 M | 20.24 M |
| SA Fine Grained Prune | 90.82% | 3.20 ms | 15.91 MiB | 3984.39 M | 20.24 M |
| Unifrom Fine Grained Prune | 90.47% | 3.17 ms | 19.36 MiB | 3984.39 M | 20.24 M |
| Channel Prune (prune_ratio=0.3) | 89.50% | 3.41 ms | 39.29 MiB | 1978.00 M | 10.30 M |

#### Linear Quantization

| Model Type | Accuracy | Latency (CPU) | Size | MACs | Params |
| --- | --- | --- | --- | --- | --- |
| Raw VGG19 | 91.28% | 3.14 ms | 76.43 MiB | 3984.39 M | 20.24 M |
| 8-bit Linear Qnt | 91.22% | 8.48 ms | 19.10 MiB | 3984.40 M | 20.23 M |

#### mix

|  | model 1 | model 2 | Improvement |
| --- | --- | --- | --- |
| Model Type | Raw VGG19 | Mixed Prune & Qnt model | - |
| Accuracy | 91.28% | 89.43% | - |
| Latency (CPU) | 3.14 ms | 9.23 ms | ❌ |
| Size | 76.43 MiB | 2.04 MiB | **37.5 x** |
| MACs | 3984.39 M | 1978.00 M | 2.01 x |
| Params | 20.24 M | 10.30 M | 1.96 x |

- 注1：model2 在计算 size 中没有计算值为 0 的权重元素
- 注2：线性量化实际计算时仍采用了 float32 类型，增加了类型转换的开销，导致 latency增加
- TOOD: 将线性量化的计算部分，采用硬件 int8 计算库
## Reference

My Code is modified from https://hanlab.mit.edu/courses/2024-fall-65940

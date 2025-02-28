## Project Struct

```
EfficientVGG/ 
├── data.py # Data loading and preprocessing utilities 
├── vgg.py # VGG model implementation 
├── train.py # Training script for models 
├── record.py # Evaluation and result visualization 
├── model.ipynb # Tutorial notebook for model exploration 
├── requirements.txt # Dependencies for the project 
├── models/ # Directory for saved model weights 
└── README.md # Project documentation
```

## Usage

```bash

python train.py

python record.py

```

## Record

| Model | Test Acuuracy |
| ---- | ---- |
| Raw VGG19 | 90.99% |
| Pruned VGG19|  |


## Reference

My Code is modified from https://hanlab.mit.edu/courses/2024-fall-65940
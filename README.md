# Filter Pruning via Measuring Feature Map Information

Dataset： CIFAR10/100  ImageNet

CIFAR: VGG ResNet DenseNet
ImageNet: ResNet

## Result of CIFAR10
### VGG16

| Model | Acc(%) | FLOPs | Parameters |
| ---- | ---- | ---- | ---- |
| VGG16 | 93.90 | 14.72M | 313.75M |
| our_E | 93.53 | 0.99M | 83.96M |
| our_P | 93.47 | 0.93M | 89.02M |
| our_P | 93.16 | 0.90M | 79.85M |

### VGG19
| Model | Acc(%) | FLOPs  | Parameters |
| ----- | ------ | ------ | ---------- |
| VGG19 | 93.68  | 20.04M | 398.74M    |
| our_E | 93.63  | 1.55M  | 129.21M    |
| our_P | 93.58  | 1.45M  | 127.44M    |

### ResNet56
| Model    | Acc(%) | FLOPs | Parameters |
| -------- | ------ | ----- | ---------- |
| ResNet56 | 93.22  | 0.85M | 126.55M    |
| our_E    | 93.56  | 0.39M | 69.52M     |
| our_P    | 93.36  | 0.39M | 63.15M     |
| our_P    | 93.09  | 0.31M | 59.66M     |
### ResNet164
| Model     | Acc(%) | FLOPs | Parameters |
| --------- | ------ | ----- | ---------- |
| ResNet164 | 95.04  | 1.71M | 254.50M    |
| our_E     | 94.66  | 0.67M | 111.33M    |
| our_P     | 93.65  | 0.73M | 105.86M    |
### DenseNet40
| Model      | Acc(%) | FLOPs | Parameters |
| ---------- | ------ | ----- | ---------- |
| DenseNet40 | 94.26  | 1.06M | 290.13M    |
| our_E      | 94.04  | 0.38M | 110.72M    |
| our_P      | 93.75  | 0.37M | 100.12M    |

## Result of CIFAR100
### VGG16
| Model | Acc(%) | FLOPs  | Parameters |
| ----- | ------ | ------ | ---------- |
| VGG16 | 73.80  | 14.77M | 313.8M     |
| our_E | 73.17  | 4.94M  | 150.70M    |
| our_E | 73.06  | 4.05M  | 129.52M    |
| our_P | 73.17  | 4.09M  | 147.99M    |
### VGG19
| Model | Acc(%) | FLOPs  | Parameters |
| ----- | ------ | ------ | ---------- |
| VGG19 | 73.81  | 20.08M | 398.79M    |
| our_E | 73.29  | 4.21M  | 183.69M    |
| our_P | 73.15  | 4.17M  | 195.77M    |
| our_P | 73.01  | 3.94M  | 180.51M    |
### ResNet56
| Model    | Acc(%) | FLOPs | Parameters |
| -------- | ------ | ----- | ---------- |
| ResNet56 | 71.77  | 0.86M | 71.77M     |
| our_E    | 71.28  | 0.50M | 80.48M     |
| our_E    | 70.67  | 0.41M | 69.88M     |
### ResNet164
| Model     | Acc(%) | FLOPs | Parameters |
| --------- | ------ | ----- | ---------- |
| ResNet164 | 76.74  | 1.73M | 253.97M    |
| our_E     | 76.28  | 0.94M | 150.57M    |
| our_P     | 75.27  | 0.94M | 123.09M    |
### DenseNet40
| Model      | Acc(%) | FLOPs | Parameters |
| ---------- | ------ | ----- | ---------- |
| DenseNet40 | 74.37  | 1.11M | 287.75M    |
| our_E      | 74.50  | 0.40M | 109.55M    |
| our_E      | 73.74  | 0.34M | 95.79M     |
| our_P      | 74.25  | 0.39M | 108.81M    |
| our_P      | 73.62  | 0.34M | 94.84M     |



## Result of ImageNet
### ResNet50
| Model    | Top1% | Top5% | FLOPs | Parameters |
| -------- | ----- | ----- | ----- | ---------- |
| ResNet50 | 76.15 | 92.87 | 4.09B | 25.50M     |
| our_E    | 72.02 | 90.69 | 1.84B | 11.41M     |
| our_E    | 70.41 | 89.91 | 1.41B | 8.51M      |
| our_P    | 69.91 | 89.46 | 1.70B | 11.06M     |
| our_P    | 68.62 | 88.62 | 1.34B | 8.23M      |

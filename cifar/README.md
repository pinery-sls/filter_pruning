## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg`,`resnet` or
`densenet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python main.py --dataset cifar10 --arch vgg --depth 19
python main.py --dataset cifar10 --arch resnet --depth 164
python main.py --dataset cifar10 --arch densenet --depth 40
```

## Train with Sparsity

```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19
python main_resnet56.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 56
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40
```

## Prune-entory

```shell
python vggprune-e.py --dataset cifar10 --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python resprune-e.py --dataset cifar10 --depth 164 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python resprune56-e.py --dataset cifar10 --depth 56 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python denseprune-e.py --dataset cifar10 --depth 40 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.

## Prune-parallel

```shell
python vggprune-p.py --dataset cifar10 --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python resprune-p.py --dataset cifar10 --depth 164 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python resprune56-p.py --dataset cifar10 --depth 56 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python denseprune-p.py --dataset cifar10 --depth 40 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 164
python main_resnet56.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 56
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch densenet --depth 40
```

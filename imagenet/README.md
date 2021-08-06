# Train
```
python main50.py --arch resnet50 --s 0.00001 --save [PATH TO SAVE RESULTS] [IMAGENET]
```

# Prune
```
python prune50.py --percent 0.5 --model [PATH TO THE BASE MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

# Finetune
```
python main_finetune50.py --arch vgg11_bn --refine [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

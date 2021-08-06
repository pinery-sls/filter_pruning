import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
import math
from numpy import *
from tqdm import tqdm

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'], False)
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# print(model)

def fun(a):
    mask = []
    for x in a:
        x = (x - min(a))/(max(a) - min(a))
        mask.append(x)
    # mask1 = [math.log10(x) for x in a]
    # tmax = np.log10(max(a))  # 最大值的对数
    # mask = np.array(mask1) / tmax
    return mask

def softmax(z):
    z = np.array(z)
    z1 = np.mean(z)
    z = (np.array(z) - z1) ** 2
    max = np.max(z)
    z = np.exp(z-max)  # 求e^zi值
    softmax_z = z / np.sum(z)
    return softmax_z


def entory(z):
    e = 0
    for i in z.flat:
        if i > 0:
            e += i * math.log(i)
    e_num = -e
    return e_num

def matrix(z,b):
    l = len(z)
    bb = np.array(b)
    mat = np.zeros((l,l))
    mat[0,:] = 1
    mat[1:,0] = z[0]
    for i in range(1, l):
        mat[i,i] = z[i]
    # print('转置 = ', bb.reshape(bb.shape[0], 1))
    # print('转置 = ', np.linalg.inv(np.array(mat)))
    result = np.matmul(np.linalg.inv(np.array(mat)), bb.reshape(bb.shape[0], 1))
    return list(result.T[0])

import collections.abc
def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

total = 0
covflag = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
        covflag += 1
bn = torch.zeros(total)

total_entory = []
flag = 0
layer_idx = 0
# 定义 forward hook function
def hook_fn_forward(module, input, output):
    global layer_idx
    global flag

    a = output.shape[0]
    b = output.shape[1]
    # coventory = torch.tensor([entory(F.softmax(output[i, j, :, :].view(1, -1), dim=1).cpu().detach().numpy()) for i in range(a) for j in range(b)])
    coventory = torch.tensor([entory(softmax(output[i,j,:,:].cpu().detach().numpy().astype(float))) for i in range(a) for j in range(b)])

    coventory = coventory.view(a, -1).numpy()
    coventory = coventory.sum(0).tolist()
    if flag == 0:
        total_entory.append(coventory)
    else:
        total_entory[layer_idx] = (np.array(total_entory[layer_idx]) + np.array(coventory)).tolist()
        layer_idx += 1


modules = model.named_modules()  #
# modules = model.named_children()
for name, module in modules:
    if isinstance(module, nn.BatchNorm2d):
        module.register_forward_hook(hook_fn_forward)

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test1(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= 1:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

pbar = tqdm(total=100)
for i in range(5):
    test1(model)
    flag = 1
    layer_idx  = 0
    pbar.update(20)
pbar.close()

print('长度 = ', len(total_entory))

# 进行归一化
n = 0
index = 0
total1 = []
layer_num = []
while (n < covflag):
    size = len(total_entory[n])
    weight_copy = fun(list(np.array(total_entory[n])))
    # print(weight_copy)
    # layer_num.append(np.mean(np.array(weight_copy)))
    total1.append(weight_copy)
    bn[index:(index+size)] = torch.FloatTensor(weight_copy)
    index += size
    n += 1

# b = np.zeros(len(layer_num))
# # print('b = ', b)
# b[0] = total*args.percent
# layernum = matrix(layer_num, list(b))
# print('layer number = ', layernum)



y, i = torch.sort(bn, descending=True)
thre_index = int(total * args.percent)
thre = y[thre_index]
print('阈值 = ', thre)

# def get_pruning_strategy(min_rate, max_rate, num_layer):
#     channel_config = np.random.rand(num_layer)
#     channel_config = channel_config * max_rate
#     channel_config = channel_config + min_rate
#     channel_config = np.around(np.array(channel_config), 1)
#     return channel_config
#
# channel_config = get_pruning_strategy(min_rate=0, max_rate=0.9, num_layer=flag)


pruned = 0
cfg = []
cfg_mask = []
layer = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        # y, i = torch.sort(torch.FloatTensor(total_entory[layer]), descending=True)
        # # print('weight_copy = ', weight_copy)
        # thre_index = int(len(total_entory[layer])*channel_config[layer])
        # thre1 = y[thre_index]
        # print('wwwwwww = ', total_entory[layer])
        # thre1 = np.mean(total_entory[layer]) * 0.75
        mask = [1 if i < thre else 0 for i in total1[layer]]
        mask = torch.FloatTensor(mask).float().cpu()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
        layer += 1
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total
print('剪枝率 = ', pruned_ratio.item())
print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

print(cfg)
newmodel = vgg(dataset=args.dataset, cfg=cfg)

print('    Total params: %.2fM' % (sum(p.numel() for p in newmodel.parameters())/1000000.0))

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune_vgg16666.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write('熵 = ' + "\n")
    fp.write(str(total_entory)+"\n")
    fp.write('正则化熵 = ' + "\n")
    fp.write(str(total1) + "\n")

    # fp.write("Test accuracy: \n"+str(acc))

# Make real prune
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned_vgg16100our1.pth'))

# print(newmodel)
model = newmodel
test(model)

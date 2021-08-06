import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.preresnet56 import resnet
from thop import profile, clever_format
import math
from tqdm import tqdm


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
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

model = resnet(depth=args.depth, dataset=args.dataset)
# print(model)
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

def fun(a):
    mask = []
    for x in a:
        x = (x - min(a))/(max(a) - min(a))
        mask.append(x)
    return mask

total = 0
covflag = 0
flag1 = 1
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        covflag += 1
        if flag1 == 0:
            flag1 = 1
            total += m.weight.data.shape[0]
        else:
            flag1 = 0


bn1 = torch.zeros(total)
bn2 = torch.zeros(total)
index = 0
flag1 = 1
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if flag1 == 0:
            flag1 = 1
            size = m.weight.data.shape[0]
            bn1[index:(index+size)] = m.weight.data.abs().clone()
            index += size
        else:
            flag1 = 0

y1, i = torch.sort(bn1)
thre_index1 = int(total * args.percent)
thre1 = y1[thre_index1]
print("slimming = ", thre1.item())

total_entory = []
layer_idx = 0
flag = 0
# 定义 forward hook function
def hook_fn_forward(module, input, output):
    global layer_idx
    global flag
    global total_entory

    if isinstance(module, nn.Conv2d):
        # print('output shape = ', output.shape)
        a = output.shape[0]
        b = output.shape[1]    # print('output = ', entory(F.softmax(output[0, 0, :, :].cpu().view(1, -1), dim=1).detach().numpy()))
        # print('output1 = ', entory(softmax(output[0, 0, :, :].cpu().detach().numpy())))
        coventory = torch.tensor([entory(softmax(output[i, j, :, :].cpu().detach().numpy().astype(float))) for i in range(a) for j in range(b)])
        # coventory = torch.tensor(
        #     [entory(F.softmax(output[i, j, :, :].view(1, -1), dim=1).cpu().detach().numpy()) for i in range(a) for j in
        #      range(b)])

        coventory = coventory.view(a, -1).numpy()
        coventory = coventory.sum(0).tolist()
        if flag == 0:
            total_entory.append(coventory)
        else:
            total_entory[layer_idx] = (np.array(total_entory[layer_idx]) + np.array(coventory)).tolist()
            layer_idx += 1


# modules = model.named_children()
modules = model.named_modules()
for name, module in modules:
    module.register_forward_hook(hook_fn_forward)



def test1(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
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
# pbar.set_description("pruning")
for i in range(5):
    test1(model)
    layer_idx = 0
    flag = 1
    pbar.update(20)
pbar.close()

# print('长度 = ', len(total_entory))
# print('covflag = ', covflag)


#归一化
# print('熵 = ', total_entory)
n = 0
index = 0
total1 = []
while (n < covflag):
    # print('===========', total_entory[n])
    if n % 2 != 0:
        size = len(total_entory[n])
        # print('weight = ', len(total_entory[n]))
        weight_copy = fun(total_entory[n])
        total1.append(weight_copy)
        bn2[index:(index+size)] = torch.FloatTensor(weight_copy)
        index += size
    n += 1


y, i = torch.sort(bn2, descending=True)
thre_index = int(total * args.percent)
thre2 = y[thre_index]
print('阈值 = ', thre2.item())

pruned = 0
cfg = []
cfg_mask = []
flag = 1
layer1 = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if flag == 0:
            flag = 1
            weight_copy1 = m.weight.data.abs().clone()
            mask1 = weight_copy1.gt(thre1).float().cuda()

            weight_copy = total1[layer1]
            mask2 = [1 if i < thre2 else 0 for i in weight_copy]
            mask2 = torch.FloatTensor(mask2)

            if (torch.sum(mask1) <= torch.sum(mask2) and torch.sum(mask1) > 0 and torch.sum(mask2) > 0):
                mask = mask1
            elif (torch.sum(mask1) > torch.sum(mask2) and torch.sum(mask1) > 0 and torch.sum(mask2) > 0):
                mask = mask2
            elif (torch.sum(mask1) == 0):
                mask = mask2
            else:
                mask = mask1

            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # m.weight.data.mul_(mask)
            # m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
            layer1 += 1
        elif flag == 1:
            flag = 0
            weight_copy = m.weight.data.abs().clone()
            mask = [1 for i in range(len(weight_copy))]
            mask = torch.FloatTensor(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))

    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total
print('pruned_ratio = ', pruned_ratio)
print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
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

print('长度 = ', len(cfg))
# acc = test(model)

print("Cfg:")
print(cfg)

newmodel = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)

print('    Total params: %.2fM' % (sum(p.numel() for p in newmodel.parameters())/1000000.0))

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "pruned_resnet164_20.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    # fp.write("Test accuracy: \n"+str(acc))

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
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

        # If the current convolution is not the last convolution in the residual block, then we can change the
        # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
        if conv_count % 3 != 1:
            w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
        # continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        # m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned_resnet56fusion.pth'))

# print(newmodel)
model = newmodel
test(model)

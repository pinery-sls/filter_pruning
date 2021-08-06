import torch
from resnet50 import resnet
from thop import profile


# model = resnet(layers=[3,4,6,3])


checkpoint = torch.load('pruned/pruned_sm50.pth',map_location='cpu')
model = resnet(layers=[3, 4, 6, 3], cfg=checkpoint['cfg'])
model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'])

# x = torch.randn(1,3,224,224)
#
# flops, params = profile(model, inputs=(x,))
# print(' Total flops = %.2fB' % (flops/1e9))
# print(' Total params = %.2fM' % (params/1e6))
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
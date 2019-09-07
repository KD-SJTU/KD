import torch
import numpy as np
from model.models import *
from function.dataset import SigmaDataSet
import matplotlib.pyplot as plt
import os
import sys

#### plot highest conv feature map ###
def plot_feature_new(feature,visual_path):
    feature = feature.data.cpu().numpy()

    feature_map_plot = np.zeros((14, 14))
    for channels in range(512):
        feature_map_plot += feature[channels, :, :]

    fig, ax = plt.subplots()
    # my_dpi = 96
    # plt.figure(figsize=(224 / my_dpi, 224 / my_dpi), dpi=my_dpi)

    im = ax.imshow(feature_map_plot, cmap=plt.get_cmap('hot'))
    fig.colorbar(im)
    plt.savefig(visual_path, dpi=70)
    plt.close()

model = student_network_bn(200, seed=0)
# model = vgg16_pretrained(200, seed=0)
root = '/home/data2/rzf/KD/trained_model/distill_vgg16_fc2_CUB/feature_maps'
if not os.path.exists(root):
    os.makedirs(root)
student_checkpoint = torch.load('/home/data2/rzf/KD/trained_model/distill_vgg16_fc2_CUB/149.pth.tar')
model.load_state_dict(student_checkpoint['state_dict'])
model.cuda(3)

train_dir = '/home/data2/rzf/KD/dataset/CUB/train'
train_loader = torch.utils.data.DataLoader(SigmaDataSet(train_dir),batch_size=64, shuffle=True,num_workers=8, pin_memory=True)

for i, (id, img_name, input, target) in enumerate(train_loader):
    print (i)
    input = input.cuda(3)
    features = model.features[:15](input)
    # features = model.net.features[:30](input)
    for feature_id in range(features.shape[0]):
        folder = os.path.join(root,img_name[feature_id].split('/')[0])
        if not os.path.exists(folder):
            os.mkdir(folder)
        plot_feature_new(features[feature_id],os.path.join(folder,img_name[feature_id].split('/')[1]))
sys.exit('Plot complete!')




################# cal data network std ######

distillation_pos = 'fc2'
model = student_network_bn(200, seed=0)
student_checkpoint = torch.load('/home/data2/rzf/KD/trained_model/student_908_CUB/149.pth.tar')
model.load_state_dict(student_checkpoint['state_dict'])
model.cuda(3)
train_dir = '/home/data2/rzf/KD/dataset/CUB/train'
train_loader = torch.utils.data.DataLoader(SigmaDataSet(train_dir), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

if distillation_pos != 'fc3':
    all_sample = torch.zeros((train_loader.dataset.size,4096))
else:
    all_sample = torch.zeros((train_loader.dataset.size, 200))
for i, (id, img_name, input, target) in enumerate(train_loader):

    print (i)
    input = input.cuda(3)

    if distillation_pos == 'fc1' :
        pre = model.features(input)
        pre = pre.view(pre.shape[0], -1)
        outputs = model.classifier[:1](pre)
        all_sample[id] = outputs.data.cpu()

    elif distillation_pos == 'fc2' :
        pre = model.features(input)
        pre = pre.view(pre.shape[0], -1)
        outputs = model.classifier[:4](pre)
        all_sample[id] = outputs.data.cpu()

    elif distillation_pos == 'fc3' :
        pre = model.features(input)
        pre = pre.view(pre.shape[0], -1)
        outputs = model.classifier(pre)
        all_sample[id] = outputs.data.cpu()

print(torch.std(all_sample))




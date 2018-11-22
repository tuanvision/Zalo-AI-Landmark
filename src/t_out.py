import os, sys
from zalo_utils import *
from sklearn.model_selection import train_test_split
import glob

import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image

from time import time, strftime

# device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

json_file = '../../data/train_val2018.json'
data_dir = '../../data/TrainVal/'

data_dir_public = '../../data/Private/'
# files = glob.glob(data_dir)
# for file in os.listdir(data_dir):
#     print(file)

test_fns = glob.glob(data_dir_public + '*.jpg')
test_lbs = [0] * len(test_fns)
print(len(test_fns), " ", len(test_lbs))
print(type(test_fns), " ", type(test_lbs))
# for name in glob.glob(data_dir + '*.jpg'):
#     print(name)

# for i in range(11655, len(test_fns)):
#     print("i = :", i)
#     image = Image.open(test_fns[i])

    # 1586567, 1743730, 1577654


print("Loading data")
fns, lbs, cnt = get_fns_lbs(data_dir, json_file)
print('Total files in the original dataset: {}'.format(cnt))
print('Total files with > 0 byes: {}'.format(len(fns)))
print('Total files with zero bytes {}'.format(cnt - len(fns)))

print("*" * 50)
print('Split data')
train_fns, val_fns, train_lbs, val_lbs = train_test_split(fns, lbs, test_size=0.2, random_state=68)
print('Number of training imgs: {}'.format(len(train_fns)))
print('Number of validation imgs: {}'.format(len(val_fns)))
# print(val_fns)
# print(train_lbs)
print(type(train_fns), " ", type(train_lbs))


print("DataLoader")

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
# input_size = 224
scale = 256
input_size = 224
batch_size = 256
epochs = 20
check_after = 5

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=90),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'test': transforms.Compose([
        transforms.Scale(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

dsets = dict()
dsets['train'] = MyDataset(train_fns, train_lbs, transform=data_transforms['train'])
dsets['test'] = MyDataset(test_fns, test_lbs, transform=data_transforms['test'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=256,
                                   shuffle=(x != 'test'),
                                   num_workers=1)
    for x in ['train', 'test']
}

print("Load model:")

savel_model_fn = 'inception_v3'
old_model = '../checkpoint/' + 'inception_v3_0817_0059.pkl'

model = MobileNetV2(103)

if os.path.isfile(old_model):
    print("Load old model")
    model.load_state_dict(torch.load(old_model))


model = model.to(device)



print("Number test set: ", len(test_fns))

# submit file
fw  = open('submit_private.csv', 'w')
fw.write('id,predicted' + '\n')


torch.set_grad_enabled(False)
model.eval()
answer = []
dem = 0
for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['test']):
    print("Batch_idx: ", batch_idx)
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    k = 3
    pred = outputs.data.cpu().numpy()
    topk = np.argsort(pred, axis = 1)[:, -k:][:, ::-1]
    for image_label in topk:
        answer.append(image_label[0])
        answer.append(image_label[1])
        answer.append(image_label[2])

answer = np.array(answer)
answer = answer.reshape(-1, 3)
print("Len answer: ", len(answer))
for i in range(len(answer)):
    value = answer[i]
    image_name = test_fns[i]
    image_name = image_name.split('/')
    image_name = image_name[4][:-4]
    # print(image_name)
    fw.write(image_name + ',' + str(value[0]) + ' ' + str(value[1]) +  ' ' + str(value[2]) + '\n')








      
    


    





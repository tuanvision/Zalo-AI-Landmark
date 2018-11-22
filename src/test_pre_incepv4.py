import os, sys
from zalo_utils import *
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from MobileNetV2 import MobileNetV2
from inceptionv4 import inceptionv4, InceptionV4
from time import time, strftime

# device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

json_file = '../../data/train_val2018.json'
data_dir = '../../data/TrainVal/'

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

print("DataLoader")

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
scale = 256
input_size = 224
batch_size = 64
epochs = 60
check_after = 10

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=90),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

dsets = dict()
dsets['train'] = MyDataset(train_fns, train_lbs, transform=data_transforms['train'])
dsets['val'] = MyDataset(val_fns, val_lbs, transform=data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=batch_size,
                                   shuffle=(x != 'val'),
                                   num_workers=1)
    for x in ['train', 'val']
}

print("Load model:")

savel_model_fn = 'inception_v4'
MobileNetV2_model = '../model/' + 'mobilenet_v2.pth.tar'
old_model = '../checkpoint/' + 'inception_v4.pkl' 

# model = MyMobileNetV2(MobileNetV2_model, len(set(train_lbs)))

# model = MyInception_v3(103)
model = InceptionV4(num_classes=103)

# model.load_state_dict(torch.load(MobileNetV2_model))
# model = models.vgg16(pretrained=True) 
# model = MyNet(len(set(train_lbs)))
# model = model.to(device)
# model.frozen_until(15)
# chay di# train cai kia con lau
# exit(0) chay dc chua # ok chay dc r
# model = model.features
# model = model.to(device)
# print(model)
# o terminal khac# doi chay not cai valid roi cancel
# for parameter in model.parameters():
#     print(parameter.name(), " ", parameter.size()) 
# print(model)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print (name, " ", param.size())
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", pytorch_total_params)

if os.path.isfile(old_model):
    print("Load old model")
    model.load_state_dict(torch.load(old_model))

model = model.to(device)
print("Tuan")
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


print("Number lables: ", len(set(train_lbs)))


N_train = len(train_lbs)
N_valid = len(val_lbs)


best_top3 = 1.0
t0 = time()
for epoch in range(epochs):
    print('#################################################################')
    print('=> Training Epoch #%d ' % (epoch + 1))
    
    run_top1_correct = 0.0
    run_top3_correct = 0.0
    total_size = 0
    run_loss = 0.0

    exp_lr_scheduler.step()
    model.train() # why we need this here
    torch.set_grad_enabled(True)

    # Training
    for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['train']):
        inputs = inputs.to(device)
        # print(inputs.size())
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # end of this training

        # Get the top 1, top 3 output
        # top1
        _, preds = torch.max(outputs.data, 1)
        top3_correct, _ =  mytopk(outputs.data.cpu().numpy(), labels, k=3)
        run_top3_correct += top3_correct
        top1_correct = preds.eq(labels.data).cpu().sum()

        run_top1_correct += top1_correct
        run_loss += loss.item()
        total_size += labels.size(0)

        top1error = 1 - float(run_top1_correct)/total_size
        top3error = 1 - float(run_top3_correct)/total_size
        
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f'
                    % (epoch + 1, epochs, batch_idx + 1,
                    (len(train_fns) // batch_size), loss.item()/ batch_size,
                    top1error, top3error))
        sys.stdout.flush()
        sys.stdout.write('\r')


    top1_error = 1 - float(run_top1_correct) / N_train
    top3_error = 1 - float(run_top3_correct) / N_train
    
    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
        % (run_loss, top1_error, top3_error))

    print_eta(t0, epoch, epochs)

    # Validation
    print("Validation: ")
    if (epoch + 1) % check_after == 0:
       
        run_top1_correct = 0.0
        run_top3_correct = 0.0
        total_size = 0
        run_loss = 0.0
        torch.set_grad_enabled(False)
        model.eval()

        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            top3_correct, _ =  mytopk(outputs.data.cpu().numpy(), labels, k=3)
            run_top3_correct += top3_correct

            top1_correct = preds.eq(labels.data).cpu().sum()

            run_top1_correct += top1_correct
            run_loss += loss.item()
            total_size += labels.size(0)

        top1_error = 1 - float(run_top1_correct) / N_valid
        top3_error = 1 - float(run_top3_correct) / N_valid
        
        print('| Validation loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
            % (run_loss, top1_error, top3_error))
        
        save_point = '.'
        if top3_error < best_top3:
            best_top3 = top3_error
            print("Saving model:")
            save_model_fn_new = savel_model_fn + '_' + strftime('%m%d_%H%M') 
            save_point = '../checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(model.state_dict(), save_point + save_model_fn_new + '.pkl') 
            print("=" * 50)
            print("model saved to %s" % (save_point + save_model_fn_new + '.pkl'))



      
    


    





from __future__ import print_function 
import os
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
import pickle
import pdb
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import torch.backends.cudnn as cudnn
from time import time
from MobileNetV2 import MobileNetV2

def get_fns_lbs(base_dir, json_file, pickle_fn = 'mydata.p', force = False):    
    pickle_fn = base_dir + pickle_fn 
    # pdb.set_trace() 
    if os.path.isfile(pickle_fn) and not force:
        mydata = pickle.load(open(pickle_fn, 'rb'))
        fns = mydata['fns']
        lbs = mydata['lbs']
        cnt = mydata['cnt']
        return fns, lbs, cnt

    f = open(json_file, 'r')
    line = f.readlines()[0] # only one line 
    end = 0 
    id_marker = '\\"id\\": '
    cate_marker = '\\"category\\": '
    cnt = 0 
    fns = [] # list of all image filenames
    lbs = [] # list of all labels
    while True:
        start0 = line.find(id_marker, end)
        if start0 == -1: break 
        start_id = start0 + len(id_marker)
        end_id = line.find(',', start_id) 

        start0 = line.find(cate_marker, end_id)
        start_cate = start0 + len(cate_marker)
        end_cate = line.find('}', start_cate)

        end = end_cate
        cnt += 1
        cl = line[start_cate:end_cate]
        fn = base_dir + cl + '/' + line[start_id:end_id] + '.jpg'
        if os.path.getsize(fn) == 0: # zero-byte files 
            continue 
        lbs.append(int(cl))
        fns.append(fn)

    # pdb.set_trace()
    mydata = {'fns':fns, 'lbs':lbs, 'cnt':cnt}
    pickle.dump(mydata, open(pickle_fn, 'wb'))
    print(os.path.isfile(pickle_fn))

    return fns, lbs, cnt 

class MyDataset(Dataset):

    def __init__(self, filenames, labels, transform=None):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels 
        self.transform = transform

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        # TODO: replace Image by opencv
        image = Image.open(self.fns[idx])
        tmp = image.getpixel((0, 0))
        if isinstance(tmp, int) or len(tmp) != 3: # not rgb image
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.lbs[idx], self.fns[idx]


class MyResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        self.num_classes = num_classes
        # model = models.vgg19(pretrained)
        # self.num_ftrs = model.classifier[6].in_features

        self.shared = nn.Sequential(*list(model.children())[:-1]) # dong nay la bo cai child cuoi cung, tuc la remove cai layer 1000 classes di
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes)) # cai nay la them 1 lop fc 

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer): #freeze net
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1
        print("Child counter: ", child_counter)

class MyMobileNetV2(nn.Module):
    def __init__(self, model_dir, num_classes):
        super(MyMobileNetV2, self).__init__()
        
        model = MobileNetV2(n_class=1000)
        model.load_state_dict(torch.load(model_dir))

        self.num_ftrs = 1000 # (MobileNetV2)
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        return self.target(x)
    
    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class MyInception_v3(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(MyInception_v3, self).__init__()
        self.model_conv = models.inception_v3(pretrained='imagenet')
        # for i, param in self.model_conv.named_parameters():
        #     param.requires_grad = False # cai nay free all the net day. nhung minh thay cai fully cua minh vao r ma, train cai fully thoi
        self.num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(self.num_ftrs, 103)
        # self.num_classes = num_classes
        #ct = []
        #for name, child in self.model_conv.named_children():
        #    if "Conv2d_4a_3x3" in ct:
        #        for params in child.parameters():
        #            params.requires_grad = True
        #    ct.append(name)

        #self.shared = nn.Sequential(*list(model.children())[:-1])
        #self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        #x = self.shared(x) # cu cai gi co 1 thi no remove di yep
        #x = torch.squeeze(x) # cai squeeze nay de lam j# giam chieu du lieu: vi du A * 1* B thi thanh A* B 
        return self.model_conv(x)

    def frozen_until(self, to_layer):
        # print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # # if to_layer = -1, frozen all
        # child_counter = 0
        # for child in self.shared.children():
        #     if child_counter <= to_layer:
        #         print("child ", child_counter, " was frozen")
        #         for param in child.parameters():
        #             param.requires_grad = False
        #         # frozen deeper children? check
        #         # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
        #     else:
        #         print("child ", child_counter, " was not frozen")
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     child_counter += 1
        
        # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
        for name, child in model_conv.named_children():
            if "Conv2d_4a_3x3" in ct:
                for params in child.parameters():
                    params.requires_grad = True


class MyNet(nn.Module): 
    def __init__(self, num_classes, pretrained = True):
        super(MyNet, self).__init__()
        # model = models.inception_v3(pretrained)
        # self.num_ftrs = 129
        model = models.vgg19(pretrained)
        # self.num_ftrs = model.fc.in_features
        # self.num_classes = num_classes
        self.num_ftrs = model.classifier[6].in_features
        # model.classifier._modules['6'] = nn.Linear(4096, num_classes)
        # go dc ma rrr
        self.shared = nn.Sequential(*list(model.children())[:-1])

        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
      
        return self.target(x)
    
    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0 # the chac no co 1 layer that. huhu
        for child in self.shared.children(): # thang nay co 1 layer thoi a. Sao counter = 1, toi cung dang thac mac day. de toi sua thanh Mỷyresnet
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1
        print("Child counter: ", child_counter)


def mytopk(pred, gt, k=3):
    """
    compute topk error
    pred: (n_sample,n_class) np array
    gt: a list of ground truth
    --------
    return:
        n_correct: number of correct prediction 
        topk_error: error, = n_connect/len(gt)
    """
    # topk = np.argpartition(pred, -k)[:, -k:]
    topk = np.argsort(pred, axis = 1)[:, -k:][:, ::-1]
    diff = topk - np.array(gt).reshape((-1, 1))
    n_correct = np.where(diff == 0)[0].size 
    topk_error = float(n_correct)/pred.shape[0]
    return n_correct, topk_error


def net_frozen(args, model):
    print('********************************************************')
    # model.frozen_until(args.frozen_until) # chi can dong nay là free den cai child thu until
    init_lr = args.lr
    if args.trainer.lower() == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr, weight_decay=args.weight_decay)
    elif args.trainer.lower() == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr,  weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer


def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def second2str(second):
    h = int(second/3600.)
    second -= h*3600.
    m = int(second/60.)
    s = int(second - m*60)
    return "{:d}:{:02d}:{:02d} (s)".format(h, m, s)


def print_eta(t0, cur_iter, total_iter):
    """
    print estimated remaining time
    t0: beginning time
    cur_iter: current iteration
    total_iter: total iterations
    """
    time_so_far = time() - t0
    iter_done = cur_iter + 1
    iter_left = total_iter - cur_iter - 1
    second_left = time_so_far/float(iter_done) * iter_left
    s0 = 'Epoch: '+ str(cur_iter + 1) + '/' + str(total_iter) + ', time so far: ' \
        + second2str(time_so_far) + ', estimated time left: ' + second2str(second_left)
    print(s0)

def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)

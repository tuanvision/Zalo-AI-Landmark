import os
import torch
import pickle
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_fns_lbs(base_dir, json_file, pickle_fn = 'mydata.p', force = False):    
    pickle_fn = base_dir + pickle_fn 
    print("Pickle_fn: ", pickle_fn)

    # pdb.set_trace() 
    # if os.path.isfile(pickle_fn) and not force:
    #     mydata = pickle.load(open(pickle_fn, 'rb'))
    #     fns = mydata['fns']
    #     lbs = mydata['lbs']
    #     cnt = mydata['cnt']
    #     return fns, lbs, cnt

    f = open(json_file, 'r')
    line = f.readlines()[0] # only one line 
    # print("Len line: ", len( f.readlines() ) )
    # print("Type line: ", type(line))

    end = 0 
    id_marker = '\\"id\\": '
    cate_marker = '\\"category\\": '
    cnt = 0 
    fns = [] # list of all image filenames
    lbs = [] # list of all labels
    # print("Line: ", line)
    # print(id_marker, " ", cate_marker)

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
        # print(int(cl), " ", fn)

        lbs.append(int(cl))
        fns.append(fn)
        

    # pdb.set_trace()
    # print("len fns: ", len(fns) )
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



json_file = '../../data/train_val2018.json'
data_dir = '../../data/TrainVal/'

# print("json_file: {}, data_dir: {}".format(json_file, data_dir))
# print('Loading data')
fns, lbs, cnt = get_fns_lbs(data_dir, json_file)
print('Total files in the original dataset: {}'.format(cnt))
print('Total files with > 0 byes: {}'.format(len(fns)))
print('Total files with zero bytes {}'.format(cnt - len(fns)))

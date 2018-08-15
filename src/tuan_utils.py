from torch.utils.data import Dataset


class LandmarkDataset(Dataset):

    def __init__(self, root_dir, size=(480, 480)):
        self.files = glob(root_dir)
        self.size = size
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # img = np.array(Image.open(self.files(idx)).resize(self.size))
        # label = self.files[idx].split('/')[-2]
        # return img, label
        pass

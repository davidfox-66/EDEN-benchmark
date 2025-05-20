import os
import torch, torchvision
from torch.utils.data import Dataset
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore")


class SNUFILMDataset(Dataset):
    def __init__(self, data_dir="datasets/SNU_FILM", mode="extreme"):
        self.data_root = data_dir
        self.mode = mode
        self.image_root = os.path.join(self.data_root, "test")
        test_fn = os.path.join(self.data_root, f"test-{mode}.txt")
        with open(test_fn, "r") as f:
            testlist = f.read().splitlines()
        self.meta_data = testlist[:-1]

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        imgs_path = self.meta_data[index].split(" ")
        # img_paths = [os.path.join(self.image_root, img_path) for img_path in imgs_path]
        img_paths = [os.path.join(self.image_root, *img_path.split('/')[3:]) for img_path in imgs_path]
        frame0 = torchvision.io.read_image(img_paths[0])
        frame1 = torchvision.io.read_image(img_paths[2])
        gt = torchvision.io.read_image(img_paths[1])
        frames = torch.stack((frame0, frame1, gt), dim=0)
        return frames

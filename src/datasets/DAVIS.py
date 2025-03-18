import os
import torch, torchvision
from torch.utils.data import Dataset
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import warnings
warnings.filterwarnings("ignore")


class DAVISDataset(Dataset):
    def __init__(self, data_dir="datasets/DAVIS", height=480, width=854, is_val=False):
        self.h = height
        self.w = width
        self.is_val = is_val
        self.data_root = data_dir
        self.meta_data = self.read_data()

    def __len__(self):
        return len(self.meta_data)

    def read_data(self):
        videos = [os.path.join(self.data_root, video_name) for video_name in os.listdir(self.data_root)]
        vid_list = []
        for vid in videos:
            for i in os.listdir(vid):
                vid_list.append(os.path.join(vid, i))
        return vid_list

    def __getitem__(self, index):
        vid_path = self.meta_data[index]
        frame0 = torchvision.io.read_image(f"{vid_path}/frame_0.jpg")
        frame1 = torchvision.io.read_image(f"{vid_path}/frame_2.jpg")
        gt = torchvision.io.read_image(f"{vid_path}/frame_1.jpg")
        frames = torch.stack((frame0, frame1, gt), dim=0)
        if self.is_val:
            frames = v2.Resize(size=(self.h, self.w), antialias=True)(frames)
        return frames

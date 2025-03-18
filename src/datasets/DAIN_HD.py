import os
import torch, torchvision
from torch.utils.data import Dataset
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore")


class HDDataset(Dataset):
    def __init__(self, data_dir="datasets/DAIN_HD", res=544):
        self.data_root = data_dir
        self.res = res
        self.meta_data = self.read_data()

    def __len__(self):
        return len(self.meta_data)

    def read_data(self):
        dataset_path = f"{self.data_root}/{self.res}p"
        videos = [os.path.join(dataset_path, video_name) for video_name in os.listdir(dataset_path)]
        vid_list = []
        for vid in videos:
            if "images" in vid:
                frames_num = len(os.listdir(vid))
                for i in range(1, frames_num - 1, 2):
                    vid_list.append([vid, (i - 1, i, i + 1)])
        return vid_list

    def __getitem__(self, index):
        vid_path = self.meta_data[index]
        frame0 = torchvision.io.read_image(f"{vid_path[0]}/{vid_path[1][0]:03d}.png")
        frame1 = torchvision.io.read_image(f"{vid_path[0]}/{vid_path[1][2]:03d}.png")
        gt = torchvision.io.read_image(f"{vid_path[0]}/{vid_path[1][1]:03d}.png")
        frames = torch.stack((frame0, frame1, gt), dim=0)
        return frames

import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def get_album(album_path, album_clip_length, img_size):
    files = os.listdir(album_path)
    n_files = len(files)
    idx_fetch = np.linspace(0, n_files-1, album_clip_length, dtype=int)
    tensor_batch = torch.zeros(len(idx_fetch), img_size, img_size, 3)
    for i, id in enumerate(idx_fetch):
        im = Image.open(os.path.join(album_path, files[id]))
        im_resize = im.resize((img_size, img_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2)   # HWC to CHW
    return tensor_batch

class CUFED(Dataset):
    event_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                    'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation',
                    'GroupActivity', 'Halloween', 'Museum', 'NatureTrip',
                    'PersonalArtActivity', 'PersonalMusicActivity', 'PersonalSports',
                    'Protest', 'ReligiousActivity', 'Show', 'Sports', 'ThemePark',
                    'UrbanTrip', 'Wedding', 'Zoo']

    def __init__(self, root_dir, split_dir, is_train, img_size, album_clip_length):
        self.img_size = img_size
        self.album_clip_length = album_clip_length
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if self.phase == 'train':
            split_path = os.path.join(split_dir, 'train_split.txt')
        else:
            split_path = os.path.join(split_dir, 'val_split.txt')

        label_path = os.path.join(root_dir, "event_type.json")
        with open(label_path, 'r') as f:
            album_data = json.load(f)

        with open(split_path, 'r') as f:
            album_names = f.readlines()
        vidname_list = [name.strip() for name in album_names]

        labels_np = np.zeros((len(vidname_list), len(self.event_labels)), dtype=np.float32)

        for i, vidname in enumerate(vidname_list):
            for lbl in album_data[vidname]:
                idx = self.event_labels.index(lbl)
                labels_np[i, idx] = 1

        self.labels = labels_np
        self.videos = vidname_list
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        dataset_path = os.path.join(self.root_dir, 'images')
        album_path = os.path.join(dataset_path, self.videos[idx])
        album_tensor = get_album(album_path, self.album_clip_length, self.img_size)
        return album_tensor, self.labels[idx]

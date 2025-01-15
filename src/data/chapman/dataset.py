import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


class ChapmanDataset(Dataset):
    def __init__(self, path, phase, term, grouptype='MSML', class_list=[0, 1, 2, 3], transform=["normalization"],
                 percentage=1.0):
        self.transform = transform
        self.class_list = class_list
        self.pkl_path = path
        self.grouptype = grouptype

        # Load pids
        try:
            with open(self.pkl_path + "pid_phases_chapman.pkl", 'rb') as file:
                self.pids = pickle.load(file)['ecg'][1][phase][term]
        except:
            print("Failed to load pids")
        # Load labels
        try:
            with open(self.pkl_path + "labels_phases_chapman.pkl", 'rb') as file:
                self.labels = pickle.load(file)['ecg'][1][phase][term]
        except:
            print("Failed to load labels")
        # Load data
        try:
            with open(self.pkl_path + "frames_phases_chapman.pkl", 'rb') as file:
                self.data = np.moveaxis(pickle.load(file)['ecg'][1][phase][term], 2, 1)
        except:
            print("Failed to load data")

        # Initially data shape is (pid, channel, 5000) (MSML)
        sample_num = self.data.shape[0]  # namely pid
        channel_num = self.data.shape[1]
        time_len = self.data.shape[2] // 2  # 2500
        if grouptype == 'MSML':
            # Split frame on 2 parts and save channels
            # (pid, channel*2, time)
            self.data = self.data.reshape(sample_num, channel_num * 2, time_len)
        elif grouptype == 'MS':
            # Multi segment, along time
            # Move channels as different datapoints
            #  (pid*channel, 2, time)
            self.data = self.data.reshape(sample_num * channel_num, 2, time_len)
            # Align pids and labels
            self.pids = self.pids.repeat(channel_num)
            self.labels = self.labels.repeat(channel_num)
        elif grouptype == 'ML':
            # Multi lead, along channels
            # (pid, channel, 2, time)
            self.data = self.data.reshape(sample_num, channel_num, 2, time_len)
            # (pid, 2, channel, time)
            self.data = np.moveaxis(self.data, 2, 1)
            # (pid*2, channel, time)
            self.data = self.data.reshape(sample_num * 2, channel_num, time_len)
            # Align pids and labels
            self.pids = self.pids.repeat(2)
            self.labels = self.labels.repeat(2)

        # Take part of the data if needed
        if (type(percentage) == float) and (0.0 < percentage < 1.0):
            num_items = len(self.labels) * percentage
            self.data = self.data[:num_items]
            self.labels = self.labels[:num_items]
            self.pids = self.pids[:num_items]

    def __len__(self):
        """
        :return: Number of datapoints in the dataset
        """
        return len(self.labels)

    def normalize(self, frame):
        """
        Normalize along axis 1 from (0, 1)
        :param frame: datapoint of shape (channels, time_len)
        :return: normalized datapoint
        """
        if isinstance(frame, torch.Tensor) and len(frame.shape) == 2:
            frame = frame.T
            frame = (frame - torch.min(frame, dim=0)[0]) / (
                    torch.max(frame, dim=0)[0] - torch.min(frame, dim=0)[0] + 1e-8)
            frame = frame.T
        else:
            print("Problems with data normalization")
        return frame

    def __getitem__(self, index):
        # Take a datapoint
        data = torch.from_numpy(self.data[index]).float()

        if self.transform:
            data = self.normalize(data)

        sample = {"data": data, "label": self.labels[index], "patient": self.pids[index], "grouptype": self.grouptype}
        return sample

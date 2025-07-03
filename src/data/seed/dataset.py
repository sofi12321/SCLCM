from torch.utils.data import Dataset
import numpy as np
import scipy
import torch
import os

class SeedDataset(Dataset):
    def __init__(self, path, class_list=[-1, 0, 1], segment_length=400, pids=None, grouptype="CMLC",
                 transform=["normalization"], channels=None, percentage=1.0):
        self.transform = transform
        self.class_list = class_list
        self.channels = [cl.upper() for cl in channels] if channels else None
        self.general_path = path
        self.grouptype = grouptype

        # Prepare to load data
        # Sort files to be loaded by pid and date in name
        all_files = [f for f in os.listdir(self.general_path) if (f[-3:] == "mat") and ("_" in f)]
        all_files = sorted(all_files,
                           key=lambda x: (int(x.split("_")[0]),
                                          int(x.split(".")[0].split("_")[1])
                                          ))
        # Select some patients if needed
        if pids:
            all_files = [f for f in all_files if int(f.split("_")[0]) in pids]
        pids = np.array([f.split("_")[0] for f in all_files])
        # Select channels if needed
        all_files = [os.path.join(self.general_path, f) for f in all_files]
        channels_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                          'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
                          'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
                          'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
                          'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
        sorter = np.argsort(channels_order)
        if channels:
            selected_chans = sorter[np.searchsorted(channels_order, channels, sorter=sorter)]
        else:
            selected_chans = list(range(len(channels_order)))

        # Define time length based on segment length and num
        # Number of segments have to be even
        time_length = (37001 // segment_length) * segment_length
        self.segment_num = time_length // segment_length
        if self.segment_num % 2 == 1:
            self.segment_num -= 1
        time_length = self.segment_num * segment_length

        # Save info
        labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

        self.num_datapoints = len(all_files) * len(labels)
        self.num_channels = len(selected_chans)
        self.segment_length = segment_length

        # Load data from files
        all_data = np.array([])
        for f in all_files:
            d = scipy.io.loadmat(os.path.join(self.general_path, f))

            for r in range(15):
                # Load data
                eeg_name = [k for k in d.keys() if k[-2:] == "13"][0][:-2] + str(r + 1)
                part_data = d[eeg_name][selected_chans]
                leni = part_data.shape[1]

                # Take only central segment, if video length is more than needed
                if all_data.shape[0] == 0:
                    # Store the first item
                    all_data = np.array(
                        [part_data[:, (leni - time_length) // 2:(leni - time_length) // 2 + time_length]])
                else:
                    # Add to already loaded data
                    all_data = np.concatenate(
                        [all_data, [part_data[:, (leni - time_length) // 2:(leni - time_length) // 2 + time_length]]])

        # Reformat to a shape of (pid * label, ch, segm_n, time )
        self.data = all_data.reshape(self.num_datapoints, self.num_channels, self.segment_num, self.segment_length)

        if grouptype == "MSML":
            # Multi-segment, multi-lead
            self.segment_num = self.segment_num // 2
            # self.num_datapoints, self.num_channels, self.segment_num, 1, self.segment_length
            self.data = self.data[:, :, :, None, :]
            self.data = np.concatenate([
                self.data[:, :, ::2, :, :],
                self.data[:, :, 1::2, :, :]
            ], axis=3)
            # self.num_datapoints,  self.segment_num, 2, self.num_channels, self.segment_length
            self.data = np.moveaxis(np.moveaxis(self.data, 2, 1), 3, 2)

            # self.num_datapoints * self.segment_num, 2 * self.num_channels,self.segment_length
            self.data = self.data.reshape(self.num_datapoints * self.segment_num,
                                          2 * self.num_channels,
                                          self.segment_length)

            self.num_datapoints = self.num_datapoints * self.segment_num
            self.num_channels *= 2

        elif grouptype == "MS":
            # multi-segment
            self.segment_num = self.segment_num // 2
            # self.num_datapoints, self.num_channels, self.segment_num, 1, self.segment_length
            self.data = self.data[:, :, :, None, :]
            self.data = np.concatenate([
                self.data[:, :, ::2, :, :],
                self.data[:, :, 1::2, :, :]
            ], axis=3)
            # self.num_datapoints, self.segment_num, self.num_channels, 2, self.segment_length
            self.data = np.moveaxis(self.data, 2, 1)

            # self.num_datapoints * self.segment_num * self.num_channels, 2, self.segment_length
            self.data = self.data.reshape(self.num_datapoints * self.segment_num * self.num_channels,
                                          2,
                                          self.segment_length)

            self.num_datapoints = self.num_datapoints * self.segment_num * self.num_channels
            self.num_channels = 2
        else:
            # ML, multi-lead
            # self.num_datapoints, self.segment_num, self.num_channels, self.segment_length
            self.data = np.moveaxis(self.data, 2, 1)
            self.data = self.data.reshape(self.num_datapoints * self.segment_num,
                                          self.num_channels,
                                          self.segment_length)
            self.num_datapoints *= self.segment_num

        # Make pids
        self.pids = pids.repeat(len(labels) * self.segment_num)
        # Make labels
        self.labels = np.array(labels * len(pids)).repeat(self.segment_num)

        self.pids = np.array(self.pids)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.pids)

    def normalize(self, frame):
        if isinstance(frame, torch.Tensor):
            frame = frame.T
            frame = (frame - torch.min(frame, dim=0)[0]) / (
                    torch.max(frame, dim=0)[0] - torch.min(frame, dim=0)[0] + 1e-8)
            frame = frame.T
        else:
            print("heeeelp")
        return frame

    def __getitem__(self, index):
        # loa from the path

        data = torch.from_numpy(self.data[index]).float()
        # print(data.shape)

        if self.transform:
            data = self.normalize(data)

        sample = {"data": data,
                  "label": self.labels[index] + 1,
                  "patient": self.pids[index],
                  "datatype": self.grouptype}
        return sample

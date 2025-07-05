import pickle
import random 
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class PreloadedDataset(Dataset):
    def __init__(self, data, labels, pids, add_data=None, augmentations=None, need_norm=False):
        # Shape of the data is (freq, channels, time)
        self.data = data
        self.add_data = add_data
        self.labels = labels.astype(int)
        self.pids = pids
        self.grouptype = "ML"
        self.augmentations = augmentations
        self.min_label = min(self.labels)
        self.need_norm = need_norm

        _, c = np.unique(pids, return_counts =True)

        if self.augmentations and ("mix_dtpts" in self.augmentations) and (min(c) < 2):
            print("Cannot choose 2 not augmented datapoints of one positive pair, so drop mixup augmentation")
            del self.augmentations[self.augmentations.index("mix_dtpts")]

        if self.augmentations and ("mask_channel" in self.augmentations) and (self.data.shape[2] < 6):
            print("Too low number of channels, so drop mask_channel augmentation")
            del self.augmentations[self.augmentations.index("mask_channel")]
        if self.augmentations and len(self.augmentations ) == []:
            self.augmentations = None
        # check shuffle channels and freqs > 1


    def get_unit_len(self):
        return len(self.labels)

    def __len__(self):
        if self.augmentations is not None:
            return self.get_unit_len() * (len(self.augmentations)+1)
        else:
            return self.get_unit_len()

    def normalize_last_axis(self, data):
        sh = data.shape[-1]
        if len(data.shape) == 4:
            return np.nan_to_num( (data - np.mean(data, axis=-1)[:, :, :, None].repeat(sh, axis=-1)) / np.std(data, axis=-1)[:, :, :, None].repeat(sh, axis=-1) )
        elif len(data.shape) == 3:
            return np.nan_to_num( (data - np.mean(data, axis=-1)[:, :, None].repeat(sh, axis=-1)) / np.std(data, axis=-1)[:, :, None].repeat(sh, axis=-1) )
        else:
            print("Data was not normalized")

    def get_random_index_across_pid(self, index):
        # Finds index of some dp which is from  the same video
        # Take ind of the video
        old = np.where(self.pids == self.pids[index])[0]
        old = np.delete(old, np.where(old == index))
        rand_int = random.choice(old)
        return rand_int

    def mix_dpts(self, original_dp, other_dp):
        alpha = np.random.rand(1)[0]/2
        return (1-alpha) * original_dp + alpha * other_dp

    def mask_channel(self, dp, prop=0.2):
        num_ch = np.random.randint(1, max(2, int(dp.shape[1]*prop)))
        ch_ind = np.random.randint(0, dp.shape[1]-1, size=num_ch)
        # print(ch_ind)
        dd = dp.copy()
        dd[:, ch_ind] = 0.0
        return dd

    def get_pids(self):
        if self.augmentations:
            return np.concatenate([self.pids for _ in range(len(self.augmentations) + 1)])
        return self.pids

    def frequency_noise(self, dp):
        return dp + np.random.normal(loc=0.0, scale=0.2, size=dp.shape)

    def magnitude_warping(self, dp):
        sh = [1 for _ in range(len(dp.shape) -1)] + [dp.shape[-1]]
        return dp * np.random.normal(loc=1.0, scale=0.2, size=sh)

    def flip(self, dp):
        return -dp

    def shuffle_along_axis(self, dp, axiss=0):
        assert type(axiss) == int, "Axis must be a number"
        assert dp.shape[axiss] >= 2, "Shuffling along axis require shape of at least 2 for axis "+str(axiss)
        sh = dp.shape[axiss] // 2
        if axiss == 0:
            return np.concatenate([dp[sh:], dp[:sh]], axis=axiss)
        elif axiss == 1:
            return np.concatenate([dp[:, sh:], dp[:, :sh]], axis=axiss)
        print("Other axis are not implemented")
        return dp

    def __getitem__(self, index):
        need_augment = 0
        if index >= len(self.labels):
            need_augment = index // len(self.labels)
            index = index % len(self.labels)

        data = self.data[index].copy()

        if (self.augmentations is not None) and (need_augment > 0):
            if  self.augmentations[need_augment-1] == "mix_dtpts":
                new_ind = self.get_random_index_across_pid(index)
                data = self.mix_dpts(data, self.data[new_ind])

            elif  self.augmentations[need_augment-1] == "mask_channel":
                data = self.mask_channel(data)
            elif  self.augmentations[need_augment-1] == "frequency_noise":
                data = self.frequency_noise(data)
            elif  self.augmentations[need_augment-1] == "magnitude_warping":
                data = self.magnitude_warping(data)
            elif self.augmentations[need_augment-1] == "flip":
                data = self.flip(data)
            elif self.augmentations[need_augment-1] == "shuffle_channels":
                data = self.shuffle_along_axis(data, axiss=1)
            elif self.augmentations[need_augment-1] == "shuffle_frequences":
                data = self.shuffle_along_axis(data, axiss=0)
            else:
                print("Unexpected augmentation. Send unchanged datapoint")

        # Manage additional data
        add_data = ""
        if self.add_data:
            add_data = self.add_data[index].copy()
        # Swap channel and frequency to have
        # dim = (fr, ch, time)
        if self.need_norm:
            data = self.normalize_last_axis(data)
            if add_data != "":
                add_data = self.normalize_last_axis(add_data)

        data = torch.from_numpy(data).float()
        if add_data != "":
            add_data = torch.from_numpy(add_data).float()
        # channel, time
        return {"data": data,
                  "add_data": add_data,
                  "label": self.labels[index] - self.min_label,
                  "patient": self.pids[index],
                  "datatype": self.grouptype,
                  "index": index + need_augment * len(self.labels) }

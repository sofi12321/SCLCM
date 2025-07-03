from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from data.general.dataset import PreloadedDataset
from data.general.batching import SubsetRandomSampler

def split_data(data, session_labels, pid_video,
               num_video, subj_list, video_list,
               need_norm = False, add_data = None, batch_size = 128,
               augmentations = ["mix_dtpts", "mask_channel", "frequency_noise"],
               split_by="subject", test_fraction=0.3):
    # split_by = "subject" / "video" / "record" / "random" / "first_n"

    if split_by == "subject":
        subj_list = np.array(subj_list).reshape(-1, 1)
        mask = np.any((pid_video // num_video) == subj_list, axis=0)
        print(sum(mask))
    elif split_by == "video":
        video_list = np.array(video_list).reshape(-1, 1)
        mask = np.any((pid_video % num_video) == video_list, axis=0)
    elif split_by == "record":
        inds = np.random.choice(np.unique(pid_video), size=int(len(np.unique(pid_video)) * test_fraction)).reshape(-1, 1)
        mask = np.any(pid_video == inds, axis=0)
    elif split_by == "first_n":
        mask = np.concatenate([np.ones(int(test_fraction * data.shape[0])), np.zeros(data.shape[0] - int(test_fraction * data.shape[0]))])
    else:
        mask_tr, mask_test = train_test_split(
            range(len(data)),
            test_size=test_fraction, random_state=21, shuffle=True, stratify=session_labels)
        mask = np.sum(np.array(range(len(data))).reshape(-1, 1) == mask_test, axis=1) > 0


    test_data, test_labels, test_pids = data[mask], session_labels[mask], pid_video[mask]
    train_data, train_labels, train_pids = data[~mask], session_labels[~mask], pid_video[~mask]

    train_add_data = None
    test_add_data = None
    if add_data is not None:
        test_add_data = add_data[mask]
        train_add_data = add_data[~mask]

    print("Labels")
    print("Train", np.unique(train_labels, return_counts=True))
    print("Test", np.unique(test_labels, return_counts=True))

    train_dataset = PreloadedDataset(train_data, train_labels, train_pids,
                                    add_data=train_add_data,
                                    augmentations=augmentations, need_norm=need_norm)
    train_loader = DataLoader(train_dataset,
            batch_sampler=SubsetRandomSampler(train_dataset.get_pids(),
                                            batch_size, need_update=True, drop_last_batch=False, num_groups_in_batch=None, generator=None)
                                            )
    test_dataset = PreloadedDataset(test_data, test_labels, test_pids,
                                        add_data=test_add_data,
                                        augmentations=augmentations, need_norm=need_norm)
    test_loader = DataLoader(test_dataset,
            batch_sampler=SubsetRandomSampler(test_dataset.get_pids(),
            batch_size, need_update=False, drop_last_batch=False, num_groups_in_batch=None, generator=None)
                                            )


    test_dataset0 = PreloadedDataset(test_data, test_labels, test_pids,
                                    add_data=test_add_data,
                                    augmentations=None, need_norm=need_norm)
    test_loader0 = DataLoader(test_dataset0,
            batch_size=batch_size,
            shuffle=False
                                            )

    train_dataset0 = PreloadedDataset(train_data, train_labels, train_pids,
                                    add_data=train_add_data,
                                    augmentations=None, need_norm=need_norm)
    train_loader0 = DataLoader(train_dataset0,
            batch_size=batch_size,
            shuffle=True
                                                )
    return train_loader, test_loader, train_loader0, test_loader0

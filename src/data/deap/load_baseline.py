import numpy as np

def load_deap_baseline(data_path='EEG_Dataset/DEAP/deap_baseline1s_data.dat',
                 labels_path='EEG_Dataset/DEAP/deap_baseline1s_labels.dat',
                 label_by="valence", lower=3.5, higher = 6.5
                       ):
    data = np.fromfile(data_path).reshape(76800, 32, 128)
    labels = np.fromfile(labels_path).reshape(76800, 4)
    pid_video = np.array([v+p*40 for p in range(32) for t in range(60) for v in range(40)])

    data = data[:, None, :, :]
    # valence, arousal, dominance, liking
    if label_by == "valence":
        labels_no_split = labels.reshape(32, 60, 40, 4)[:,0, :,:].reshape(32, 40, 4)
        mask_labels = parse_valence(labels_no_split)
        mask_labels = mask_labels[:, None, :].repeat(60, axis=1).reshape(-1) == 1

        pid_video = pid_video[mask_labels]
        data = data[mask_labels]
        session_labels = (labels[:, 0].reshape(-1) > 5).astype(int)
        session_labels = session_labels[mask_labels]
    else:
        session_labels = (labels[:, 1] > 5).astype(int)

    # Reorder channels
    channels, data = reorder_channels_deap(data)
    # Get metadata
    num_video, subj_list, video_list = deap_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels


def load_deap_baseline_psd(data_path='EEG_Dataset/DEAP/deap_baseline1s_data.dat',
                 labels_path='EEG_Dataset/DEAP/deap_baseline1s_labels.dat',
                 label_by="valence", lower=3.5, higher = 6.5
                       ):
    data = np.fromfile(data_path).reshape(76800, 32, 128)
    labels = np.fromfile(labels_path).reshape(76800, 4)
    pid_video = np.array([v+p*40 for p in range(32) for v in range(40)])

    data = data[:, None, :, :]

    data = np.moveaxis(data.reshape(32, 60, 40, 32, 128), 1, 3)
    data = data.reshape(32*40, 1, 32, 60*128)
    # data = data
    # num, 1, 32, 128
    mask = np.ones((32, 60, 40))
    labels = labels.reshape(32,60,40,4)[:, 0].reshape(-1, 4)

    if label_by == "valence":
        labels_no_split = labels.reshape(32, 40, 4)
        mask_labels = parse_valence(labels_no_split)
        mask_labels = mask_labels.reshape(-1) == 1

        pid_video = pid_video[mask_labels]
        data = data[mask_labels]
        session_labels = (labels[:, 0].reshape(-1) > 5).astype(int)
        session_labels = session_labels[mask_labels]
    else:
        session_labels = (labels[:, 1] > 5).astype(int)

    all_data = np.zeros((0, 1, data.shape[2], 129))
    for ind in range(data.shape[0]):
        arr = np.zeros((0, 129))
        for ch in range(data.shape[2]):
            arr = np.concatenate([arr, [compute_psd(data[ind, 0, ch],
                    fs = 128, sec = 1, nfft = 256, noverlap = 128)[1]]], axis=0)
        all_data = np.concatenate([all_data, [[arr]]], axis=0)
    data = all_data

    # Reorder channels
    channels, data = reorder_channels_deap(data)
    # Get metadata
    num_video, subj_list, video_list = deap_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

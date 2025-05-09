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

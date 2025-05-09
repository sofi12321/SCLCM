import numpy as np 

def load_deap_de(data_path='/content/drive/MyDrive/EEG_Dataset/DEAP/deap_de_data_new.dat',
                 labels_path='/content/drive/MyDrive/EEG_Dataset/DEAP/deap_de_labels_new.dat',
                 label_by="valence", lower=3.5, higher = 6.5, data_shape=None, move_axis=None):
    # data in shape
    if data_shape is None:
        data = np.fromfile(data_path).reshape( (-1, 32, 5, 62))
    else:
        data = np.fromfile(data_path).reshape( data_shape)
    if move_axis is not None:
        data = np.moveaxis(data, move_axis[0], move_axis[1])

    labels = np.fromfile(labels_path).reshape( (-1, 40, 4))

    data = (data - np.mean(data, axis=-1)[:, :, :, None].repeat(data.shape[-1], axis=-1)) / np.std(data, axis=-1)[:, :, :, None].repeat(data.shape[-1], axis=-1)
    pid_video = make_DEAP_labels(pos_groupping="trial")
    # valence, arousal, dominance, liking
    if label_by == "valence":
        axiss = 0
    else:
        axiss = 1

    # Move to binary labels
    if label_by == "valence":
        labels_no_split = labels.reshape(32, 40, 4)
        mask_labels = parse_valence(labels_no_split)
        mask_labels = mask_labels.reshape(-1) == 1

        pid_video = pid_video[mask_labels]
        data = data[mask_labels]
        session_labels = (labels[:, :, axiss].reshape(-1) > 5).astype(int)
        session_labels = session_labels[mask_labels]

    elif lower < higher:
        mask = (labels[:, :, axiss].reshape(-1) <= lower) | (labels[:, :, axiss].reshape(-1) >= higher)
        session_labels = labels[:, :, axiss].reshape(-1)[mask]
        session_labels = (session_labels > higher).astype(int)
        pid_video = pid_video[mask]
        data = data[mask]
    else:
        session_labels = labels.reshape(-1, 4)[:, 1].round(0)

    # Reorder channels
    channels, data = reorder_channels_deap(data)
    # Get metadata
    num_video, subj_list, video_list = deap_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

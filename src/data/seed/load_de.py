import numpy as np
from data.seed.ancillary import make_SEED_labels, seed_metadata, reorder_channels_seed

def load_seed_de(data_path='EEG_Dataset/SEED/seed_de_new.dat', time_len=250, pos_groupping="trial", num_channels=62, data_shape=None, move_axis=None):
    if data_shape is None:
        data = np.fromfile(data_path).reshape( (-1, 62, 5, 288))
    else:
        data = np.fromfile(data_path).reshape( data_shape)
    if time_len is not None:
        data = data[:,:,:, -time_len:]
    if move_axis is not None:
        data = np.moveaxis(data, move_axis[0], move_axis[1])
    # Normalize
    data = (data - np.mean(data, axis=-1)[:, :, :, None].repeat(data.shape[-1], axis=-1)) / np.std(data, axis=-1)[:, :, :, None].repeat(data.shape[-1], axis=-1)

    session_labels, pid_video, num_video = make_SEED_labels(pos_groupping=pos_groupping)
    channels, data = reorder_channels_seed(data, num_channels=num_channels)
    num_video, subj_list, video_list = seed_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

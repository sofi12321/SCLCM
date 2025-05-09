import numpy as np 

def load_seed_raw(num_channels=62, segment_length=200, segment_num=60, general_path =  r"/content/drive/MyDrive/EEG_Dataset/SEED/"):
    if segment_length*segment_num > 37000:
        print("segment_length*segment_num is greater than 37 000, segment_num is set to", 37000//segment_length)
        segment_num = 37000 // segment_length

    # Prepare to load data
    # Sort files to be loaded by pid and date in name
    true_labels = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
    # Load data from files
    all_data = np.zeros((0, segment_num, 1, 62, segment_length))
    pbar = tqdm(total=675)
    for f in ['seed_raw_0_7.npz', 'seed_raw_8_14.npz']:
        data_list = np.load(general_path + f)
        for k in data_list.keys():
            # Load data
            all_data = np.concatenate([all_data,
                [np.moveaxis(data_list[k][:, -segment_num*segment_length:].reshape((1, 62, segment_num, segment_length)), 2, 0)]])
            # print(all_data.shape)
            pbar.update(1)
    session_labels = np.concatenate([np.array(true_labels) for _ in range(15*3)])
    pid_video = np.array([v+p*15 for p in range(15) for t in range(3) for v in range(15)])

    channels, data = reorder_channels_seed(all_data.reshape(-1, 1, 62, segment_length), num_channels=num_channels)
    num_video, subj_list, video_list = seed_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

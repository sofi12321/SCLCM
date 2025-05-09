import numpy as np

def load_seed_baseline(data_path='EEG_Dataset/SEED/Baseline/seed_baseline1s_60_data.dat',
                       sec=10, num_channels=62, total_len=128, N=60):
    # metadata
    FREQ = 200
    TOTAL_LEN = 128
    N = 60
    true_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    data = np.fromfile(data_path).reshape(-1, 62, total_len)
    session_labels = np.concatenate([np.array(true_labels).repeat(N) for _ in range(15*3)])
    pid_video = np.array([v+p*15 for p in range(15) for t in range(3) for v in range(15) for b in range(N)])

    # Make for 10 sec
    if not ((sec >= 1) and (60 % sec == 0)):
        sec = 1
        print(f"Replace sec to 1, because {sec} is not valid")

    data = np.moveaxis(data.reshape(-1, sec, 62, total_len), 1, 2).reshape(-1, 1, 62, total_len * sec)
    session_labels = session_labels[::sec]
    pid_video = pid_video[::sec]

    channels, data = reorder_channels_seed(data, num_channels=num_channels)
    num_video, subj_list, video_list = seed_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

import numpy as np
import tqdm
from data.seed.ancillary import seed_metadata, reorder_channels_seed
from data.general.feature_extraction import compute_psd

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

def load_seed_baseline_psd(num_channels=62, segment_length=200, segment_num=60, baseline_length=200, baseline_num = 3, general_path =  r"/content/drive/MyDrive/EEG_Dataset/SEED/"):
    if segment_length*segment_num + baseline_length*baseline_num > 37000:
        print("segment_length*segment_num + baseline_length*baseline_num is greater than 37 000, so segment_num is set to", (37000 - baseline_length* baseline_num)//segment_length)
        segment_num = (37000 - segment_length* baseline_num)//segment_length

    # Prepare to load data
    # Sort files to be loaded by pid and date in name
    true_labels = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
    # Load data from files
    all_data = np.zeros((0, 1, 62, 129))
    pbar = tqdm(total=675)
    for f in ['seed_raw_0_7.npz', 'seed_raw_8_14.npz']:
        data_list = np.load(general_path + f)
        for k in data_list.keys():
            # Load data
            d = data_list[k][:, -segment_length * segment_num:]
            # Subtract baseline
            mm = data_list[k][:, :baseline_length*baseline_num].reshape(-1, baseline_num, baseline_length)
            mm = np.mean(mm, axis=1)
            mm = mm[:, None, :].repeat(segment_num, axis=-2).reshape(d.shape)
            d = d - mm
            # Normalize d
            d = (d - np.mean(d, axis=-1)[:,  None].repeat(segment_num * segment_length, axis=-1)) / np.std(d, axis=-1)[:,  None].repeat(segment_num * segment_length, axis=-1)

            d = d.reshape(-1, segment_num * segment_length)
            data_psd = np.zeros((0, 129))
            for ch in range(d.shape[0]):
                data_psd = np.concatenate([data_psd, [compute_psd(d[ch],
                        fs = 200, sec = 1, nfft = 256, noverlap = 128)[1]]], axis=0)

            all_data = np.concatenate([all_data,
                                      [[data_psd]]])
            pbar.update(1)
    session_labels = np.concatenate([np.array(true_labels) for _ in range(15*3)])
    pid_video = np.array([v+p*15 for p in range(15) for t in range(3) for v in range(15)])

    channels, data = reorder_channels_seed(all_data, num_channels=num_channels)
    num_video, subj_list, video_list = seed_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

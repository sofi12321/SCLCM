import numpy as np

def load_deap_raw( length=128, general_path =  r"/content/DEAP/"):
    num = 60 * 128 // length

    # !mkdir DEAP
    !unzip -q /content/drive/MyDrive/EEG_Dataset/DEAP/data_preprocessed_matlab.zip -d DEAP


    # Prepare to load data
    # Sort files to be loaded by pid and date in name
    all_files = [f for f in os.listdir(general_path) if (f[-3:] == "mat") and ("s" == f[0])]
    all_files = sorted(all_files,
                        key=lambda x: int(x.split(".")[0][1:]))
    print(all_files)

    labels = []
    # Load data from files
    labels = np.zeros((0, 4))
    all_data = np.zeros((0, num , 1, 32, length))

    for f in all_files:
        inf = scipy.io.loadmat(os.path.join(general_path, f))

        d = inf["data"][:, :32, -num*length:]
        d = np.moveaxis(d.reshape((-1, 1, 32, num, length)), 3, 1)

        all_data = np.concatenate( [all_data, d] , axis=0)
        labels = np.concatenate([labels, inf["labels"]], axis=0)

        # 40, 40 subj, 40 subj for 20 iters
        print(f)
        print(all_data.shape)
        print(labels.shape)

    all_data.tofile('deap_raw_1s_data.dat')
    labels.tofile('deap_raw_1s_labels.dat')

    pid_video = np.array([v + p *40 for p in range(32) for v in range(40)])

    # valence, arousal, dominance, liking
    mask_labels = parse_valence(labels.reshape(32, 40, 4)).reshape(-1) == 1
    labels = labels[mask_labels]
    data = all_data[mask_labels]
    pid_video = pid_video[mask_labels]

    session_labels = (labels[:, 0] > 5).astype(int)

    num_video, subj_list, video_list  = deap_metadata()

    session_labels = session_labels.repeat(60)
    pid_video= pid_video.repeat(60)
    chs, data = reorder_channels_deap(data.reshape(-1, 1, 32, length))

    # Reorder channels
    channels, data = reorder_channels_deap(data)
    # Get metadata
    num_video, subj_list, video_list = deap_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels


def load_deap_raw_psd(data_path='deap_raw_1s_data.dat',
                 labels_path='deap_raw_1s_labels.dat',
                 label_by="valence", lower=3.5, higher = 6.5, length=128
                       ):
    num = 60 * 128 // length
    data = np.fromfile(data_path).reshape(-1, num , 1, 32, length)
    data = np.moveaxis(data, 1, 3).reshape(-1, 1, 32, num*length)
    labels = np.fromfile(labels_path).reshape(32*40, 4)
    pid_video = np.array([v + p *40 for p in range(32) for v in range(40)])

    print(data.shape)
    # data = data
    # N, 1, 32, length*num

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

    all_data = np.zeros((0, 1, data.shape[2], 129))
    for ind in range(data.shape[0]):
        arr = np.zeros((0, 129))
        for ch in range(data.shape[2]):
            arr = np.concatenate([arr, [compute_psd(data[ind, 0, ch],
                    fs = 128, sec = 60, nfft = 256, noverlap = 128)[1]]], axis=0)
        all_data = np.concatenate([all_data, [[arr]]], axis=0)

    # Reorder channels
    channels, data = reorder_channels_deap(all_data)
    # Get metadata
    num_video, subj_list, video_list = deap_metadata()

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

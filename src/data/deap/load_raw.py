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

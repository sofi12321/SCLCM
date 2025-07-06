import os 
import numpy as np 
import tqdm
import scipy

def reformat_seed(length = 37000, general_path =  r"/content/drive/MyDrive/EEG_Dataset/SEED/"):
    # Prepare to load data
    # Sort files to be loaded by pid and date in name
    all_files = [f for f in os.listdir(general_path) if (f[-3:] == "mat") and ("_" in f)]
    all_files = sorted(all_files,
                        key=lambda x: (int(x.split("_")[0]),
                                        int(x.split(".")[0].split("_")[1])
                                        ))
    # Load data from files
    all_data = np.array([])
    for f in tqdm.tqdm(all_files):
        d = scipy.io.loadmat(os.path.join(general_path, f))

        for r in range(15):
            # Load data
            eeg_name = [k for k in d.keys() if k[-2:] == "13"][0][:-2] + str(r + 1)
            part_data = d[eeg_name]
            leni = part_data.shape[1]
            # print(part_data.shape)
            part_data = part_data[:, -length:]
            for _ in range(10):
                gc.collect()
           
            if all_data.shape[0] == 0:
                # Store the first item
                all_data = np.array([part_data])
            else:
                # Add to already loaded data
                all_data = np.concatenate(
                    [all_data, [part_data]])
        # Save data to file every iteration to be able to continue process in case of small memory
        all_data.tofile(os.path.join(general_path, 'seed_all.dat') )
    data = all_data[:, None, :, :]
    
    # Format metadata
    session_labels, pid_video, num_video = make_SEED_labels(pos_groupping=pos_groupping)
    channels, data = reorder_channels_seed(data, num_channels=num_channels)
    num_video, subj_list, video_list = seed_metadata()
    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

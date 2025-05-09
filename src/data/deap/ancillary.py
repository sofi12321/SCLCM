import numpy as np
import pandas as pd

def deap_metadata():
    num_video = 40
    subj_list = np.array([26, 27, 28, 29, 30, 31])
    video_list = np.array([0,1,2,3,4,5,6,7])
    return num_video, subj_list, video_list

def make_DEAP_labels(pos_groupping="trial"):
    # labels = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
    # labels_edited = np.concatenate([labels for _ in range(45)])

    if pos_groupping == "video":
        pid_video = np.array([v  for p in range(32) for v in range(40)])
    elif pos_groupping == "person":
        pid_video = np.array([p for p in range(32) for v in range(40)])
    elif pos_groupping == "trial":
        pid_video = np.array([v + p*40 for p in range(32) for v in range(40)])
    else:
        pid_video = list(range(32*40))
    pid_video = np.array(pid_video)
    return pid_video
    
def parse_valence(labels, threshold = 20):
    valence = (labels[:,:,0] > 6)*1 + ((labels[:,:,0] <= 6) & (labels[:,:,0] >= 5))*2
    df = pd.DataFrame(valence, columns = [f"video_{i+1}" for i in range(valence.shape[1])])
    df1 = pd.concat(((df == 0).sum(), (df == 1).sum(), (df == 2).sum()), axis = 1)
    # df2 = pd.concat((df1[0]/df1.sum(axis = 1), df1[1]/df1.sum(axis = 1), df1[2]/df1.sum(axis = 1)), axis = 1)
    # plt.figure()
    # df1.plot.bar(figsize=(18,3))
    df3 = df1[df1 >= threshold].idxmax(axis = 1)
    result = []
    for j in range(40):
        tmp = []
        val = df3.loc[f"video_{j+1}"]
        # print(val)
        for i in range(32):
            if np.isnan(val):
                tmp.append(0)
            elif df.iloc[i, j] == val:
                tmp.append(1)
            else:
                tmp.append(0)
        result.append(np.array(tmp))
    return np.array(result).T

def subset_DEAP_subject(data, labels, pids, subj_list):
    """
    Extracts a subset of data, labels, and participant IDs for specified subjects from the SEED dataset.

    Args:
        data (numpy.ndarray): The dataset containing the data samples.
        labels (numpy.ndarray): The array of labels corresponding to the data samples.
        pids (numpy.ndarray): The array of participant IDs corresponding to the data samples.
        subj_list (list of int): A list of subject IDs for which the data, labels, and pids should be extracted.

    Returns:
        A tuple containing:
            - subset_data (numpy.ndarray): The subset of data samples for the specified subjects.
            - subset_labels (numpy.ndarray): The subset of labels for the specified subjects.
            - subset_pids (numpy.ndarray): The subset of participant IDs for the specified subjects.
    """
    sh = [0] + np.array(data.shape).tolist()[1:]
    subset_data = np.zeros(sh)
    subset_labels = []
    subset_pids = []

    for subject_id in range(32):
        n_start = subject_id * 40
        n_end = (subject_id + 1) * 40

        if subject_id in subj_list:
            subset_labels = np.concatenate([subset_labels, labels[n_start:n_end]])
            subset_data = np.concatenate([subset_data , data[n_start:n_end]])
            subset_pids = np.concatenate([subset_pids, pids[n_start:n_end]])

    return subset_data, subset_labels, subset_pids
    

def reorder_channels_deap(data):
    deap_channels = np.array([
        'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5',
        'CP1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4',
        'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6',
        'CP2', 'P4', 'P8', 'PO4', 'O2'
    ])
    chosen_channels_32 = np.array([
        'FP1', 'FP2', 'AF4', 'AF3', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC6',
        'FC2', 'FC1', 'FC5', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP6', 'CP2',
        'CP1', 'CP5', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO4', 'PO3', 'O1',
        'OZ', 'O2'])

    return chosen_channels_32, data[:,:, np.sum((deap_channels.reshape(-1, 1) == chosen_channels_32).astype(int)*np.array(range(1,len(deap_channels) + 1)).reshape(-1, 1), axis=0) - 1, :]

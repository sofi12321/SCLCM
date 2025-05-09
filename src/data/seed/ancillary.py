import numpy as np

def seed_metadata():
    num_video = 15
    subj_list = sorted([4, 7, 10])
    video_list = [9,10,11]
    return num_video, subj_list, video_list
    
def reorder_channels_seed(data, num_channels=62):
    seed_channels = np.array(['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'])
    if num_channels == 62:
        chosen_channels = np.array([
            'FP1', 'FPZ', 'FP2', 'AF4', 'AF3',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'FC1', 'FC3', 'FC5', 'FT7',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
            'TP8', 'CP6', 'CP4', 'CP2', 'CPZ', 'CP1', 'CP3', 'CP5', 'TP7',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO8', 'PO6', 'PO4', 'POZ', 'PO3', 'PO5', 'PO7',
            'CB1', 'O1', 'OZ', 'O2', 'CB2'])
    else:
        chosen_channels = np.array([
            'FP1', 'FP2', 'AF4', 'AF3', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC6',
            'FC2', 'FC1', 'FC5', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP6', 'CP2',
            'CP1', 'CP5', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO4', 'PO3', 'O1',
            'OZ', 'O2'])

    return chosen_channels, data[:, :, np.sum((seed_channels.reshape(-1, 1) == chosen_channels).astype(int)*np.array(range(1,len(seed_channels)+1)).reshape(-1, 1), axis=0) - 1, :]


def make_SEED_labels(pos_groupping="trial"):
    """
    Generates SEED labels and positional groupings based on the specified grouping method.

    Args:
        pos_groupping (str): The method of grouping positions. It can be one of the following:
            - "video": Groups by video, repeating the sequence of labels for each video.
            - "person": Groups by person, repeating the sequence of labels for each person.
            - "trial": Groups by trial, adjusting the sequence of labels for each trial.
            - Any other string will result in a default sequential grouping.

    Returns:
        A tuple containing:
            - labels_edited (np.ndarray): An array of repeated labels.
            - pid_video (np.ndarray): An array representing the positional grouping based on the specified method.
    """
    labels = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
    labels_edited = np.concatenate([labels for _ in range(45)])
    num_video  = 15
    if pos_groupping == "video":
        pid_video = np.concatenate([np.array(list(range(15)))  for i in range(45)])
    elif pos_groupping == "person":
        pid_video = np.array([p   for p in range(15) for _ in range(45)])
    elif pos_groupping == "trial":
        pid_video = np.concatenate([np.array(list(range(15))) + i//3*15  for i in range(45)])
    else:
        pid_video = np.array([ v + t*15 + p*15*3 for p in range(15) for t in range(3)  for v in range(15)])
        num_video = 15*3
    pid_video = np.array(pid_video)
    return labels_edited, pid_video, num_video


def subset_SEED_subject(data, labels, pids, subj_list):
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

    for subject_id in range(15):
        n_start = subject_id * 45
        n_end = (subject_id + 1) * 45

        if subject_id in subj_list:
            subset_labels = np.concatenate([subset_labels, labels[n_start:n_end]])
            subset_data = np.concatenate([subset_data , data[n_start:n_end]])
            subset_pids = np.concatenate([subset_pids, pids[n_start:n_end]])

    return subset_data, subset_labels, subset_pids

def subset_SEED_video(data, labels, pids, video_list):
    """
    Filters the dataset to include only the specified video indices.

    Args:
        data (np.ndarray): The dataset containing the video data.
        labels (np.ndarray): The array of labels corresponding to the data.
        pids (np.ndarray): The array of participant IDs corresponding to the data.
        video_list (list): A list of video indices to include in the subset.

    Returns:
        A tuple containing:
            - subset_data (np.ndarray): The filtered dataset containing only the specified videos.
            - subset_labels (np.ndarray): The labels corresponding to the filtered dataset.
            - subset_pids (np.ndarray): The participant IDs corresponding to the filtered dataset.
    """
    mask = np.isin(pids % 15, video_list)
    #Splitting Dataset into train, validation, test
    subset_labels = labels[mask]
    subset_data = data[mask]
    subset_pids = pids[mask]

    return subset_data, subset_labels, subset_pids

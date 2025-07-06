from data.deap.load_row import load_deap_raw

def reformat_deap(data_path):
    # Reformating is made by loading raw data
    data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_deap_raw( length=60*128, data_path)
    return data, session_labels, pid_video, num_video, subj_list, video_list, channels

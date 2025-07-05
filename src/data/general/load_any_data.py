from data.seed.load_raw import load_seed_raw
from data.seed.load_de import load_seed_de
from data.seed.load_baseline import load_seed_baseline, load_seed_baseline_psd

from data.deap.load_raw import load_deap_raw
from data.deap.load_de import load_deap_de
from data.deap.load_baseline import load_deap_baseline, load_deap_baseline_psd

from data.general.feature_extraction import extract_dasm, extract_rasm, extract_dcau, reshape_3d
import numpy as np

def load_data_by_selection(selection, general_metadata, dataset_metadata, channels_cut):

    # Load data 
    if selection == "seed_raw":
        # Load SEED raw
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_seed_raw(
                num_channels =   dataset_metadata["num_channels"],
                segment_length = dataset_metadata["segment_length"], 
                segment_num =    dataset_metadata["segment_num"],
                general_path =   dataset_metadata["data_path"])
    elif selection == "seed_baseline":
        # Load SEED baseline
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_seed_baseline(
                data_path    = dataset_metadata["data_path"],
                sec          = dataset_metadata["sec"], 
                num_channels = dataset_metadata["num_channels"])
    elif selection in ["seed_de", "seed_psd", "seed_de_baseline"]:
        # Load SEED DE
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_seed_de(
                data_path     = dataset_metadata["data_path"],
                data_shape    = dataset_metadata["data_shape"], 
                move_axis     = dataset_metadata["move_axis"],
                time_len      = dataset_metadata["time_len"], 
                pos_groupping = dataset_metadata["pos_groupping"], 
                num_channels  = dataset_metadata["num_channels"])
    elif selection == "seed_baseline_psd":
        # Load SEED baseline
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_seed_baseline_psd(
                general_path    = dataset_metadata["data_path"],
                num_channels    = dataset_metadata["num_channels"], 
                segment_length  = dataset_metadata["segment_length"], 
                segment_num     = dataset_metadata["segment_num"],
                baseline_length = dataset_metadata["baseline_length"], 
                baseline_num    = dataset_metadata["baseline_num"]
                )
    
    elif selection == "deap_raw":
        # Load DEAP raw
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_deap_raw(
                length = dataset_metadata["length"])
    elif selection in ["deap_de", "deap_baseline_de", "deap_psd"]:
        # Load DEAP DE
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_deap_de(
                data_path   = dataset_metadata["data_path"],
                labels_path = dataset_metadata["labels_path"],
                data_shape   = dataset_metadata["data_shape"],
                move_axis   = dataset_metadata["move_axis"],
                label_by    = dataset_metadata["label_by"], 
                lower       = dataset_metadata["lower"], 
                higher      = dataset_metadata["higher"]
                )
    elif selection == "deap_baseline":
        # Load DEAP baseline
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_deap_baseline(
                data_path   = dataset_metadata["data_path"],
                labels_path = dataset_metadata["labels_path"],
                label_by    = dataset_metadata["label_by"], 
                lower       = dataset_metadata["lower"], 
                higher      = dataset_metadata["higher"]
                )
    elif selection == "deap_baseline_psd":
        data, session_labels, pid_video, num_video, subj_list, video_list, channels = load_deap_baseline_psd(
                data_path   = dataset_metadata["data_path"],
                labels_path = dataset_metadata["labels_path"],
                label_by    = dataset_metadata["label_by"], 
                lower       = dataset_metadata["lower"], 
                higher      = dataset_metadata["higher"]
                )
    else:
        print("Choose dataset, please")
        return [None for i in range(5)] 
    
    # Take required metadata
    need_norm    = dataset_metadata["need_norm"]
    in_channels  = dataset_metadata["in_channels"]
    in_features  = dataset_metadata["in_features"]
    num_classes  = dataset_metadata["num_classes"]
    num_channels = dataset_metadata["num_channels"] 

    # Process 
    if general_metadata["additional_processing"] == "DASM":
        data = extract_dasm(data, channels)
    elif general_metadata["additional_processing"] == "RASM":
        data = extract_rasm(data, channels)
    elif general_metadata["additional_processing"] == "DCAU":
        data = extract_dcau(data, channels)
    
    if general_metadata["spatial_transform"]:
        data, in_channels, _ = reshape_3d(data, channels, num_channels)
    elif channels_cut and len(channels_cut) < len(channels) and  general_metadata["additional_processing"] is None:
        data = data[: , :, np.isin(channels, channels_cut), :] 
        channels = channels_cut

    return data, session_labels, pid_video, num_video, subj_list, video_list, channels, need_norm, in_channels, in_features, num_classes, num_channels

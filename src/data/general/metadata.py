general_metadata = {
    "batch_size": 512,
    "additional_processing": None, # None, "DASM", "RASM" or "DCAU"
    "spatial_transform": False,
    "augmentations": ["frequency_noise","mask_channel","mix_dtpts"],
    "split_by": "subject", # "subject" - for subject-independent; "random" - subject dependent
    "test_fraction": 0.3
}
datasets_metadata = {
    "seed_raw": { # one second lenght
        "data_path": "EEG_Dataset/SEED/Raw/",
        "num_channels": 62,
        "segment_length":200, 
        "segment_num":60,
        "need_norm": True,
        "in_channels": 1,
        "in_features": 96000,
        "num_classes": 3  
    }, 
    "seed_baseline": { # one second lenght stimulated recording minus one second average of resting state
        "data_path": "EEG_Dataset/SEED/Baseline/seed_baseline1s_60_data.dat",
        "num_channels": 62,
        "sec": 1,
        "need_norm": False,
        "in_channels": 1,
        "in_features": 614400,
        "num_classes": 3  
    }, 
    "seed_de": { # all video processed by DE features sampled last 250 features
        "data_path": "EEG_Dataset/SEED/seed_de_new.dat",
        "num_channels": 62,
        "data_shape": (-1, 62, 5, 288),
        "move_axis": [2, 1],
        "time_len": 250, 
        "pos_groupping":"trial",
        "need_norm": True,
        "in_channels": 5,
        "in_features": 119040,
        "num_classes": 3  
    }, 
    "seed_baseline_de": { # all video processed by DE features sampled last 250 features
        "data_path": "EEG_Dataset/SEED/Baseline/seed_de_baseline_data.dat",
        "num_channels": 62,
        "data_shape": (-1, 62, 5, 280),
        "move_axis": [2, 1],
        "time_len": 280, 
        "pos_groupping":"trial",
        "need_norm": True,
        "in_channels": 5,
        "in_features": 134400,
        "num_classes": 3  
    },
    "seed_psd": { # 
        "data_path": "EEG_Dataset/SEED/PSD/seed_psd_data.dat",
        "num_channels": 62,
        "data_shape": (-1, 1, 62, 129),
        "move_axis": None,
        "time_len": None, 
        "pos_groupping":"trial",
        "need_norm": True,
        "in_channels": 1,
        "in_features": 119040,
        "num_classes": 3  
    }, 
    "seed_baseline_psd": { # 
        "data_path": "EEG_Dataset/SEED/Raw/",
        "num_channels": 62,
        "segment_length":200, 
        "segment_num":60,
        "baseline_length":200, 
        "baseline_num":60,
        "need_norm": False,
        "in_channels": 1,
        "in_features": 119040,
        "num_classes": 3  
    }, 

    "deap_raw": { # one second lenght
        "data_path": "EEG_Dataset/DEAP/data_preprocessed_matlab.zip",
        "length": 128,
        "need_norm": True,
        "in_channels": 1,
        "in_features": 32768,
        "num_classes": 2
    }, 
    "deap_baseline": { # all video processed by DE features sampled last 250 features
        "data_path": "EEG_Dataset/DEAP/deap_baseline1s_data.dat",
        "labels_path": "EEG_Dataset/DEAP/deap_baseline1s_labels.dat",
        "label_by": "valence",
        "lower": 3.5,
        "higher": 6.5,
        "need_norm": False,
        "in_channels": 1,
        "in_features": 32768,
        "num_classes": 2
    }, 
    "deap_de": { 
        "data_path": "EEG_Dataset/DEAP/deap_de_data_new.dat",
        "labels_path": "EEG_Dataset/DEAP/deap_de_labels_new.dat",
        "data_shape" : None,
        "move_axis": [2, 1],
        "label_by": "valence",
        "lower": 3.5,
        "higher": 6.5,
        "need_norm": True,
        "in_channels": 5,
        "in_features": 15360,
        "num_classes": 2
    }, 
    "deap_baseline_de": { 
        "data_path": "EEG_Dataset/DEAP/deap_baseline_de_data.dat",
        "labels_path": "EEG_Dataset/DEAP/deap_baseline_de_labels.dat",
        "data_shape": (1280, 32, 59, 5),
        "move_axis": [3, 1],
        "label_by": "valence",
        "lower": 3.5,
        "higher": 6.5,
        "need_norm": True,
        "in_channels": 5,
        "in_features": 15360,
        "num_classes": 2
    }, 
    "deap_psd": { 
        "data_path": "EEG_Dataset/DEAP/PSD/deap_psd_data.dat",
        "labels_path": "EEG_Dataset/DEAP/PSD/deap_psd_labels.dat",
        "data_shape": (32* 40, 1, 32, 129),
        "move_axis": None,
        "label_by": "valence",
        "lower": 3.5,
        "higher": 6.5,
        "need_norm": True,
        "in_channels": 1,
        "in_features": 15360,
        "num_classes": 2
    }, 
    "deap_baseline_psd": { 
        "data_path": "EEG_Dataset/DEAP/deap_baseline1s_data.dat",
        "labels_path": "EEG_Dataset/DEAP/deap_baseline1s_labels.dat",
        "label_by": "valence",
        "lower": 3.5,
        "higher": 6.5,
        "need_norm": True,
        "in_channels": 1,
        "in_features": 15360,
        "num_classes": 2
    }
}

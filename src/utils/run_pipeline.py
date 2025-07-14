import torch
from torch import optim
import torch.nn as nn
import numpy as np

from utils.imports import set_seed
from data.general.load_any_data import load_data_by_selection
from data.general.train_test_split import split_data
from models.cnn.encoder import SmallCNN1d, SmallCNN2d
from utils.losses import HardSoftLoss
from utils.test_run_embedding import run_embedding
from utils.visualization import plot_pca_tsne_pids
from utils.test_run_classification import run_classification
from utils.predict import print_classification_report

def run_pipeline(num_exp, dataset_selection, general_metadata, dataset_metadata, 
            emb_dim=128, model_dim=None, channels_cut=None, in_features=None, seed=21,
            pretraining_epochs = 200, pretraining_wait_epochs = 25, 
            finetuning_epochs = 200, finetuning_wait_epochs = 25, 
            sensitivity_early_stop = 1.00001, sensitivity_save_weights = 1.00001,
            save_weights_path = "model_weights/",
            need_lr_scheduler=False, need_encoder_freezing=False
):
    """
    Main pipeline for training and evaluating a CNN model with pretraining and fine-tuning.
    
    Args:
        num_exp (str): Experiment naming to store weights and imgs
        dataset_selection (str): Dataset and feature axtraction identifier
        general_metadata (dict): General metadata parameters
        dataset_metadata (dict): Dataset-specific metadata parameters
        emb_dim (int): Embedding dimension size
        model_dim (int): Model dimension (1D or 2D)
        channels_cut (list): List of channels to use
        in_features (int): Input feature size
        seed (int): Random seed for reproducibility
        pretraining_epochs (int): Number of pretraining epochs
        pretraining_wait_epochs (int): Number of epochs to wait before early stopping in pretraining
        finetuning_epochs (int): Number of fine-tuning epochs
        finetuning_wait_epochs (int): Number of epochs to wait before early stopping in fine-tuning
        sensitivity_early_stop (float): Sensitivity threshold for early stopping
        sensitivity_save_weights (float): Sensitivity threshold for saving weights
        save_weights_path (str): Path to save model weights
        need_lr_scheduler (bool): Whether to use learning rate scheduler
        need_encoder_freezing (bool): Whether to freeze encoder during fine-tuning
        
    Returns:
        ft_model: Fine-tuned model
    """
    # Set seed to reproduce experiments
    set_seed(seed)
    in_features_init = in_features
    # Load data
    data, session_labels, pid_video, num_video, subj_list, video_list, channels, need_norm, in_channels, in_features, num_classes, num_channels = load_data_by_selection(dataset_selection, general_metadata, dataset_metadata, channels_cut)
    if in_features_init:
        in_features = in_features_init
    # Make train test split
    train_loader_pt, test_loader_pt, train_loader_ft, test_loader_ft = split_data(
            data, session_labels, pid_video,
            num_video, subj_list, video_list,
            need_norm = need_norm, add_data = None, batch_size = general_metadata["batch_size"],
            augmentations = general_metadata["augmentations"],
            split_by=general_metadata["split_by"], test_fraction=general_metadata["test_fraction"])
    # Set device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create encoder model
    model = SmallCNN2d(emb_dim,  in_channels=in_channels, in_features=in_features)
    if model_dim == 1 or len(channels_cut) < 6:
        model = SmallCNN1d(emb_dim,  in_channels=in_channels, in_features=in_features  )

    # Prepare for training
    save_pt_model_path = save_weights_path + "cnn_pt_"+num_exp+".pth"
    finetuning_criterion = HardSoftLoss(pos=0.4, hard_neg=0.4, soft_person_neg=0.4, soft_video_neg=0.1, num_video=num_video, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    pt_scheduler = None
    if need_lr_scheduler:
        pt_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85)
    model = model.to(device)

    # Pre-training
    _, losses, pt_train_time, pt_test_time = run_embedding(device,save_pt_model_path, model,
                    train_loader_pt, test_loader_pt,
                    optimizer, finetuning_criterion,

                    finetuning_epochs, finetuning_wait_epochs,
                    sensitivity_early_stop = sensitivity_early_stop,
                    sensitivity_save_weights = sensitivity_save_weights,
                    pt_scheduler =pt_scheduler)
    # Print statistics
    print()
    print("First loss is", round(losses[0], 4))
    print("Best loss is", round(min(losses), 4), "at epoch", np.argmin(losses))
    print("Train time", round(pt_train_time, 3), "Test time",  round(pt_test_time, 3))
    # Save overfitted model
    torch.save(model.state_dict(), save_weights_path + "cnn_ovpt_"+num_exp+".pth")

    # Load best model
    model.load_state_dict(torch.load(save_pt_model_path, weights_only=True))


    # Visualize embeddings
    # One color mean one positive pair
    plot_pca_tsne_pids(f"Pretrained embedding with augmentations", f"outputs/"+ num_exp+"_pt_vid_aug",
            model, test_loader_pt, num_video, num_elements=500, show_vid=True, device=device)

    from models.cnn.classifier import get_sequential_classifier
    # Create classificator based on the pre-trained model
    ft_model = get_sequential_classifier(model, num_classes, emb_dim=emb_dim)

    # If needed layers freezing
    if need_encoder_freezing:
        for param in ft_model[0].parameters():
            param.requires_grad = False

    # Prepare for training
    save_final_model_path = save_weights_path + "cnn_ft_"+num_exp+".pth"

    optimizer = optim.Adam(ft_model.parameters(), lr=0.0001)
    ft_scheduler = None
    if need_lr_scheduler:
        ft_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85)
    # Select loss based on the number of features
    finetuning_criterion = nn.CrossEntropyLoss()
    if num_classes == 2:
        finetuning_criterion = nn.BCEWithLogitsLoss()
    ft_model = ft_model.to(device)

    # Fine-tuning
    _, losses, accs, ft_train_time, ft_test_time = run_classification(device, ft_model,
                    train_loader_ft, test_loader_ft,
                    optimizer, finetuning_criterion,
                    finetuning_epochs, finetuning_wait_epochs,
                    sensitivity_early_stop = sensitivity_early_stop,
                    sensitivity_save_weights = sensitivity_save_weights,
                    save_final_model_path=save_final_model_path, pt_scheduler =ft_scheduler)

    # Print statistics
    print()
    print("First loss", round(losses[0], 4))
    print("Best loss", round(min(losses), 4), "at epoch", np.argmin(losses))
    print()
    print("Stable accuracy", str(round(100*accs[np.argmin(losses)], 1)) + "%")
    print("Best accuracy", str(round(100*max(accs), 1))+ "% at epoch", np.argmax(accs))
    print("Train time", round(ft_train_time, 3), "sec \nTest time",  round(ft_test_time, 3), "sec")

    # Load best model
    ft_model.load_state_dict(torch.load(save_final_model_path, weights_only=True))
    # Print results per class
    print_classification_report(ft_model, test_loader_ft, num_classes, device='cpu')

    return ft_model

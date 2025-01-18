import numpy as np
import torch
from torch import nn
from itertools import combinations

def obtain_criterion(task_type, output_expl):
    """
    Return function to calculate loss. Appropi
    :param task_type: "embedding_extraction" or "classification"
    :param output_expl: type of loss or number of classes
    :return: 
    """
    if task_type == "classification":
        if output_expl == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            print("cross entropy")
    else:
        # "embedding_extraction"
        if output_expl.upper() == "CLOCS":
            # if output_expl in ["MS", "ML", "MSML"]:
            criterion = contrastive_loss_patient_specific
        # else:
        #     # SimCLR
        #     criterion = contrastive_loss_simclr
    return criterion


def contrastive_loss_patient_specific(embeddings, pids):
    """
    Patient-specific contrastive loss
    :param embeddings: embeddings of shape (batch, view, time_len)
    :param pids: list of patient indices of shape (batch,)
    :return: scalar loss
    """
    # Prepare matrix to handle in-patient similarity
    pids = np.array(pids, dtype=object)
    pid1, pid2 = np.meshgrid(pids, pids)
    pid_matrix = pid1 + '-' + pid2
    # find positions of same pids
    pids_of_interest = np.unique(pids + '-' + pids)
    bool_matrix_of_interest = np.zeros((len(pids), len(pids)))
    for pid in pids_of_interest:
        bool_matrix_of_interest += pid_matrix == pid
    rows1, cols1 = np.where(np.triu(bool_matrix_of_interest, 1))
    rows2, cols2 = np.where(np.tril(bool_matrix_of_interest, -1))

    # Compute loss along views
    # Along channels for ML, time frames for MS, and both for MSML
    nviews = set(range(embeddings.shape[1]))
    view_combinations = combinations(nviews, 2)
    loss = 0
    loss_terms = 2
    ncombinations = 0
    for combination in view_combinations:
        view1_array = embeddings[:, combination[0], :]
        view2_array = embeddings[:, combination[1], :]
        norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
        norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(view1_array, view2_array.transpose(0, 1))
        norm_matrix = torch.mm(norm1_vector.transpose(0, 1), norm2_vector)
        temperature = 0.1
        argument = sim_matrix / (norm_matrix * temperature)
        sim_matrix_exp = torch.exp(argument)

        diag_elements = torch.diag(sim_matrix_exp)

        triu_sum = torch.sum(sim_matrix_exp, 1)
        tril_sum = torch.sum(sim_matrix_exp, 0)

        # Calculate for diagonal elements (itself)
        loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
        loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))

        loss = loss_diag1 + loss_diag2
        loss_terms = 2

        # Calculate for upper triangle
        if len(rows1) > 0:
            triu_elements = sim_matrix_exp[rows1, cols1]
            loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[rows1]))
            loss += loss_triu
            loss_terms += 1

        # Calculate for lower triangle
        if len(rows2) > 0:
            tril_elements = sim_matrix_exp[rows2, cols2]
            loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[cols2]))
            loss += loss_tril
            loss_terms += 1

        ncombinations += 1
    loss = loss / (loss_terms * ncombinations)
    return loss

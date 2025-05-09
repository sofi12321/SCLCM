import numpy as np
import torch
from torch import nn
from itertools import combinations

class BasicLoss(nn.Module):
    def __init__(self, theta = 0.5, a=0.5):
        """
        Initializes the BasicLoss module with specified parameters.

        Args:
            theta (float, optional): A scaling factor for the similarity matrix. Default is 0.5.
            a (float, optional): A weighting factor for the loss calculation. Default is 0.5.
        """
        super(BasicLoss, self).__init__()
        self.a = a
        self.theta = theta

    def similarity(self, x):
        """
        Computes the similarity matrix for the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A similarity matrix where each element represents the cosine similarity between pairs of input еутыщк.
        """
        x_norm = nn.functional.normalize(x, dim = 1)
        return x_norm @ x_norm.T

    def sigma(self, x):
        """
        Applies the sigmoid function element-wise to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The result of applying the sigmoid function to each element of the input tensor.
        """
        return nn.functional.sigmoid(x)

    def forward(self, x, ids):
        """
        Computes the loss for the given input and positive pairs' identifiers.

        Args:
            x (torch.Tensor): The input tensor for which the loss is computed.
            ids (torch.Tensor): A tensor containing identifiers for each input sample, used to determine positive/negative pairs.

        Returns:
            torch.Tensor: The computed loss value.
        """
        x = self.similarity(x)/self.theta
        x = self.sigma(x)
        ids = ids.unsqueeze(0)
        mask1 = (ids == ids.T)*(1 - torch.eye(np.max(ids.shape)).to(device))
        mask2 = ids != ids.T
        loss = -(self.a*mask1*torch.log(x) + (1 - self.a)*mask2*torch.log(1 - x))
        n = np.max(ids.shape)
        return loss.sum() * 2 / n / (n-1)


class HardSoftLoss(nn.Module):
    def __init__(self, theta = 0.5, pos=0.4, hard_neg=0.4, soft_person_neg=0.1, soft_video_neg=0.1, num_video=15):
        """
        Initializes the HardSoftLoss class with the specified parameters.

        Args:
            theta (float, optional): A scaling factor for the similarity matrix. Default is 0.5.
            pos (float, optional): Weight for positive samples in the loss calculation. Default is 0.4.
            hard_neg (float, optional): Weight for hard negative samples in the loss calculation. Default is 0.4.
            soft_person_neg (float, optional): Weight for person soft negative samples in the loss calculation. Default is 0.1.
            soft_video_neg (float, optional): Weight for video soft negative samples in the loss calculation. Default is 0.1.
        """
        super(HardSoftLoss, self).__init__()
        self.theta = theta
        self.pos=pos
        self.hard_neg=hard_neg
        self.soft_person_neg=soft_person_neg
        self.soft_video_neg = soft_video_neg
        self.num_video = num_video

    def similarity(self, x):
        """
        Computes the similarity matrix for the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A similarity matrix where each element represents the cosine similarity between pairs of input еутыщк.
        """
        x_norm = nn.functional.normalize(x, dim = 1)
        return x_norm @ x_norm.T

    def sigma(self, x):
        """
        Applies the sigmoid function element-wise to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The result of applying the sigmoid function to each element of the input tensor.
        """
        return nn.functional.sigmoid(x)

    def forward(self, x, ids):
        """
        Computes the loss based on the input features and their corresponding IDs.

        Args:
            x (torch.Tensor): The input tensor for which the loss is computed.
            ids (torch.Tensor): A tensor containing identifiers for each input sample, used to determine positive/negative pairs.

        Returns:
            torch.Tensor: The computed loss value as a scalar tensor.
        """
        x = self.similarity(x)/self.theta
        x = self.sigma(x)

        pid = ids // self.num_video
        vid = ids % self.num_video
        pid = pid.unsqueeze(0)
        vid = vid.unsqueeze(0)

        ids = ids.unsqueeze(0)
        mask_neg = 1*(ids != ids.T)
        mask_vid = (vid == vid.T)*mask_neg
        mask_pid = (pid == pid.T)*mask_neg

        mask_neg = mask_neg - mask_vid - mask_pid

        mask_pos = (ids == ids.T)*(1 - torch.eye(np.max(ids.shape)).to(device))

        loss = -self.pos*mask_pos*torch.log(x) - self.hard_neg*mask_neg*torch.log(1 - x) - \
                self.soft_person_neg*mask_pid*torch.log(1 - x) - self.soft_video_neg*mask_vid*torch.log(1 - x)
        n = np.max(ids.shape)
        return loss.sum() * 2 / n / (n-1)


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

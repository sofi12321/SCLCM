from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def train_one_epoch(model, device, train_loader, optimizer, criterion, task_type="embedding", scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        criterion (callable): The loss function used to compute the loss.
        task_type (str, optional): The type of task. Default is "embedding".
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
    """
    # Want to update weights of model
    model = model.to(device)
    model.train()
    # A wrapper over data loader to show progress bar
    bar = tqdm(train_loader, disable=True)
    iteration = 0
    overall_loss = 0
    targets = []
    outputs = []
    is_binary = False
    for samples in bar:
        # Take batch (batch, channels, time_len)
        data, target, patients = samples["data"], samples["label"], samples["patient"]
        # Device of data and model must be the same
        data, target, patients = data.to(device), target.to(device), patients.to(device)
        add_data = samples['add_data']

        # Avoid an accumulation of gradients
        optimizer.zero_grad()

        # Make prediction
        if add_data[0] == "":
            output = model(data)
        else:
            add_data = add_data.to(device)
            output = model(data, add_data)

        # Calculate loss between prediction and ground truth
        if task_type == "embedding":
            loss = criterion(output, patients)
            current_loss = loss.item()
            # * data.shape[0]
        else:
            # Flatten channels to make predictions across each channel
            if output.shape[-1] == 1:
                output = output.to('cpu')
                target = target.to('cpu')
                target = target.to(torch.float32)
                # print(target)
                is_binary = True
                output = output.reshape(output.shape[:-1])
            loss = criterion(output, target)
            current_loss = loss.item()

            outputs += output.cpu().detach().numpy().tolist()
            targets += target.to("cpu").numpy().tolist()

        # Compute gradient
        loss.backward()
        # Update params of model
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        iteration += 1
        overall_loss += current_loss
        if task_type == "embedding":
            # Pretrain
            bar.set_postfix({"Loss": format(overall_loss / iteration, '.6f')})
        elif is_binary:
            # Binary classification
            roc_auc = roc_auc_score(targets, outputs)
            auc = accuracy_score(targets, (torch.sigmoid(torch.tensor(outputs)) > 0.5).numpy().astype(int).reshape(-1))
            bar.set_postfix({"Loss": format(overall_loss / iteration, '.6f'),
                         "ROC_AUC": format(roc_auc, '.6f'),
                             "AUC": format(auc, '.6f')})
        else:
            # For multi-class
            output1 = np.argmax(outputs, axis=1)
            auc = accuracy_score(targets, output1)
            f1 = f1_score(targets, output1, average="macro")
            bar.set_postfix({"Loss": format(overall_loss / iteration, '.6f'),
                         "AUC": format(auc, '.6f'),
                         "F1": format(f1, '.6f')})
    print("Train loss:", format(overall_loss / iteration, '.6f'))

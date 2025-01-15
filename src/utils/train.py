from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def train(model, device, train_loader, optimizer, criterion, epoch, task_type="pretrain"):
    # Want to update weights of model
    model.train()
    # A wrapper over data loader to show progress bar
    bar = tqdm(train_loader)
    iteration = 0
    overall_loss = 0
    targets = []
    outputs = []
    for samples in bar:
        # Take batch (batch, channels, time_len)
        data, target, patients = samples["data"], samples["label"], samples["patient"]
        # Device of data and model must be the same
        data, target = data.to(device), target.to(device)
        # Avoid an accumulation of gradients
        optimizer.zero_grad()
        # Make prediction
        output = model(data)

        # Calculate loss between prediction and ground truth
        if task_type == "pretrain":
            loss = criterion(output, patients)
            current_loss = loss.item()
            # * data.shape[0]
        else:
            # Flatten channels to make predictions across each channel
            target = target.repeat(output.shape[1], 1).T
            target = torch.reshape(target, (-1,))
            output = torch.reshape(output, (-1, output.shape[-1]))

            loss = criterion(output, target)
            current_loss = loss.item()
            
            outputs += output.cpu().detach().numpy().tolist()
            targets += target.to("cpu").numpy().tolist()
            
        # Compute gradient
        loss.backward()
        # Update params of model
        optimizer.step()

        iteration += 1
        overall_loss += current_loss
        if task_type == "pretrain":
            # Pretrain
            bar.set_postfix({"Epoch": epoch,
                             "Loss": format(overall_loss / iteration, '.6f')})
        elif output.shape[-1] == 1:
            # Binary classification
            auc = roc_auc_score(targets, outputs)
            bar.set_postfix({"Epoch": epoch,
                         "Loss": format(overall_loss / iteration, '.6f'),
                         "ROC_AUC": format(auc, '.6f')})
        else:
            # For multi-class
            output1 = np.argmax(outputs, axis=1)
            auc = accuracy_score(targets, output1)
            f1 = f1_score(targets, output1)
            bar.set_postfix({"Epoch": epoch,
                         "Loss": format(overall_loss / iteration, '.6f'),
                         "AUC": format(auc, '.6f'),
                         "F1": format(f1, '.6f')})

import tqdm
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils.train import train_one_epoch

def test_classification(model, device, test_loader, criterion):
    """
    Evaluates the performance of a classification model on a test dataset.

    Args:
        model (torch.nn.Module): The classification model to be evaluated.
        device (torch.device): The device (CPU or GPU) on which the model and data should be loaded.
        test_loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.
        criterion (torch.nn.Module): Loss function used to compute the loss between the model output and target labels.

    Returns:
        A tuple containing:
            - float: The average test loss over the entire test dataset.
            - float: The AUC (Area Under the Curve) score for binary classification or accuracy score for multi-class classification.
    """
    model = model.to(device)
    model.eval()
    test_loss = 0
    bar = tqdm.tqdm(test_loader)
    score = 0
    targets = []
    outputs = []
    is_binary = False
    with torch.no_grad():
        for samples in bar:
            # Extract batched data (batch, channels, time_len)
            data, target, patients = samples["data"], samples["label"], samples["patient"]
            # data, target = data.to(device), target.to(device)

            # Device of data and model must be the same
            # print(data.shape, target.shape)
            data, target, patients = data.to(device), target.to(device), patients.to(device)
            # To avoid an accumulation of gradients
            add_data = samples['add_data']

            # Make prediction
            if add_data[0] == "":
                output = model(data)
            else:
                add_data = add_data.to(device)
                output = model(data, add_data)

            # Flatten channels to make predictions across each channel
            # target = target.repeat(output.shape[1], 1).T
            # target = torch.reshape(target, (-1,))
            # output = torch.reshape(output, (-1, output.shape[-1]))
            if output.shape[-1] == 1:
                target = target.to(torch.float32)
                is_binary = True
                output = output.reshape(output.shape[:-1])
            loss = criterion(output, target)

            outputs += output.to("cpu").detach().numpy().tolist()
            targets += target.to("cpu").numpy().tolist()
            # Sum up batch loss
            test_loss += loss.item()
            score += 1

            # Calculate metrics
            if is_binary:
                # Binary classification
                roc_auc = roc_auc_score(targets, outputs)
                auc = accuracy_score(targets, (torch.sigmoid(torch.tensor(outputs)) > 0.5).numpy().astype(int).reshape(-1))
                bar.set_postfix({"Loss": format(test_loss / score, '.6f'),
                                 "ROC_AUC": format(roc_auc, '.6f'),
                             "AUC": format(auc, '.6f')})
            else:
                # For multi-class
                output1 = np.argmax(outputs, axis=1)
                auc = accuracy_score(targets, output1)
                f1 = f1_score(targets, output1, average="macro")

                bar.set_postfix({"Loss": format(test_loss / score, '.6f'),
                                 "AUC": format(auc, '.6f'),
                                 "F1": format(f1, '.6f')})

    return test_loss / len(test_loader), auc


def run_classification(device,
                   final_model, posttrain_loader, posttest_loader, finetuning_optimizer, finetuning_criterion,
        finetuning_epochs, finetuning_wait_epochs, sensitivity_early_stop = 1.01, sensitivity_save_weights = 1.01, save_final_model_path=None,
                       pt_scheduler = None):
    """
    Performs fine-tuning of a classification model and evaluates its performance.

    Args:
        device (torch.device): The device on which the model and data should be loaded (e.g., 'cpu' or 'cuda').
        final_model (torch.nn.Module): The model to be fine-tuned.
        posttrain_loader (torch.utils.data.DataLoader): DataLoader for the training dataset used during fine-tuning.
        posttest_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset used for evaluation.
        finetuning_optimizer (torch.optim.Optimizer): Optimizer used for fine-tuning the model.
        finetuning_criterion (torch.nn.Module): Loss function used during fine-tuning.
        finetuning_epochs (int): Number of epochs to perform fine-tuning.
        finetuning_wait_epochs (int): Number of epochs to wait before early stopping if no improvement.
        sensitivity_early_stop (float, optional): Sensitivity factor for early stopping. Default is 1.01.
        sensitivity_save_weights (float, optional): Sensitivity factor for saving model weights. Default is 1.01.
        save_final_model_path (str, optional): Path to save the model weights if improvement is observed. Default is None.
        pt_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler for fine-tuning. Default is None.

    Returns:
        A tuple containing:
            - final_model (torch.nn.Module): The fine-tuned model.
            - finetuning_losses (list): List of loss values for each epoch during fine-tuning.
            - finetuning_acc (list): List of accuracy values for each epoch during fine-tuning.
    """
    # Fine-tuning
    finetuning_losses = []
    finetuning_acc = []
    final_model.to(device)
    best_loss = 10 ** 8
    best_iter = -1
    prev_loss = 10 ** 5
    scheduler = 0
    epoch = 1
    train_time = 0
    test_time = 0


    start_time = time.time()
    cur_loss, cur_acc = test_classification(final_model, device, posttest_loader, finetuning_criterion)
    test_time += time.time() - start_time
    finetuning_losses.append(cur_loss)
    finetuning_acc.append(cur_acc)
    best_loss = cur_loss
    prev_loss = cur_loss
    torch.save(final_model.state_dict(), save_final_model_path)

    for epoch in range(1, finetuning_epochs + 1):
        print("Epoch", epoch)
        start_time = time.time()
        train_one_epoch(final_model, device, posttrain_loader, finetuning_optimizer, finetuning_criterion,
              task_type="finetuning", scheduler = pt_scheduler)
        train_time += time.time() - start_time

        start_time = time.time()
        cur_loss, cur_acc = test_classification(final_model, device, posttest_loader, finetuning_criterion)
        test_time += time.time() - start_time

        finetuning_losses.append(cur_loss)
        finetuning_acc.append(cur_acc)

        # Check upgrade and save weights if needed
        if cur_loss * sensitivity_save_weights < best_loss:
            best_loss = cur_loss
            best_iter = epoch
            if save_final_model_path:
                torch.save(final_model.state_dict(), save_final_model_path)
                print("The model weights were saved")
        # Early stop
        if cur_loss * sensitivity_early_stop < prev_loss:
            prev_loss = cur_loss
            scheduler = 0
        else:
            scheduler += 1

        if scheduler >= finetuning_wait_epochs:
            # Overfitting, so early stop
            print("The finetuning starts to overfit, so stop the process")
            print("The best loss is", round(best_loss, 5), "at epoch", best_iter)
            break

    train_time = train_time / epoch
    test_time = test_time / (epoch + 1)
    return final_model, finetuning_losses, finetuning_acc, train_time, test_time

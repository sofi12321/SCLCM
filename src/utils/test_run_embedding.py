from tqdm import tqdm
import torch
import numpy as np

def test_embedding(model, device, test_loader, criterion):
    """
    Evaluates the model on a test dataset and computes the average loss.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device on which the model and data should be loaded (e.g., 'cpu' or 'cuda').
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (callable): The loss function used to compute the loss between the model output and the target.

    Returns:
        float: The average loss over the test dataset.
    """
    # Do not train
    model = model.to(device)
    model.eval()
    test_loss = 0
    bar = tqdm(test_loader, disable=False)
    ind = 0
    with torch.no_grad():
        for samples in bar:
            # Extract batched data (batch, channels, time_len)
            data, target, patients = samples["data"], samples["label"], samples["patient"]

            # Device of data and model must be the same
            data, target, patients = data.to(device), target.to(device), patients.to(device)
            # data, target = data.to(device), target.to(device)
            add_data = samples['add_data']

            # Make prediction
            if add_data[0] == "":
                output = model(data)
            else:
                add_data = add_data.to(device)
                output = model(data, add_data)
            # Sum up batch loss
            loss = criterion(output, patients).item()

            test_loss += loss
            #  * data.shape[0]
            ind += 1
            bar.set_postfix({"Loss": format(test_loss / ind, '.6f')})

    print(f"Test set: Average loss: {test_loss / len(test_loader)}")
    # print()
    return test_loss / len(test_loader)


def run_embedding(device, save_pretrain_path,
                  pretrain_model, pretrain_loader, preval_loader,
                  pretrain_optimizer, pretrain_criterion,
                  pretrain_epochs, pretrain_wait_epochs,
                  sensitivity_save_weights = 1.01, sensitivity_early_stop = 1.01,
                  pt_scheduler = None
                  ):
    """
    Executes the embedding training process for a specified number of epochs, with early stopping and model saving
    based on validation loss.

    Args:
        device (torch.device): The device on which the model and data should be loaded (e.g., 'cpu' or 'cuda').
        save_pretrain_path (str): The file path where the best performing model's weights will be saved.
        pretrain_model (torch.nn.Module): The model to be trained.
        pretrain_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        preval_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        pretrain_optimizer (torch.optim.Optimizer): Optimizer used for training the model.
        pretrain_criterion (torch.nn.Module): Loss function used to evaluate the model's performance.
        pretrain_epochs (int): The maximum number of epochs to train the model.
        pretrain_wait_epochs (int): The number of epochs to wait before early stopping if no improvement is seen.
        sensitivity_save_weights (float, optional): Sensitivity factor for saving model weights. Default is 1.01.
        sensitivity_early_stop (float, optional): Sensitivity factor for early stopping. Default is 1.01.
        pt_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.

    Returns:
        A tuple containing:
            - pretrain_model (torch.nn.Module): The trained model.
            - pretrain_losses (list): A list of validation losses recorded at each epoch.
    """
    pretrain_losses = []
    best_loss = 10 ** 8
    best_iter = -1
    prev_loss = 10 ** 5
    scheduler = 0
    epoch = 1
    train_time = 0
    test_time = 0

    start_time = time.time()
    cur_loss = test_embedding(pretrain_model, device, preval_loader, pretrain_criterion)
    test_time += time.time() - start_time

    pretrain_losses.append(cur_loss)
    best_loss = cur_loss
    prev_loss = cur_loss
    torch.save(pretrain_model.state_dict(), save_pretrain_path)

    for epoch in range(1, pretrain_epochs + 1):
        # Run one epoch
        print("Epoch", epoch)
        start_time = time.time()
        train_one_epoch(pretrain_model, device, pretrain_loader, pretrain_optimizer, pretrain_criterion,
                        task_type="embedding", scheduler=pt_scheduler)
        train_time += time.time() - start_time

        start_time = time.time()
        cur_loss = test_embedding(pretrain_model, device, preval_loader, pretrain_criterion)
        test_time += time.time() - start_time
        pretrain_losses.append(cur_loss)

        # if cur_loss < best_loss:
            # print("save_w", cur_loss, best_loss)
            # print("prop_", best_loss / cur_loss)
            # print("perc_", best_loss - cur_loss * sensitivity_save_weights, cur_loss * sensitivity_save_weights)

        # Save weights of best performing model
        if cur_loss * sensitivity_save_weights < best_loss:
            best_loss = cur_loss
            best_iter = epoch
            torch.save(pretrain_model.state_dict(), save_pretrain_path)
            print("The model weights were saved")

        # Early stop
        if cur_loss * sensitivity_early_stop < prev_loss:
            prev_loss = cur_loss
            scheduler = 0
        else:
            scheduler += 1

        if scheduler >= pretrain_wait_epochs:
            # Overfitting, so early stop
            print("The pretrain starts to overfit, so stop the pretraining process")
            print("The best loss obtained is", round(best_loss, 5), "at epoch", best_iter)
            break

    train_time = train_time / epoch
    test_time = test_time / (epoch + 1)
    return pretrain_model, pretrain_losses, train_time, test_time

from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def test_embedding(model, device, test_loader, criterion):
    """
    Evaluation of pretrained model for embedding extraction
    :param model: Model to be evaluated
    :param device: location, cpu or cuda
    :param test_loader: dataset
    :param criterion: loss evaluator
    :return: evaluated loss
    """
    # Do not train
    model.eval()
    test_loss = 0
    bar = tqdm(test_loader)
    ind = 0
    with torch.no_grad():
        for samples in bar:
            # Extract batched data (batch, channels, time_len)
            data, target, patients = samples["data"], samples["label"], samples["patient"]

            # Device of data and model must be the same
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Sum up batch loss
            loss = criterion(output, patients).item()

            test_loss += loss
            #  * data.shape[0]
            ind += 1
            bar.set_postfix({"Loss": format(test_loss / ind, '.6f')})

    print(f"Test set: Average loss: {test_loss / len(test_loader)}")
    print()
    return test_loss / len(test_loader)

def run_embedding(device, save_pretrain_path,
                  pretrain_model, pretrain_loader, preval_loader, 
                  pretrain_optimizer, pretrain_criterion,
                  pretrain_epochs, pretrain_wait_epochs, 
                  sensitivity_save_weights = 1.01, sensitivity_early_stop = 1.01
                  ):
    pretrain_losses = []
    best_loss = 10 ** 8
    best_iter = -1
    prev_loss = 10 ** 5
    scheduler = 0

    for epoch in range(1, pretrain_epochs + 1):
        # Run one epoch
        print("Epoch", epoch)
        train_one_epoch(pretrain_model, device, pretrain_loader, pretrain_optimizer, pretrain_criterion,
                        task_type="embedding")
        cur_loss = test_embedding(pretrain_model, device, preval_loader, pretrain_criterion)
        pretrain_losses.append(cur_loss)

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
    return pretrain_model, pretrain_losses

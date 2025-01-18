from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def test_classification(model, device, test_loader, criterion):
    """
    Evaluation of model for classification
    :param model: Model to be evaluated
    :param device: location, cpu or cuda
    :param test_loader: dataset
    :param criterion: loss evaluator
    :return: evaluated loss and prediction accuracy
    """
    model.eval()
    test_loss = 0
    bar = tqdm(test_loader)
    score = 0
    targets = []
    outputs = []
    with torch.no_grad():
        for samples in bar:
            # Extract batched data (batch, channels, time_len)
            data, target, patients = samples["data"], samples["label"], samples["patient"]

            # Device of data and model must be the same
            # print(data.shape, target.shape)
            data, target = data.to(device), target.to(device)
            # To avoid an accumulation of gradients
            output = model(data)

            # Flatten channels to make predictions across each channel
            target = target.repeat(output.shape[1], 1).T
            target = torch.reshape(target, (-1,))
            output = torch.reshape(output, (-1, output.shape[-1]))

            loss = criterion(output, target)

            outputs += output.to("cpu").detach().numpy().tolist()
            targets += target.to("cpu").numpy().tolist()
            # Sum up batch loss
            test_loss += loss.item()
            score += 1

            # Calculate metrics
            if output.shape[-1] == 1:
                # Binary classification
                auc = roc_auc_score(targets, outputs)
                bar.set_postfix({"Loss": format(test_loss / len(test_loader), '.6f'),
                                 "ROC_AUC": format(auc, '.6f')})
            else:
                # For multi-class
                output1 = np.argmax(outputs, axis=1)
                auc = accuracy_score(targets, output1)
                f1 = f1_score(targets, output1, average="macro")

                bar.set_postfix({"Loss": format(test_loss / len(test_loader), '.6f'),
                                 "AUC": format(auc, '.6f'),
                                 "F1": format(f1, '.6f')})

    return test_loss / len(test_loader), auc
  

def run_classification(device,
                   final_model, posttrain_loader, posttest_loader, finetuning_optimizer, finetuning_criterion,
        finetuning_epochs, finetuning_wait_epochs, sensitivity_early_stop = 1.01, sensitivity_save_weights = 1.01, save_final_model_path=None):
    # Fine-tuning
    finetuning_losses = []
    finetuning_acc = []
    final_model.to(device)
    best_loss = 10 ** 8
    best_iter = -1
    prev_loss = 10 ** 5
    scheduler = 0
    for epoch in range(1, finetuning_epochs + 1):
        print("Epoch", epoch)
        train_one_epoch(final_model, device, posttrain_loader, finetuning_optimizer, finetuning_criterion,
              task_type="finetuning")
        cur_loss, cur_acc = test_classification(final_model, device, posttest_loader, finetuning_criterion)
        finetuning_losses.append(cur_loss)
        finetuning_acc.append(cur_acc)
        
        # Check upgrade and save weights if needed
        if cur_loss * sensitivity_save_weights < best_loss:
            best_loss = cur_loss
            best_iter = epoch
            if save_final_model_path:
                torch.save(final_model.state_dict(), save_final_model_path)
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

    return final_model, finetuning_losses, finetuning_acc

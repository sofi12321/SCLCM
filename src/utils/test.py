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
                f1 = f1_score(targets, output1)

                bar.set_postfix({"Loss": format(test_loss / len(test_loader), '.6f'),
                                 "AUC": format(auc, '.6f'),
                                 "F1": format(f1, '.6f')})

    return test_loss / len(test_loader), auc

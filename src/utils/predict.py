import numpy as np
import torch 
from sklearn.metrics import classification_report

def get_predictions(model, test_loader, device='cpu'):
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        X = []
        y = []
        pids = []
        # pred = []
        for el in test_loader:
            X += model(el['data'].to(device)).cpu().tolist()
            y += el['label'].tolist()

    X = np.array(X)
    y = np.array(y)
    return X, y

def print_classification_report(model, test_loader, num_classes, device='cpu')
        
    if num_classes == 2:
        XX, yy = get_predictions(ft_model, test_loader_ft, device=device)
        preds = (XX > 0.5).astype(int)
        print(classification_report(yy, preds, target_names=['negative','positive']))

    elif num_classes == 3:
        XX, yy = get_predictions(ft_model, test_loader_ft, device=device)
        preds = np.argmax(XX, axis=1)
        print(classification_report(yy, preds, target_names=['negative','neutral', 'positive']))
    elif num_classes > 3:
        XX, yy = get_predictions(ft_model, test_loader_ft, device=device)
        preds = np.argmax(XX, axis=1)
        print(classification_report(yy, preds, target_names=(np.array(range(num_classes))).astype(str))
    else:
        print(f"Unfortunately, the number of classes {num_classes} is not appropriate for the classification report")

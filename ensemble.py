import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader
import csv

NUM_WORKERS = 0

COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', ]
VAL_CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

def ensemble(models, X):
    predictions = []
    for model, is_trans in models:
        if is_trans:
            X= X.permute(0, 2, 1)

        prediction = model(X)
        predictions.append(prediction)
    ensemble_prediction = sum(predictions) / len(models)
    return ensemble_prediction

def load_models(model_dict, device):
    models = []
    for instance in model_dict:
        model = instance['model']
        path = instance['path']
        is_trans = instance['is_trans']

        model.to(device)
        model = model.cuda()

        model.load_state_dict(torch.load(path))
        model.eval()
        models.append((model, is_trans))
    return models

def evaluate_ensemble(models, dataset, device, all_classes, long_mode = True):
    dataloader = DataLoader(dataset, batch_size = 100, num_workers = NUM_WORKERS)
    for model, is_trans in models:
        model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            if device == torch.device('cuda'):
                images = images.cuda()
                labels = labels.cuda()

            outputs = ensemble(models,images)
            
            images = images.cpu()
            labels = labels.cpu()
            outputs = outputs.cpu()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

        # converting classes
        for key, value in all_classes.items():
            if value not in VAL_CLASSES:
                all_classes[key] = "unknown"

        y_true = np.array([all_classes[round(num)] for num in y_true])
        y_pred = np.array([all_classes[round(num)] for num in y_pred])

        y_true = [CLASSES.index(name) for name in y_true]
        y_pred = [CLASSES.index(name) for name in y_pred]

        target_names = [CLASSES[cls] for cls in range(12)]

    if long_mode:
        print(classification_report(y_true = y_true, 
                                    y_pred = y_pred, 
                                    target_names = target_names,
                                    digits=4))

        cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
            
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = target_names)
        disp.plot()
    else:
        acc = accuracy_score(y_true = y_true, y_pred = y_pred)
        print(f"Test accuracy: {acc}")
        
    return y_pred, y_true


def evaluate_ensemble_test(models, dataset, device, all_classes):
    dataloader = DataLoader(dataset, batch_size = 100, num_workers = NUM_WORKERS)
    for model, is_trans in models:
        model.eval()
    y_true = []
    y_pred = []
    # converting classes
    for key, value in all_classes.items():
        if value not in VAL_CLASSES:
            all_classes[key] = "unknown"
            
    with open('outputs/ensamble_output.csv', 'w', newline='') as csvfile:
        
        fieldnames = ['fname', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() 
        with torch.no_grad():
            for inputs, file_names in dataloader:
                if device == torch.device('cuda'):
                    inputs = inputs.cuda()

                outputs = ensemble(models,torch.tensor(inputs).float().transpose(2, 1))

                outputs = outputs.cpu()

                _, predicted = torch.max(outputs, 1)

                predicted = predicted.numpy()
                
                for num,file_name in zip(predicted,file_names):
                    predicted_word = all_classes[num]
                    y_pred.append(predicted_word)
                    writer.writerow({'fname': file_name, 'label': predicted_word})
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import torch
from torch.utils.data import DataLoader

RANDOM_SEED = 17
NUM_WORKERS = 0

COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', ]
VAL_CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def perform_test(model, train_data, test_data, num_epochs, criterion, optimizer, batch_size, all_classes, device, scheduler = None, should_print = True, random_seed=RANDOM_SEED, trans_mode = False):
    generator = torch.Generator().manual_seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    dataloader_train = DataLoader(train_data, batch_size = batch_size, shuffle = True, generator = generator, num_workers = NUM_WORKERS)

    model, total_loss_all, accs = run_network(model = model, 
                                        optimizer = optimizer, 
                                        criterion = criterion, 
                                        dataloader = dataloader_train,
                                        should_print = should_print,
                                        num_epochs = num_epochs,
                                        all_classes = all_classes,
                                        device = device,
                                        valid_data = test_data,
                                        scheduler = scheduler,
                                        trans_mode = trans_mode)
    
    # evaluating model
    if should_print:
        print_err(total_loss_all)
        
        print("Test eval:")
        _, _, acc = evaluate(model = model, dataset = test_data, device = device, all_classes = all_classes, trans_mode = trans_mode)
        accs.append(acc)

        plot_accs(accs)
        
        
    return model,

def run_network(model, criterion, optimizer, dataloader, num_epochs, device, scheduler, valid_data, all_classes, trans_mode, should_print = True, test_frequency = 1):
    total_loss = []
    accs = []
    model.train()

    for epoch in range(num_epochs):
        model.train()
        loss = 0
    
        for inputs, labels in dataloader:
            if(trans_mode):
                inputs = inputs.permute(0, 2, 1)

            if device == torch.device('cuda'):
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            loss += loss.item()
    
        if (scheduler is not None):
            scheduler.step()
        
        # stats per epoch
        total_loss.append(loss)
        if should_print:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {loss / len(dataloader):.4f}")

            if (epoch + 1) % test_frequency == 0 and (epoch + 1) < num_epochs:
                _, _, acc = evaluate(model = model, dataset = valid_data, device = device, long_mode = False, all_classes = all_classes, trans_mode = trans_mode)
                accs.append(acc)
                print(f"Test accuracy: {acc}")
                
        
    return model, total_loss, accs

def print_err(total_loss):
    total_loss_c = []
    for l in total_loss:
        l = l.cpu()
        l = l.detach().numpy()
        total_loss_c.append(l)

    plt.figure(figsize=(8, 6))
    plt.plot(range(0, len(total_loss_c)), total_loss_c) 
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.title('error in each epoch')
    
def evaluate(model, dataset, device, all_classes, trans_mode, long_mode = True):
    dataloader = DataLoader(dataset, batch_size = 100, num_workers = NUM_WORKERS)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            if(trans_mode):
                inputs = inputs.permute(0, 2, 1)

            if device == torch.device('cuda'):
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)

            inputs = inputs.cpu()
            labels = labels.cpu()
            outputs = outputs.cpu()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    # converting classes
    # class_dict = dict(all_classes)
    # y_true = np.array([class_dict[num] if class_dict.get(num) in VAL_CLASSES else "unknown" for num in y_true])
    # y_pred = np.array([class_dict[num] if class_dict.get(num) in VAL_CLASSES else "unknown" for num in y_pred])

    # class_to_number = {class_name: i for i, class_name in enumerate(CLASSES)}

    # y_true = np.array([class_to_number[class_dict[num]] if class_dict.get(num) in CLASSES else -1 for num in y_true])
    # y_pred = np.array([class_to_number[class_dict[num]] if class_dict.get(num) in CLASSES else -1 for num in y_pred])

    #target_names = [CLASSES[cls] for cls in range(12)]

    acc = accuracy_score(y_true = y_true, y_pred = y_pred)
    if long_mode:
        print(classification_report(y_true = y_true, 
                                    y_pred = y_pred, 
                                    #target_names = target_names,
                                    digits=4))

        cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
            
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, #display_labels = target_names
                                      )
        disp.plot()  
        
    return y_pred, y_true, acc

def plot_accs(accs):
    epochs = range(1, len(accs) + 1)

    plt.plot(epochs, accs, 'bo')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import torch
from torch.utils.data import DataLoader
from augmentation import mixup_data, mixup_criterion

RANDOM_SEED = 17
NUM_WORKERS = 0

COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', ]
VAL_CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def perfrom_test(model, train_data, test_data, num_epochs, criterion, optimizer, batch_size, all_classes, device, scheduler = None, should_print = True, random_seed=RANDOM_SEED):
    generator = torch.Generator().manual_seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    dataloader_train = DataLoader(train_data, batch_size = batch_size, shuffle = True, generator = generator, num_workers = NUM_WORKERS)

    model, total_loss_all = run_network(model = model, 
                                        optimizer = optimizer, 
                                        criterion = criterion, 
                                        dataloader = dataloader_train,
                                        should_print = should_print,
                                        num_epochs = num_epochs,
                                        all_classes = all_classes,
                                        device = device,
                                        valid_data = test_data,
                                        scheduler = scheduler)
    
    # evaluating model
    if should_print:
        print_err(total_loss_all)
        
        print("Train eval:")
        evaluate(model = model, dataset = train_data, device = device, all_classes = all_classes)
        
        print("Test eval:")
        evaluate(model = model, dataset = test_data, device = device, all_classes = all_classes)
        
        
    return model   

def run_network(model, criterion, optimizer, dataloader, num_epochs, device, scheduler, valid_data, all_classes, should_print = True, test_frequency = 10):
    total_loss = []
    model.train()

    for epoch in range(num_epochs):
        model.train()
        loss = 0
    
        for images, labels in dataloader:
            if device == torch.device('cuda'):
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            loss += loss.item()
    
        if (scheduler is not None):
            scheduler.step()
        
        total_loss.append(loss)
        if should_print:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss / len(dataloader):.4f}, Lr: {lr}")

            if (epoch + 1) % test_frequency == 0 and (epoch + 1) < num_epochs:
                evaluate(model = model, dataset = valid_data, device = device, long_mode = False, all_classes = all_classes)
        
    return model, total_loss

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
    
def evaluate(model, dataset, device, all_classes, long_mode = True):
    dataloader = DataLoader(dataset, batch_size = 100, num_workers = NUM_WORKERS)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            if device == torch.device('cuda'):
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            
            images = images.cpu()
            labels = labels.cpu()
            outputs = outputs.cpu()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    # converting classes
    class_dict = dict(all_classes)
    y_true = np.array([class_dict[num] if class_dict.get(num) in VAL_CLASSES else "unknown" for num in y_true])
    y_pred = np.array([class_dict[num] if class_dict.get(num) in VAL_CLASSES else "unknown" for num in y_pred])

    class_to_number = {class_name: i for i, class_name in enumerate(CLASSES)}

    y_true = np.array([class_to_number[class_dict[num]] if class_dict.get(num) in CLASSES else -1 for num in y_true])
    y_pred = np.array([class_to_number[class_dict[num]] if class_dict.get(num) in CLASSES else -1 for num in y_pred])

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


# # compute mean for dataset
# def compute_mean(dataset):
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = NUM_WORKERS)
#     mean = torch.zeros(3)
#     for image, _ in dataloader:
#         for i in range(3):
#             mean[i] += image[:, i, :, :].mean()
#     mean.div_(len(dataset))
#     return mean

# # compute std for dataset
# def compute_std(dataset):
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = NUM_WORKERS)
#     std = torch.zeros(3)
#     for image, _ in dataloader:
#         for i in range(3):
#             std[i] += image[:, i, :, :].std()
#     std.div_(len(dataset))
#     return std


# from torchvision import transforms
# from torchvision.datasets import DatasetFolder

# def compute_norm_stats(train_dir, valid_dir, ratio):
#     transform = transforms.Compose([
#         transforms.ToTensor(), 
#     ])

#     # adding train and valid data to one dataset and them splitting them again with a new ratio
#     dataset1 = DatasetFolder(root = train_dir, 
#                             loader = image_loader, 
#                             transform = transform,
#                             extensions='.png')

#     dataset2 = DatasetFolder(root = valid_dir, 
#                             loader = image_loader, 
#                             transform = transform,
#                             extensions='.png')

#     train_valid_datasets = torch.utils.data.ConcatDataset([dataset1, dataset2])

#     generator = torch.Generator().manual_seed(RANDOM_SEED)
        
#     # splitting data
#     train_size = int(ratio * len(train_valid_datasets))
#     valid_size = len(train_valid_datasets) - train_size
#     pre_train_data, _ = torch.utils.data.random_split(train_valid_datasets, [train_size, valid_size], generator = generator)

#     std_train = compute_std(pre_train_data)
#     mean_train = compute_mean(pre_train_data)

#     return mean_train, std_train
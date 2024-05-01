import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import torch
from torch.utils.data import DataLoader
from augmentation import mixup_data, mixup_criterion
from torch.optim.lr_scheduler import CyclicLR

RANDOM_SEED = 17
NUM_WORKERS = 0

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def perfrom_test(model, train_data, test_data, num_epochs, criterion, optimizer, batch_size, device, scheduler = None, mixup_alpha=0, should_print = True, random_seed=RANDOM_SEED):
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
                                        mixup_alpha = mixup_alpha,
                                        device = device,
                                        valid_data = test_data,
                                        scheduler = scheduler)
    
    # evaluating model
    if should_print:
        print_err(total_loss_all)
        
        print("Train eval:")
        evaluate(model = model, dataset = train_data, device = device)
        
        print("Test eval:")
        evaluate(model = model, dataset = test_data, device = device)
        
        
    return model   

def run_network(model, criterion, optimizer, dataloader, num_epochs, device, scheduler, valid_data, mixup_alpha = 0, should_print = True, test_frequency = 10):
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
            
            if mixup_alpha == 0:
                outputs = model(images)
                loss = criterion(outputs, labels)
            else:            
                mixed_data, target_a, target_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(mixed_data)            
                loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)

            loss.backward()
            optimizer.step()
            loss += loss.item()
        
            if (scheduler is not None) and isinstance(scheduler, CyclicLR): # schedulers step evey batch or every epoch depending on type
                scheduler.step()
    
        if (scheduler is not None) and (not isinstance(scheduler, CyclicLR)):
            scheduler.step()
        
        total_loss.append(loss)
        if should_print:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss / len(dataloader):.4f}, Lr: {lr}")

            if (epoch + 1) % test_frequency == 0 and (epoch + 1) < num_epochs:
                evaluate(model = model, dataset = valid_data, device = device, long_mode = False)
        
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
    
def evaluate(model, dataset, device, long_mode = True):
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

    target_names = [CLASSES[cls] for cls in range(10)]

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

def image_loader(img_path):
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

# compute mean for dataset
def compute_mean(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = NUM_WORKERS)
    mean = torch.zeros(3)
    for image, _ in dataloader:
        for i in range(3):
            mean[i] += image[:, i, :, :].mean()
    mean.div_(len(dataset))
    return mean

# compute std for dataset
def compute_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = NUM_WORKERS)
    std = torch.zeros(3)
    for image, _ in dataloader:
        for i in range(3):
            std[i] += image[:, i, :, :].std()
    std.div_(len(dataset))
    return std


from torchvision import transforms
from torchvision.datasets import DatasetFolder

def compute_norm_stats(train_dir, valid_dir, ratio):
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    # adding train and valid data to one dataset and them splitting them again with a new ratio
    dataset1 = DatasetFolder(root = train_dir, 
                            loader = image_loader, 
                            transform = transform,
                            extensions='.png')

    dataset2 = DatasetFolder(root = valid_dir, 
                            loader = image_loader, 
                            transform = transform,
                            extensions='.png')

    train_valid_datasets = torch.utils.data.ConcatDataset([dataset1, dataset2])

    generator = torch.Generator().manual_seed(RANDOM_SEED)
        
    # splitting data
    train_size = int(ratio * len(train_valid_datasets))
    valid_size = len(train_valid_datasets) - train_size
    pre_train_data, _ = torch.utils.data.random_split(train_valid_datasets, [train_size, valid_size], generator = generator)

    std_train = compute_std(pre_train_data)
    mean_train = compute_mean(pre_train_data)

    return mean_train, std_train
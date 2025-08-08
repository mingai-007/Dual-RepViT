import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

from models.model import Dual_RepViT
from dataloader.dataloader import TBDataLoader


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    label_list = []
    prob_list = []
    total_loss = 0.0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)

        outputs = outputs.squeeze(dim=1)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


        label_list.append(labels.cpu().numpy())
        prob_list.append(torch.sigmoid(outputs).cpu().detach().numpy())

    label_array = np.concatenate(label_list)
    prob_array = np.concatenate(prob_list)


    pred_array = (prob_array > 0.5).astype(int)

    accuracy = accuracy_score(label_array, pred_array)
    f1 = f1_score(label_array, pred_array, average='binary')
    precision = precision_score(label_array, pred_array, average='binary')
    recall = recall_score(label_array, pred_array, average='binary')

    tn = np.sum((label_array == 0) & (pred_array == 0))
    fp = np.sum((label_array == 0) & (pred_array == 1))
    specificity = tn / (tn + fp + 1e-10)

    return total_loss / len(dataloader), accuracy, f1, precision, recall, specificity


def validate(model, dataloader, criterion, device):
    model.eval()
    label_list = []
    prob_list = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.float().to(device)


            outputs = model(images)

            outputs = outputs.squeeze(dim=1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            label_list.append(labels.cpu().numpy())
            prob_list.append(torch.sigmoid(outputs).cpu().numpy())

    label_array = np.concatenate(label_list)
    prob_array = np.concatenate(prob_list)

    pred_array = (prob_array > 0.5).astype(int)

    accuracy = accuracy_score(label_array, pred_array)
    f1 = f1_score(label_array, pred_array, average='binary')
    precision = precision_score(label_array, pred_array, average='binary')
    recall = recall_score(label_array, pred_array, average='binary')

    tn = np.sum((label_array == 0) & (pred_array == 0))
    fp = np.sum((label_array == 0) & (pred_array == 1))
    specificity = tn / (tn + fp + 1e-10)

    return total_loss / len(dataloader), accuracy, f1, precision, recall, specificity


if __name__ == '__main__':

    image_size = 256
    batch_size = 8
    image_path = r'E:\TB_project\datasets\TBX11k_enhance'
    data_loader = TBDataLoader(root_dir=image_path,image_size=image_size, batch_size=batch_size)
    train_loader, val_loader, class_map = data_loader.get_loaders()
    print('class_map: ', class_map)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Dual_RepViT(num_classes=1)

    sum_parameters = sum(p.numel() for p in model.parameters())

    lr = 0.0001
    weight_decay = 0.0001
    epochs = 20

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    model.to(device)


    train_losses = []
    train_accuracies = []
    train_f1es = []
    train_precisiones = []
    train_recalles = []
    train_specificities = []

    val_losses = []
    val_accuracies = []
    val_f1es = []
    val_precisiones = []
    val_recalles = []
    val_specificities = []

    best_f1 = 0
    train_data_dir = os.path.join(os.path.dirname(__file__), "train_data")
    os.makedirs(train_data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(train_data_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc, train_f1, train_precision, train_recall,train_spec = train_one_epoch(model, train_loader, optimizer,
                                                                                         criterion, device)
        val_loss, val_acc, val_f1, val_precision, val_recall, val_spec = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1es.append(train_f1)
        train_precisiones.append(train_precision)
        train_recalles.append(train_recall)
        train_specificities.append(train_spec)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1es.append(val_f1)
        val_precisiones.append(val_precision)
        val_recalles.append(val_recall)
        val_specificities.append(val_spec)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},train f1: {train_f1:.4f},train precision: {train_precision:.4f}, train_recall: {train_recall:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f},val f1: {val_f1:.4f},val precision: {val_precision:.4f},val recall: {val_recall:.4f}")
        # 保存模型权重
        if val_f1 >= best_f1:
            best_f1 = val_f1
            print('find best model in epoch %d with F score %.4f' % (epoch + 1, best_f1))
            torch.save(model, os.path.join(output_dir, f'model_epoch_{epoch + 1}.pth'))


    plt.figure(figsize=(25, 10))


    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')

    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        plt.text(i, train_loss, f'{train_loss:.4f}', fontsize=8, ha='center', va='bottom')
        plt.text(i, val_loss, f'{val_loss:.4f}', fontsize=8, ha='center', va='top')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()


    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')

    for i, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
        plt.text(i, train_acc, f'{train_acc:.4f}', fontsize=8, ha='center', va='bottom')
        plt.text(i, val_acc, f'{val_acc:.4f}', fontsize=8, ha='center', va='top')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()


    plt.subplot(2, 3, 3)
    plt.plot(train_f1es, label='Train F1 Score')
    plt.plot(val_f1es, label='Validation F1 Score')

    for i, (train_f1, val_f1) in enumerate(zip(train_f1es, val_f1es)):
        plt.text(i, train_f1, f'{train_f1:.4f}', fontsize=8, ha='center', va='bottom')
        plt.text(i, val_f1, f'{val_f1:.4f}', fontsize=8, ha='center', va='top')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()


    plt.subplot(2, 3, 4)
    plt.plot(train_precisiones, label='Train Precision')
    plt.plot(val_precisiones, label='Validation Precision')

    for i, (train_precision, val_precision) in enumerate(zip(train_precisiones, val_precisiones)):
        plt.text(i, train_precision, f'{train_precision:.4f}', fontsize=8, ha='center',
                 va='bottom')  # 标出训练集 precision 值
        plt.text(i, val_precision, f'{val_precision:.4f}', fontsize=8, ha='center', va='top')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.legend()


    plt.subplot(2, 3, 5)
    plt.plot(train_recalles, label='Train Recall')
    plt.plot(val_recalles, label='Validation Recall')


    for i, (train_recall, val_recall) in enumerate(zip(train_recalles, val_recalles)):
        plt.text(i, train_recall, f'{train_recall:.4f}', fontsize=8, ha='center', va='bottom')
        plt.text(i, val_recall, f'{val_recall:.4f}', fontsize=8, ha='center', va='top')

    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()


    plt.subplot(2, 3, 6)
    plt.plot(train_specificities, label='Train Specificity')
    plt.plot(val_specificities, label='Validation Specificity')
    for i, (train_spe, val_spe) in enumerate(zip(train_specificities, val_specificities)):
        plt.text(i, train_spe, f'{train_spe:.4f}', fontsize=8, ha='center', va='bottom')
        plt.text(i, val_spe, f'{val_spe:.4f}', fontsize=8, ha='center', va='top')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.title('Specificity Curve')
    plt.legend()



    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'training_plot.png')
    plt.savefig(plot_path)
    plt.show()


    arg_filename='arg.txt'
    with open(os.path.join(output_dir,arg_filename),'w') as file:
        file.write(str(class_map )+'\n')
        file.write('image_data_path:{}\n'.format(image_path))

        file.write('batch_size:{}\n'.format(batch_size))
        file.write('lr:{}\n'.format(lr))
        file.write('weight_decay:{}\n'.format(weight_decay))
        file.write('image_size:{}\n'.format(image_size))
        file.write('sum_parameters:{}\n'.format(sum_parameters))


        file.write('val_f1es:')
        file.write(','.join(map(str, val_f1es)) + '\n')

        file.write('val_accuracies:')
        file.write(','.join(map(str, val_accuracies)) + '\n')

        file.write('val_precisiones:')
        file.write(','.join(map(str, val_precisiones)) + '\n')

        file.write('val_recalles:')
        file.write(','.join(map(str, val_recalles)) + '\n')

        file.write('val_specificities:')
        file.write(','.join(map(str, val_specificities)) + '\n')

        file.write('val_losses:')
        file.write(','.join(map(str, val_losses)) + '\n')


        file.write('train_f1es:')
        file.write(','.join(map(str, train_f1es)) + '\n')

        file.write('train_accuracies:')
        file.write(','.join(map(str, train_accuracies)) + '\n')

        file.write('train_precisiones:')
        file.write(','.join(map(str, train_precisiones)) + '\n')

        file.write('train_recalles:')
        file.write(','.join(map(str, train_recalles)) + '\n')

        file.write('train_specificities:')
        file.write(','.join(map(str, train_specificities)) + '\n')

        file.write('train_losses:')
        file.write(','.join(map(str, train_losses)) + '\n')

        file.write("Model structure:\n")
        file.write(str(model) + "\n")

    print(f"File created and content written: {os.path.join(output_dir, arg_filename)}")


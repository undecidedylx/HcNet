# -- coding: utf-8 --
import torch.cuda
import os
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ToPILImage
from model.HcNet import HcNet
import parameter as paras
from tqdm import tqdm
import sys
import csv


def train():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"HcNet training--------device is {device}----------------")
    nw = min(os.cpu_count(), paras.batch_size if paras.batch_size > 1 else 0, 8)

    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomRotation(degrees=90),
        transforms.Resize((256,256))
    ])

    print(os.path.join(paras.last_data_dir, 'train'))
    train_dataset = Dataset(nii_dir=os.path.join(paras.last_data_dir, 'train'), transform=transform)
    print(os.path.join(paras.last_data_dir, 'val'))
    val_datset = Dataset(nii_dir=os.path.join(paras.last_data_dir, 'val'), transform=transform)

    # 总样本数
    num_train = len(train_dataset)
    num_val = len(val_datset)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=8, num_workers=nw,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_datset, shuffle=False, batch_size=8, num_workers=nw,
                            pin_memory=True)

    # 一轮epcoh 迭代的步长
    train_step = len(train_loader)

    model = HcNet(num_classes=2, K=16, group_num=50)
    model.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    loss_list = []
    acc_list = []
    best_acc = 0

    for epoch in range(paras.epoch):
        train_name_list = []
        val_name_list = []

        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_acc_num = 0
        for step, train_data in enumerate(train_bar):
            train_images, train_labels, train_name = train_data

            optimizer.zero_grad()
            outputs = model(train_images.to(device))
            out_1 = outputs[0]
            out_2 = outputs[1]

            loss_1 = loss_function(out_1, train_labels.to(device))
            loss_2 = loss_function(out_2, train_labels.to(device))
            loss = 0.5 * loss_1 + 0.5 * loss_2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predict_y = torch.max(out_1 + out_2, dim=1)[1]
            train_acc_num += torch.eq(predict_y, train_labels.to(device)).sum().item()

            train_correct_indices = torch.nonzero(torch.ne(predict_y, train_labels.to(device))).squeeze().tolist()
            if not isinstance(train_correct_indices, list):
                train_correct_indices = [train_correct_indices]
            train_correct_names = [train_name[i] for i in train_correct_indices]
            for name in train_correct_names:
                if name not in train_name_list:
                    train_name_list.append(name)

            train_bar.desc = "Train --- Epoch:[{}/{}] --- batch_loss:{:.3f}".format(
                epoch + 1, paras.epoch, loss)

        train_acc = train_acc_num / num_train
        print("train_acc :", train_acc)
        # print("train_error_instance:", train_name_list)

        model.eval()
        val_acc_num = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_image, val_labels, val_name = val_data
                outputs = model(val_image.to(device))
                out_1 = outputs[0]
                out_2 = outputs[1]
                predict_y = torch.max(out_1 + out_2, dim=1)[1]
                val_acc_num += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_correct_indices = torch.nonzero(torch.ne(predict_y, val_labels.to(device))).squeeze().tolist()
                if not isinstance(val_correct_indices, list):
                    val_correct_indices = [val_correct_indices]
                val_correct_names = [val_name[i] for i in val_correct_indices]
                for name in val_correct_names:
                    if name not in val_name_list:
                        val_name_list.append(name)

            val_acc = val_acc_num / num_val
            train_loss = running_loss / train_step
            loss_list.append(running_loss / train_step)
            acc_list.append(val_acc)
            print("Val: --- Epoch[{}/{}] --- train_loss:{:.3f} --- val_acc:{:.3f}".format(
                epoch + 1, paras.epoch, train_loss, val_acc
            ))
            # print("val_error_instance:", val_name_list)
            scheduler.step(train_loss)
            for param_group in optimizer.param_groups:
                print("当前学习率: ", param_group['lr'])
            print("===" * 20)

            if epoch+1 % 50 == 0:
                torch.save(model.state_dict(), os.path.join(paras.save_path, f'HcNet_{epoch}.pth'))

        reslut_file = os.path.join(paras.save_path, 'HcNet.csv')
        with open(reslut_file, 'w', newline="") as file:
            write = csv.writer(file)
            # 写入表头
            write.writerow(["epoch", "loss", "acc"])
            for e, (l, a) in enumerate(zip(loss_list, acc_list)):
                write.writerow([e, l, a])


if __name__ == '__main__':
    train()

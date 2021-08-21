# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.cutpaste import *
import monai
import torch.nn.functional as F
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity
from monai.networks.nets import DenseNet
from torch.utils.data.sampler import SubsetRandomSampler
import nibabel as nib

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    # the path of ixi IXI-T1 dataset
    on_local = False
    data_path = '/Users/ahmedelgazzar/Downloads/brain_toy/toy/*.gz' if on_local else '/data_local/deeplearning/Datasets/MOOD/brain_train/*.gz'
    all_images = glob.glob(data_path)
    show_sample = False

    # Define transforms
    healthy_transform = Compose([ScaleIntensity(), AddChannel()])
    abnormal_transforms = Compose([ScaleIntensity(),CutPasteNormal(), AddChannel()])

    # Define dataset, data loader
    healthy_ds = NiftiDataset(image_files=all_images, labels=np.zeros(len(all_images)), transform=healthy_transform)
    abnomral_ds = NiftiDataset(image_files=all_images, labels=np.ones(len(all_images)), transform=abnormal_transforms)
    dataset_all = torch.utils.data.ConcatDataset([healthy_ds,abnomral_ds])

    num_samples = len(dataset_all)
    indices = list(range(num_samples))
    split = int(np.floor(0.2 * num_samples))

    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset_all,batch_size=2, sampler=train_sampler,
                                               num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset_all, batch_size=2, sampler=valid_sampler,
                                             num_workers=2, pin_memory=True)



    if show_sample:
        samples_path = '/Users/ahmedelgazzar/Downloads/brain_toy/'
        sample_loader = torch.utils.data.DataLoader(
            dataset_all, batch_size=8, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()

        X = images.numpy()
        Y = labels.numpy()

        for i, (x,y) in enumerate(zip(X,Y)):
            print(x.shape)
            ni_img = nib.Nifti1Image(x[0],  affine=np.eye(3))
            nib.save(ni_img, samples_path + 'sample_{}_class-{}.nii.gz'.format(i,y))



    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for inputs,labels in train_loader:
            step += 1
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_loader)*2 // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
        for inputs,labels in val_loader:
            step += 1
            val_images = inputs.to(device)
            val_labels = labels.type(torch.LongTensor).to(device)
            val_outputs = model(val_images)
            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()

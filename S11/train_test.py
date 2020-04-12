# -*- coding: utf-8 -*-
"""
This file contains handy functions for train and predict on CIFAR10 data using ResNet18
"""

from __future__ import print_function
import torch
from tqdm import tqdm

# model training
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_acc = []
    train_loss = 0
    avg_train_loss = 0
    train_accuracy = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        max_prob = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += max_prob.eq(target.view_as(max_prob)).sum().item()
        processed += len(data)
        accuracy = 100 * correct / processed
        pbar.set_description(desc= f'epoch={epoch} Loss={loss.item()} Batch_id={batch_idx} Accuracy={accuracy:0.2f}')
        train_acc.append(accuracy)
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader.dataset)

    #total = len(train_loader.dataset)
    #train_accuracy = 100. * correct / total

    return avg_train_loss, accuracy



# model evaluation
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    avg_test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_test_loss = test_loss / len(test_loader.dataset)

    total = len(test_loader.dataset)
    test_accuracy = 100. * correct / total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        avg_test_loss, correct, len(test_loader.dataset), test_accuracy))
    return avg_test_loss, test_accuracy


# May be, there's an easy way to get the misclassified images that I am not aware of. Ayway, repreidtcing a set of images and getting the misclassifed
# images
def get_missclassfied_images(model, device, data_loader, classes, n=25):
    """
    Predicts on a given dataset and picks the n misclassfied images
    """
    model.eval()
    mc_images = []
    mc_labels = []
    mc_pred_labels = []

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            preds = pred.cpu().numpy()

            pred_classes = [np.array(classes)[p] for p in preds]
            # get the correct ones
            results = pred.eq(target.view_as(pred)).cpu().numpy()

            idxs = [i for i, r in enumerate(results) if not r]

            for i in idxs:
                mc_images.append(data)
                mc_labels.append(classes[target[i]])
                mc_pred_labels.append(pred_classes[i][0])

            if len(mc_images) <= n:
                break;
    return mc_images, mc_labels, mc_pred_labels


def get_missclassfied_images1(model, device, data_loader, classes, n=25):
    """
    Predicts on a given dataset and picks the n misclassfied images
    """
    model.eval()
    mc_images = []
    mc_labels = []
    mc_pred_labels = []

    with torch.no_grad():
        dataiter = iter(data_loader)
        i = 1
        while True:
            try:
                data, target = dataiter.next()
                data, target = data.to(device), target.to(device)
                output = model(data)
                print(i + 1)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                preds = pred.cpu().numpy()

                pred_classes = [np.array(classes)[p] for p in preds]
                # get the correct ones
                results = pred.eq(target.view_as(pred)).cpu().numpy()

                idxs = [i for i, r in enumerate(results) if not r]

                for i in idxs:
                    mc_images.append(data)
                    mc_labels.append(classes[target[i]])
                    mc_pred_labels.append(pred_classes[i][0])

                if len(mc_images) <= n:
                    break;
            except StopIteration:
                break
    return mc_images, mc_labels, mc_pred_labels

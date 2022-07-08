import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn import functional as F
from PIL import Image
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, ConcatDataset
import os

print("My choice:Difference between training with and without the Cutout data augmentation algorithm implemented in Task 2")
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def Cutout (img,n_holes,s):
    h = img.size(1)
    w = img.size(2)
    
    mask = np.ones((h, w), np.float32)
    s = np.random.random_integers(0,s)
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y -  s // 2, 0, h)
        y2 = np.clip(y + s // 2, 0, h)
        x1 = np.clip(x - s // 2, 0, w)
        x2 = np.clip(x + s // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return img

def apply_cutout(dataset):
    new_trainset=[]
    for image,label in dataset:
        image_cut=Cutout(image,n_holes=1,s=8)
        new_trainset.append((image_cut,label))
    return new_trainset

def train(model, train_loader, optimizer, epoch):
    model.train()
    ep = 100
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target

        # clear gradients
        optimizer.zero_grad()

        # compute loss
        loss = F.cross_entropy(model(data), target)

        # get gradients and update
        loss.backward()
        optimizer.step()
        
        if batch_idx % ep == 0:
            print("Epoch :",epoch, "batch :", batch_idx/ ep, "loss :", loss.detach().numpy())


def eval_test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total
    return test_loss, test_accuracy

transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
        ])
development_set = CIFAR10("./data", download=True, transform=transform, train=True)
houldout_test_set = CIFAR10("./data", download=True, transform=transform, train=False)

def task_a():
    print("task a using Cutout")
    # Configuration options
    k = 3
    num_epochs = 10
    batch_size = 32
    
    # cifar-10 dataset
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
        ])

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare CIFAR10 dataset by concatenating Train/Test part; we split later.
    dataset= CIFAR10("./data", download=True, transform=transform, train=True)
    dataset=apply_cutout(dataset)
    
    # Define the K-fold Cross Validator
    num_validation_samples = len(dataset) // k

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold in range(k):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        test_ids = list(range(num_validation_samples*fold, num_validation_samples*(fold+1)))
        train_ids = list(range(0, num_validation_samples*fold))  + list(range(num_validation_samples * (fold+1), len(dataset)))
        print(len(test_ids),len(train_ids))
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size, sampler=test_subsampler)
        
        model = Net()

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(1, num_epochs):
            start_time = time.time()

            # training
            train(model, trainloader, optimizer, epoch)

            trnloss, trnacc = eval_test(model, testloader)

            # print trnloss and testloss
            print('Epoch '+str(epoch)+': ' +
                str(int(time.time()-start_time))+'s', end=', ')
            print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(
                trnloss, 100. * trnacc), end=', ')

        # save the model
        torch.save(model.state_dict(), 'task_a_saved_model.pt')
        print('Model saved.')
        
def task_b():
    print("task b no using Cutout")
    # Configuration options
    k = 3
    num_epochs = 10
    batch_size = 32
    
    # cifar-10 dataset
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    dataset = CIFAR10("./data", download=True, transform=transform, train=True)
    # Define the K-fold Cross Validator
    num_validation_samples = len(dataset) // k

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold in range(k):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        test_ids = list(range(num_validation_samples*fold, num_validation_samples*(fold+1)))
        train_ids = list(range(0, num_validation_samples*fold))  + list(range(num_validation_samples * (fold+1), len(dataset)))
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size, sampler=test_subsampler)
        
        model =     Net()

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(1, num_epochs):
            start_time = time.time()

            # training
            train(model, trainloader, optimizer, epoch)

            trnloss, trnacc = eval_test(model, testloader)

            # print trnloss and testloss
            print('Epoch '+str(epoch)+': ' +
                str(int(time.time()-start_time))+'s', end=', ')
            print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(
                trnloss, 100. * trnacc), end=', ')

        # save the model
        torch.save(model.state_dict(), 'task_b_saved_model.pt')
        print('Model saved.')
        
def task_c():
    print("task c not using Cutout and using whole training data")
    # Configuration options
    num_epochs = 10
    batch_size = 32
    
    # cifar-10 dataset
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation

        # Define data loaders for training and testing data in this fold
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    model =     Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Training done.')
    trnloss, trnacc = eval_test(model, testloader)
            # print trnloss and testloss
    print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(
                trnloss, 100. * trnacc), end=', ')

    # save the model
    torch.save(model.state_dict(), 'task_c_saved_model.pt')
    print('Model saved.')

def task_d():
    print("task d  using Cutout and using whole training data")
    # Configuration options
    num_epochs = 10
    batch_size = 32
    
    # cifar-10 dataset
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
        ])

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)


    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation

        # Define data loaders for training and testing data in this fold
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset=apply_cutout(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    model =     Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Training done.')
    trnloss, trnacc = eval_test(model, testloader)
            # print trnloss and testloss
    print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(
                trnloss, 100. * trnacc), end=', ')

    # save the model
    torch.save(model.state_dict(), 'task_d_saved_model.pt')
    print('Model saved.')


def main():
    task_a()
    task_b()
    task_c()
    task_d()

    
if __name__ == '__main__':
    main()

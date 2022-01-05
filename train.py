'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
from sklearn.metrics import log_loss, brier_score_loss
from scipy.special import expit

from models import *
from calibration_tools.evaluate import evaluate
from calibration_tools.modified_training import BaselineTrainable, FocalLoss, EntropyRegularizedLoss, MMCELoss
from calibration_tools.ensemble import BaselineEnsemble
from calibration_tools.calibrate import calibrate_model, inference
from calibration_tools.cape import CaPE
from datasets import FaceDataset, CancerDataset
import h5py
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model-name', default='ckpt.pth', type=str)
parser.add_argument('--model-dir', default='crossentropy', type=str, required=True)
parser.add_argument('--label_type', default='equal', type=str)
parser.add_argument('--methods', default='ce', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def unpickling(file):
    return pickle.load(open(file, 'rb'))


class CIFAR10withidx(torch.utils.data.Dataset):
    def __init__(self, root, targets10_path, download=True, mode='train', train=True, transform=None,
                 target_transform=None):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root,
                                                    download=download,
                                                    train=train,
                                                    transform=transform,
                                                    target_transform=target_transform)
        # print(self.cifar10.targets)
        if mode == 'train':
            self.cifar10.data = self.cifar10.data[0:45000]
        elif mode == 'val':
            self.cifar10.data = self.cifar10.data[45000:]
        self.cifar10.targets = unpickling(targets10_path)[mode + '_targets']
        self.cifar10.targets_10 = unpickling(targets10_path)[mode + '_10']
        self.proposed_probs = None

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        target_10 = self.cifar10.targets_10[index]
        if self.proposed_probs is not None:
            probs = self.proposed_probs[index]
        else:
            probs = 0
        return data, target_10, probs, index, target

    def __len__(self):
        return len(self.cifar10)


def unpickling(file):
    return pickle.load(open(file, 'rb'))


if args.dataset == 'face':
    assert args.label_type in {'linear', 'sig', 'skewed', 'centered', 'discrete'}
    trainset = FaceDataset(root='./Faces_detection/', prob_type=args.label_type, mode='train', std_classifier=False)
    valset = FaceDataset(root='./Faces_detection/', prob_type=args.label_type, mode='val', std_classifier=False)
    testset = FaceDataset(root='./Faces_detection/', prob_type=args.label_type, mode='test', std_classifier=False)
elif args.dataset == 'weather':
    raise NotImplementedError
elif args.dataset == 'traffic':
    raise NotImplementedError
elif args.dataset == 'cancer':
    bag_dir = './cancer_survival/'
    trainset = CancerDataset(bag_dir + "train")
    valset = CancerDataset(bag_dir + "val")
    testset = CancerDataset(bag_dir + "test")
else:
    raise NotImplementedError


def cancer_collate_func(data):
    x_list, label_list, probs_list, index_list, gt_list = [], [], [], [], []
    for x, label, probs, index, gt in data:
        x_list.append(x)
        label_list.append(label)
        probs_list.append(probs)
        index_list.append(index)
        gt_list.append(gt)
    return x_list, torch.LongTensor(label_list), torch.FloatTensor(probs_list), torch.LongTensor(
        index_list), torch.FloatTensor(gt_list)


if args.dataset == 'cancer':
    batch_size = 16
    # baseline_method_v2.BATCH_SIZE = 16
    # baseline_trainable_v2.BATCH_SIZE = 16
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, collate_fn=cancer_collate_func, num_workers=4)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, collate_fn=cancer_collate_func, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, collate_fn=cancer_collate_func, num_workers=4)
else:
    batch_size = 256
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.dataset == 'face':
    net = torchvision.models.resnet18(num_classes=2)
elif args.dataset == 'cancer':
    # net = GatedAttention(batch=True)
    raise NotImplementedError
else:
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.model_name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# weights = [1/0.9,10]
# class_weights = torch.FloatTensor(weights).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=args.lr,
#                        weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, \
                                                       min_lr=1e-6, verbose=True)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _, _, _) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("Loss: %.3f" % (train_loss / (batch_idx + 1)))


def test(net, dataloader):
    global best_acc
    net.eval()
    targets_probs = np.zeros(len(dataloader.dataset))
    labels = np.zeros(len(dataloader.dataset))
    indices = np.zeros(len(dataloader.dataset))
    gt_labels = np.zeros(len(dataloader.dataset))
    net.eval()
    with torch.no_grad():
        pointer = 0
        for batch_idx, (inputs, label, _, ids, gt_label) in tqdm(enumerate(dataloader)):
            if "WSI" in str(type(dataloader.dataset)):
                idx = np.arange(pointer, pointer + len(ids))
                pointer += len(ids)
            else:
                idx = ids
            inputs = inputs.to(device)
            outputs = net(inputs)
            out_prob = F.softmax(outputs, dim=1)
            targets_probs[idx] = out_prob[:, 1].cpu().numpy()
            labels[idx] = label
            gt_labels[idx] = gt_label
            indices[idx] = ids

    if "WSI" in str(type(dataloader.dataset)):
        tile_level = {
            "target_prob": targets_probs,
            "labels": labels,
            "gt": gt_labels,
            "slide_id": indices
        }
        slide_level = dataloader.dataset._aggregate_tile(tile_level)
        targets_probs, labels, indices, gt_labels = np.array(slide_level['target_prob']), \
                                                    np.array(slide_level['labels']), np.array(
            slide_level['slide_id']), np.array(slide_level['gt'])
    return targets_probs, labels


print(os.path.join(args.model_dir, args.model_name))

if args.methods == 'ce':
    min_val_loss = 1e10
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        val_targets_probs, labels = test(net, valloader)

        val_loss = log_loss(y_true=labels, y_pred=val_targets_probs)

        state = {
            'net': net.state_dict(),
            'val_loss': val_loss,
            'epoch': epoch,
        }
        if min_val_loss > val_loss:
            print('Saving..')
            print('val_loss: {:.3f}'.format(val_loss))
            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(state, os.path.join(args.model_dir, args.model_name))
            min_val_loss = val_loss
        torch.save(state, os.path.join(args.model_dir, "epoch_{}.pth".format(epoch)))
        scheduler.step(val_loss)

elif args.methods == 'deepensemble':
    M = 5
    adversarial_epsilon = 0.01
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    baseline = BaselineEnsemble(net, M, optimizer, criterion, trainset, valset, adversarial_epsilon=adversarial_epsilon,
                                save_dir=os.path.join(args.model_dir, args.model_name), num_epoch=200)
    baseline.fit()

elif args.methods == 'ours_bin':
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    # lr = 1e-3
    model_path = "./checkpoints/survival_bag/ce/ckpt.pth"
    checkpoint = torch.load(model_path)['net']
    net.load_state_dict(checkpoint)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    m_kwargs = {
        "net": net,  # early stopped network
        "optimizer": optimizer,  # optimizer for finetuning
        "train_dataset": trainset,  # train set
        "val_dataset": valset,  # validation set for finetuning stopping
        "num_epoch": 100,  # max number of epochs for finetuning
        "n_bins": 5,  # number of bins for updated probabilistic labels
        "calpertrain": 2,
        "finetune_type": "bin",
        "save_dir": os.path.join(args.model_dir, args.model_name)
    }
    calibrate_model(CaPE, m_kwargs=m_kwargs, test_dataset=testset)

elif args.methods == 'ours_kd':
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_path = "./checkpoints/survival_bag/ce/ckpt.pth"
    checkpoint = torch.load(model_path)['net']
    net.load_state_dict(checkpoint)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    m_kwargs = {
        "net": net,  # early stopped network
        "optimizer": optimizer,  # optimizer for finetuning
        "train_dataset": trainset,  # train set
        "val_dataset": valset,  # validation set for finetuning stopping
        "num_epoch": 100,  # max number of epochs for finetuning
        "calpertrain": 3,
        "finetune_type": "kde",
        "sigma": 0.05,
        "window": 100,
        "save_dir": os.path.join(args.model_dir, args.model_name)
    }
    calibrate_model(CaPE, m_kwargs=m_kwargs, test_dataset=testset)

else:
    if args.methods == "focal":
        criterion = FocalLoss(alpha=None, gamma=2.0)
    elif args.methods == "entropy":
        criterion = EntropyRegularizedLoss(beta=1.0)
    elif args.methods == "MMCE":
        criterion = MMCELoss(beta=3.0)
    else:
        raise NotImplementedError
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    baseline = BaselineTrainable(net, optimizer, criterion, trainset, valset,
                                 save_dir=os.path.join(args.model_dir, args.model_name), num_epoch=200)
    baseline.fit()

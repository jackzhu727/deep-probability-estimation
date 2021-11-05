import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
import copy


class BaselineEnsemble():
    def __init__(self, net, M, optimizer, criterion, train_dataset, val_dataset, adversarial_epsilon=None, batch_size=128,
                 save_dir="best_checkpoint.pth", num_epoch=10):
        """
        Initialize class

        Params:
            net : early stopped neural network
            M: number of networks
            train_dataset : pytorch training dataset
            val_dataset : pytorch validation dataset
            num_epoch : epochs of finetuning the model

        """
        self.M = M
        self.save_dir = save_dir
        self.net = nn.ModuleList([copy.deepcopy(net) for _ in range(M)])
        for i in range(len(self.net)):
            self.init_params(self.net[i])

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = [torch.optim.SGD(self.net[i].parameters(), lr=optimizer.param_groups[0]['lr'],
                                           momentum=0.9, weight_decay=5e-4) for i in range(len(self.net))]
        self.criterion = criterion
        self.scheduler = [torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer[i], 'min', patience=5, factor=0.5, \
                                                    min_lr=1e-6, verbose=True) for i in range(len(self.net))]
        self.num_epoch = num_epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.adversarial_epsilon = adversarial_epsilon

    def init_params(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_nets(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        print('Load checkpoint: {}'.format(checkpoint['epoch']))
        self.net.load_state_dict(checkpoint['net'])

    def _adversarial(self, x, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        x_prime = x + self.adversarial_epsilon * sign_data_grad
        return x_prime

    def _train_epoch(self):
        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=8)
        self.net.train()
        train_loss = 0
        for batch_idx, (inputs, labels, _, idx, gt_label) in enumerate(trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for i in range(self.M):
                self.optimizer[i].zero_grad()
                if self.adversarial_epsilon:
                    inputs.requires_grad = True
                outputs = self.net[i](inputs)
                loss = self.criterion(outputs, labels)
                if self.adversarial_epsilon:
                    self.net[i].zero_grad()
                    loss.backward(retain_graph=True)
                    data_grad = inputs.grad.data
                    x_prime = self._adversarial(inputs, data_grad)
                    outputs_prime = self.net[i](x_prime)
                    loss_prime = self.criterion(outputs_prime, labels)
                    loss += loss_prime
                loss.backward()
                self.optimizer[i].step()
                train_loss += loss.item()/self.M
        return train_loss / (batch_idx + 1)

    def _inference(self, net, dataloader, return_gt=False):
        net.eval()
        targets_probs_list = []
        for i in range(self.M):
            targets_probs = np.zeros(len(dataloader.dataset))
            labels = np.zeros(len(dataloader.dataset))
            indices = np.zeros(len(dataloader.dataset))
            gt_labels = np.zeros(len(dataloader.dataset))
            net[i].eval()
            with torch.no_grad():
                for batch_idx, (inputs, label, _, idx, gt_label) in enumerate(dataloader):
                    inputs = inputs.to(self.device)
                    outputs = net[i](inputs)
                    out_prob = F.softmax(outputs,dim=1)
                    targets_probs[idx] = out_prob[:,1].cpu().numpy()
                    labels[idx] = label
                    gt_labels[idx] = gt_label
            targets_probs_list.append(targets_probs)
        targets_probs = np.stack(targets_probs_list).mean(axis=0)
        if return_gt:
            return targets_probs, labels, indices, gt_labels
        else:
            return targets_probs, labels, indices

    def fit(self):
        min_val_loss = 1e10
        for epoch in range(0, self.num_epoch):
            trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=8)
            valloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                                    num_workers=8)

            loss = self._train_epoch()
            # evaluate
            val_targets_probs, labels, _ = self._inference(self.net, valloader)

            val_loss = log_loss(y_true=labels, y_pred=val_targets_probs)
            for i in range(len(self.scheduler)):
                self.scheduler[i].step(val_loss)
            if min_val_loss > val_loss:
                torch.save({
                    'net': self.net.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch,}, self.save_dir)
                min_val_loss = val_loss
            print('Epoch: ', epoch, ' Loss: %.3f; Val Loss: %.3f' % (loss, val_loss))

    def predict(self, test_dataset, file=None):
        if file is not None:
            self._load_nets(file)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        test_targets_probs, labels, _, gt_labels = self._inference(self.net, testloader, return_gt=True)
        test_targets_probs = np.stack([1 - test_targets_probs, test_targets_probs]).T
        return test_targets_probs, labels, gt_labels
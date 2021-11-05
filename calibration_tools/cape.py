import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss


class CaPE():
    def __init__(self, net, optimizer, train_dataset, val_dataset, finetune_type='bin', num_epoch=10, n_bins=10,
                 calpertrain=5, sigma=0.1, window=500, batch_size=256, save_dir="best_checkpoint.pth"):
        """
        Initialize class

        Params:
            net : early stopped neural network
            train_dataset : pytorch training dataset
            val_dataset : pytorch validation dataset
            num_epoch : epochs of finetuning the model
            n_bins: number of bins to calculate probability targets.

        """
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_bins = n_bins
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.calpertrain = calpertrain
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = save_dir
        self.sigma = sigma
        self.window = window
        self.finetune_type = finetune_type
        self.batch_size = batch_size

    def _inference(self, net, dataloader, return_gt=False):
        targets_probs = np.zeros(len(dataloader.dataset))
        labels = np.zeros(len(dataloader.dataset))
        indices = np.zeros(len(dataloader.dataset))
        gt_labels = np.zeros(len(dataloader.dataset))
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, label, _, ids, gt_label) in enumerate(dataloader):
                idx = ids
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                out_prob = F.softmax(outputs, dim=1)
                targets_probs[idx] = out_prob[:, 1].cpu().numpy()
                labels[idx] = label
                indices[idx] = ids
                gt_labels[idx] = gt_label

        if return_gt:
            return targets_probs, labels, indices, gt_labels
        else:
            return targets_probs, labels, indices

    def get_new_prob(self, mean_value, prob_array, true_array, scale):
        weight = scipy.stats.norm.pdf(prob_array, loc=mean_value, scale=scale)
        return np.sum(weight * true_array) / np.sum(weight)

    def _sort_update(self, targets_probs, labels, indices, n_bins=10):
        sorted_idx = np.argsort(targets_probs)
        targets_probs = targets_probs[sorted_idx]
        labels = labels[sorted_idx]
        indices = indices[sorted_idx]
        num_sample = len(labels)
        proposed_probs = np.zeros(num_sample)
        new_labels = np.zeros(num_sample)
        if self.finetune_type == 'bin':
            for i in range(n_bins):
                left = int(i * num_sample / n_bins)
                right = int((i + 1) * num_sample / n_bins)
                new_labels[left:right] = np.mean((labels[left:right]))
        elif self.finetune_type == 'kde':
            for i in range(num_sample):
                left = np.maximum(0, i - self.window)
                right = np.minimum(i + self.window, num_sample)
                new_labels[i] = self.get_new_prob(targets_probs[i],
                                                  targets_probs[left:right], labels[left:right], scale=self.sigma)
        else:
            raise NotImplementedError

        for i in range(num_sample):
            proposed_probs[int(indices[i])] = new_labels[i]
        self.train_dataset.proposed_probs = proposed_probs
        print('%' * 40)
        print('\nUpdated the Probs Labels\n')

    def _calibrate(self, trainloader, use_prob=True):
        self.net.train()
        train_loss = 0

        if use_prob:
            print('Using Binned Probs')
        else:
            print('Using GT')

        for batch_idx, (inputs, labels, targets_probs, idx, _) in enumerate(trainloader):
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            if use_prob:
                outputs = outputs.log_softmax(dim=1)
                targets = torch.stack([1 - targets_probs, targets_probs]).T.to(self.device)
                loss = torch.mean(torch.sum(-targets * outputs, dim=1))
            else:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # print(batch_idx, len(trainloader), 'Loss: %.3f'
            #          % (train_loss/(batch_idx+1)))
        return train_loss / (batch_idx + 1)

    @staticmethod
    def _sigmoid_rampup(current, rampup_length):
        """Exponential rampup from  2"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    # Find the temperature
    def fit(self):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, \
                                                               min_lr=1e-6, verbose=True)
        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=4)
        min_val_loss = 1e4
        for epoch in range(0, self.num_epoch):
            targets_probs, labels, indices = self._inference(self.net, trainloader)
            # n_bins = _sigmoid_rampup(epoch, rampup_length)*30
            use_prob = (epoch % (self.calpertrain + 1) == 0)
            if use_prob:
                self._sort_update(targets_probs, labels, indices, n_bins=self.n_bins)
            trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=8)
            valloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                                    num_workers=8)

            loss = self._calibrate(trainloader, use_prob=use_prob)

            # evaluate
            val_targets_probs, labels, _ = self._inference(self.net, valloader)

            val_loss = log_loss(y_true=labels, y_pred=val_targets_probs)

            if min_val_loss > val_loss:
                torch.save({
                    'net': self.net.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch, }, self.save_dir)
                min_val_loss = val_loss
            print('Epoch: ', epoch, ' Loss: %.3f' % (loss), 'Val Loss: %.3f' % (val_loss))
            if epoch % 5 == 0 and epoch != 0:
                torch.save({
                    'net': self.net.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch, }, self.save_dir.rstrip('.pth') + "_{}.pth".format(epoch))
            scheduler.step(val_loss)

    def predict(self, test_dataset, file=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if file is not None:
            self.net.load_state_dict(torch.load(file)['net'])
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=8)
        test_targets_probs, labels, _, gt_labels = self._inference(self.net, testloader, return_gt=True)

        test_targets_probs = np.stack([1 - test_targets_probs, test_targets_probs]).T

        return test_targets_probs, labels, gt_labels
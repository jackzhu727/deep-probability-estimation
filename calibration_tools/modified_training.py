import torch
import numpy as np
import torch.nn


class BaselineTrainable():
    def __init__(self, net, optimizer, criterion, train_dataset, val_dataset, batch_size,
                 save_dir="best_checkpoint.pth", num_epoch=10):
        """
        Initialize class

        Params:
            net : early stopped neural network
            train_dataset : pytorch training dataset
            val_dataset : pytorch validation dataset
            num_epoch : epochs of finetuning the model
        """
        self.save_dir = save_dir
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, \
                                                                    min_lr=1e-6, verbose=True)
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _train_epoch(self):
        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.net.train()
        train_loss = 0
        for batch_idx, (inputs, labels, _, idx, gt_label) in enumerate(trainloader):
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / (batch_idx + 1)

    def _inference(self, net, dataloader, return_gt=False):
        net.eval()
        targets_probs = np.zeros(len(dataloader.dataset))
        labels = np.zeros(len(dataloader.dataset))
        indices = np.zeros(len(dataloader.dataset))
        gt_labels = np.zeros(len(dataloader.dataset))
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, label, _, idx, gt_label) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                out_prob = F.softmax(outputs,dim=1)
                targets_probs[idx] = out_prob[:,1].cpu().numpy()
                labels[idx] = label
                gt_labels[idx] = gt_label
        if return_gt:
            return targets_probs, labels, indices, gt_labels
        else:
            return targets_probs, labels, indices

    def _load_nets(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        print('Load checkpoint: {}'.format(checkpoint['epoch']))
        self.net.load_state_dict(checkpoint['net'])

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
            self.scheduler.step(val_loss)
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class EntropyRegularizedLoss(nn.Module):
    '''
    Loss regularized by entropy implementation
    L = CE - beta * H
    '''
    def __init__(self, beta=1, weight=None):
        super(EntropyRegularizedLoss, self).__init__()
        self.beta = beta
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        p_logp = (logpt * logpt.exp())
        entropy = -p_logp.sum(dim=1)
        loss = F.nll_loss(logpt, target) - self.beta * entropy.mean()
        return loss


class MMCELoss(nn.Module):
    '''
        Loss regularized by MMCE
        L = CE + beta * MMCE
        '''

    def __init__(self, beta=1, weight=None):
        super(MMCELoss, self).__init__()
        self.beta = beta
        self.weight = weight
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def mmce_w_loss(self, logits, correct_labels):
        # predicted_probs = F.softmax(logits, dim=1)
        predicted_probs = logits.exp()
        range_index = torch.arange(0, predicted_probs.shape[0]).unsqueeze(1).to(self.device)
        predicted_probs, predicted_labels = torch.max(predicted_probs, axis=1)
        gather_index = torch.cat([range_index, predicted_labels.unsqueeze(1)], dim=1)
        correct_mask = torch.where(torch.eq(correct_labels, predicted_labels),
                                   torch.ones(correct_labels.shape).to(self.device),
                                   torch.zeros(correct_labels.shape).to(self.device))
        sigma = 0.2

        def torch_kernel(matrix):
            return ((-1.0 * torch.abs(matrix[:, :, 0] - matrix[:, :, 1])) / (2 * 0.2)).exp()

        def get_out_tensor(tensor1, tensor2):
            return (tensor1 * tensor2).mean()

        def get_pairs(tensor1, tensor2):
            correct_prob_tiled = tensor1.unsqueeze(1).tile([1, tensor1.shape[0]]).unsqueeze(2)
            incorrect_prob_tiled = tensor2.unsqueeze(1).tile([1, tensor2.shape[0]]).unsqueeze(2)
            correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)], dim=2)
            incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)], dim=2)
            correct_prob_tiled_1 = tensor1.unsqueeze(1).tile([1, tensor2.shape[0]]).unsqueeze(2)
            incorrect_prob_tiled_1 = tensor2.unsqueeze(1).tile([1, tensor1.shape[0]]).unsqueeze(2)
            correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)], dim=2)
            return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs
        k = correct_mask.sum().item()
        k_p = (1.0 - correct_mask).sum().item()
        cond_k = 0 if k == 0 else 1
        cond_k_p = 0 if k == 0 else 1
        k = max(k, 1) * cond_k * cond_k_p + (1 - cond_k * cond_k_p) * 2
        k_p = max(k_p, 1) * cond_k_p * cond_k + ((1 - cond_k_p * cond_k) * (correct_mask.shape[0] - 2))
        correct_prob, _ = torch.topk(predicted_probs * correct_mask, int(k))
        incorrect_prob, _ = torch.topk(predicted_probs * (1 - correct_mask), int(k_p))
        correct_prob_pairs, incorrect_prob_pairs, \
        correct_incorrect_pairs = get_pairs(correct_prob, incorrect_prob)
        correct_kernel = torch_kernel(correct_prob_pairs)
        incorrect_kernel = torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = torch_kernel(correct_incorrect_pairs)

        sampling_weights_correct = torch.matmul((1.0 - correct_prob).unsqueeze(1),
                                                ((1.0 - correct_prob).unsqueeze(1)).T)
        correct_correct_vals = get_out_tensor(correct_kernel, sampling_weights_correct)
        sampling_weights_incorrect = torch.matmul(incorrect_prob.unsqueeze(1),
                                                  (incorrect_prob.unsqueeze(1)).T)
        incorrect_incorrect_vals = get_out_tensor(incorrect_kernel, sampling_weights_incorrect)
        sampling_correct_incorrect = torch.matmul((1.0 - correct_prob).unsqueeze(1),
                                                  (incorrect_prob.unsqueeze(1)).T)
        correct_incorrect_vals = get_out_tensor(correct_incorrect_kernel, sampling_correct_incorrect)
        correct_denom = (1.0 - correct_prob).sum()
        incorrect_denom = incorrect_prob.sum()
        m = correct_mask.sum()
        n = (1.0 - correct_mask).sum()
        mmd_error = 1.0 / (m * m + 1e-5) * correct_correct_vals.sum()
        mmd_error += 1.0 / (n * n + 1e-5) * incorrect_incorrect_vals.sum()
        mmd_error -= 2.0 / (m * n + 1e-5) * correct_incorrect_vals.sum()
        mmce_error = torch.maximum(float(cond_k * cond_k_p) * torch.sqrt(mmd_error + 1e-10), torch.Tensor([0.0])[0])
        return mmce_error

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        mmce_error = self.mmce_w_loss(logpt + 1e-10, target)
        loss = F.nll_loss(logpt+1e-10, target) + self.beta * mmce_error
        return loss

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import softmax, plot_scatter, plot_histogram, plot_empirical_distribution, plot_reliability_curve

BATCH_SIZE = 265
def calibrate_model(method, name='ours', m_kwargs={}, net=None, train_dataset=None, val_dataset=None, test_dataset=None,
                    approach='single', n_bins_4_plot=10, finetune=True, finetuned_model_path=None, plot_figure=True):
    """
    Params:
        method (class): class of the calibration method used. It must contain methods "fit" and "predict",
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))

    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    """

    df = pd.DataFrame(columns=["Name", "Error", "AUC", "ECE", "MCE", "Loss", "Brier", "Brier_w_gt", "brier_approx",
                               "KL_gt_prob", "KL_prob_gt", "ks_error"])
    if 'ours' not in name:
        # for i, f in enumerate(files):
        #     name = "_".join(f.split("_")[1:-1])
        #     FILE_PATH = join(path, f)
        #     (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)

        # Train calibration model

        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        val_logits, y_val, y_val_gt = inference(net, valloader)

        test_logits, y_test, y_test_gt = inference(net, testloader)

        probs_val = softmax(val_logits)

        probs_test = softmax(test_logits)

        error, auc, ece, mce, loss, brier, brier_gt, gt_cel_loss, kl_gt_prob, kl_prob_gt, ks_error = \
            evaluate(probs_test, y_test, y_test_gt, verbose=True)  # Test before scaling
        if plot_figure:
            fig, ax = plt.subplots(1, 4, figsize=(40, 10))
            plot_reliability_curve(y_test, probs_test, ece, mce, n_bins=n_bins_4_plot, ax=ax[0])
            plot_histogram(probs_test, y_test, y_test_gt, ax=ax[1])
            plot_scatter(probs_test, y_test, y_test_gt, ax=ax[2])
            plot_empirical_distribution(probs_test, y_test, y_test_gt, ax=ax[3], showplots=True)
        if approach == 'single':
            for k in range(probs_test.shape[1]):
                # print(np.array(y_val == k, dtype="int"))
                y_cal = np.array(y_val == k, dtype="int")

                # Train model
                model = method(**m_kwargs)
                model.fit(val_logits[:, k], y_cal)  # Get only one column with probs for given class "k"

                if "histogram" in name:
                    probs_val[:, k] = model.predict(probs_val[:, k])  # Predict new values based on the fittting
                    probs_test[:, k] = model.predict(probs_test[:, k])
                else:
                    probs_val[:, k] = model.predict_proba(val_logits[:, k])  # Predict new values based on the fittting
                    probs_test[:, k] = model.predict_proba(test_logits[:, k])

                # Replace NaN with 0, as it should be close to zero
                idx_nan = np.where(np.isnan(probs_test))
                probs_test[idx_nan] = 0

                idx_nan = np.where(np.isnan(probs_val))
                probs_val[idx_nan] = 0

            # _, probs_val = get_pred_conf(probs_val, normalize = True)
            # _, probs_test = get_pred_conf(probs_test, normalize = True)
            probs_val = probs_val / probs_val.sum(axis=1, keepdims=True)
            probs_test = probs_test / probs_test.sum(axis=1, keepdims=True)
        else:
            model = method(**m_kwargs)
            model.fit(val_logits, y_val)

            probs_test = model.predict_proba(test_logits)
            # print(probs_test)

        error2, auc2, ece2, mce2, loss2, brier2, brier_gt2, gt_cel_loss2, kl_gt_prob2, kl_prob_gt2, ks_error2 \
            = evaluate(probs_test, y_test, y_test_gt, verbose=False)
        if plot_figure:
            fig, ax = plt.subplots(1, 4, figsize=(40, 10))
            plot_reliability_curve(y_test, probs_test, ece2, mce2, n_bins=n_bins_4_plot, ax=ax[0])
            plot_histogram(probs_test, y_test, y_test_gt, ax=ax[1])
            plot_scatter(probs_test, y_test, y_test_gt, ax=ax[2])
            plot_empirical_distribution(probs_test, y_test, y_test_gt, ax=ax[3], showplots=True)

        df.loc[0] = [name, error, auc, ece, mce, loss, brier, brier_gt, gt_cel_loss, kl_gt_prob, kl_prob_gt, ks_error]
        df.loc[1] = [(name + "_calibrated"), error2, auc2, ece2, mce2, loss2, brier2, brier_gt2, gt_cel_loss2,
                     kl_gt_prob2, kl_prob_gt2, ks_error2]

    else:
        model = method(**m_kwargs)
        # probs_test, y_test, y_test_gt = model.predict(test_dataset, file=None)
        # error, auc, ece, mce, loss, brier, brier_gt, gt_cel_loss,  kl_gt_prob, kl_prob_gt, ks_error = \
        #     evaluate(probs_test, y_test, y_test_gt, verbose=True)  # Test before recalibration
        # error, auc, ece, mce, loss, brier, brier_gt, gt_cel_loss, kl_gt_prob, kl_prob_gt, ks_error = \
        #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # df.loc[0] = ['Checkpoint', error, auc, ece, mce, loss, brier, brier_gt, gt_cel_loss, kl_gt_prob, kl_prob_gt, ks_error]
        if finetune:
            model.fit()
            probs_test, y_test, y_test_gt = model.predict(test_dataset)
            error2, auc2, ece2, mce2, loss2, brier2, brier_gt2, gt_cel_loss2, kl_gt_prob2, kl_prob_gt2, ks_error2 = \
                evaluate(probs_test, y_test, y_test_gt, verbose=False)
            if plot_figure:
                fig, ax = plt.subplots(1, 4, figsize=(40, 10))
                plot_reliability_curve(y_test, probs_test, ece2, mce2, n_bins=n_bins_4_plot, ax=ax[0])
                plot_histogram(probs_test, y_test, y_test_gt, ax=ax[1])
                plot_scatter(probs_test, y_test, y_test_gt, ax=ax[2])
                plot_empirical_distribution(probs_test, y_test, y_test_gt, ax=ax[3], showplots=True)
            df.loc[0] = ['Checkpoint' + "_calibrated", error2, auc2, ece2, mce2, loss2, brier2, brier_gt2, gt_cel_loss2,
                         kl_gt_prob2, kl_prob_gt2, ks_error2]
        else:
            probs_test, y_test, y_test_gt = model.predict(test_dataset, file=finetuned_model_path)
            error2, auc2, ece2, mce2, loss2, brier2, brier_gt2, gt_cel_loss2, kl_gt_prob2, kl_prob_gt2, ks_error2 = \
                evaluate(probs_test, y_test, y_test_gt, verbose=False)
            if plot_figure:
                fig, ax = plt.subplots(1, 4, figsize=(40, 10))
                plot_reliability_curve(y_test, probs_test, ece2, mce2, n_bins=n_bins_4_plot, ax=ax[0])
                plot_histogram(probs_test, y_test, y_test_gt, ax=ax[1])
                plot_scatter(probs_test, y_test, y_test_gt, ax=ax[2])
                plot_empirical_distribution(probs_test, y_test, y_test_gt, ax=ax[3], showplots=True)
            df.loc[0] = ['Checkpoint' + "_calibrated", error2, auc2, ece2, mce2, loss2, brier2, brier_gt2, gt_cel_loss2,
                         kl_gt_prob2, kl_prob_gt2, ks_error2]
    return df


def inference(net, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    targets_logits = np.zeros((len(dataloader.dataset), 2))
    labels = np.zeros(len(dataloader.dataset))
    gt_labels = np.zeros(len(dataloader.dataset))
    indices = np.zeros(len(dataloader.dataset))
    net.eval()
    with torch.no_grad():
        pointer = 0
        for batch_idx, (inputs, label, _, ids, gt_label) in enumerate(dataloader):
            if "WSI" in str(type(dataloader.dataset)):
                idx = np.arange(pointer, pointer + len(ids))
                pointer += len(ids)
            else:
                idx = ids
            inputs = inputs.to(device)
            outputs = net(inputs)
            out_logits = outputs
            targets_logits[idx] = out_logits.cpu().numpy()
            labels[idx] = label
            gt_labels[idx] = gt_label
            indices[idx] = ids
    return targets_logits, labels, gt_labels

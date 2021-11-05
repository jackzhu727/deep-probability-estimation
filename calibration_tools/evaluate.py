import sklearn.metrics as metrics
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from utils import plot_empirical_distribution

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct) / len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin


def ECE(conf, pred, true, n_bins=None, bin_size=None):
    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """
    if n_bins:
        bin_size = 1 / n_bins
        quantiles = np.arange(bin_size, 1 + bin_size, bin_size)
        upper_bounds = np.array([np.quantile(conf, q) for q in quantiles])
    else:
        upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE

    return ece


def MCE(conf, pred, true, bin_size=0.1):
    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc - avg_conf))

    return max(cal_errors)


def ECE_balanced(y_prob, y_true, n_bins=10):
    x, y = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    ece = np.mean(np.abs(x - y))
    return ece


def MCE_balanced(y_prob, y_true, n_bins=10):
    x, y = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    mce = np.max(np.abs(x - y))
    return mce


def get_bin_info(conf, pred, true, bin_size=0.1):
    """
    Get accuracy, confidence and elements in bin information for all the bins.

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)

    return accuracies, confidences, bin_lengths

def evaluate(probs, y_true, y_true_gt, verbose=False, normalize=False, n_bins=15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

    if normalize:
        # confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
        confs = probs[:, 1] / np.sum(probs, axis=1)
    else:
        # confs = np.max(probs, axis=1)  # Take only maximum confidence
        confs = probs[:, 1]

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE
    ece = ECE_balanced(confs, y_true, n_bins=n_bins)
    # Calculate MCE
    mce = MCE_balanced(confs, y_true, n_bins=n_bins)

    loss = log_loss(y_true=y_true, y_pred=probs)
    auc = roc_auc_score(y_true=y_true, y_score=probs[:, 1])
    eps = 1e-10
    gt_cel_loss = -np.mean(y_true_gt * np.log(probs[:, 1]) + (1 - y_true_gt) * np.log(probs[:, 0]))
    kl_gt_prob = -np.mean(y_true_gt * np.log((probs[:, 1] + eps) / (y_true_gt + eps)) \
                          + (1 - y_true_gt) * np.log((probs[:, 0] + eps) / (1 - y_true_gt + eps)))
    kl_prob_gt = -np.mean(probs[:, 1] * np.log((y_true_gt + eps) / (probs[:, 1] + eps)) \
                          + probs[:, 0] * np.log((1 - y_true_gt + eps) / probs[:, 0] + eps))
    # print(y_true)

    y_prob_true = np.array([probs[i, int(idx)] for i, idx in enumerate(y_true)])  # Probability of positive class
    brier = brier_score_loss(y_true=y_true, y_prob=probs[:, 1])  # Brier Score (MSE)
    gt_brier = np.mean((y_true_gt - probs[:, 1]) ** 2)  # Brier Score for gt probs (MSE)
    brier_approx = brier - np.var(y_true) + 2 * np.cov(probs[:, 1], y_true)[0, 1]
    ks_error = plot_empirical_distribution(probs, y_true, y_true_gt, ax=None, showplots=False)

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print('AUC:', auc)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)
        print("brier with gt:", gt_brier)
        print("brier approximation:", brier_approx)
        print("Loss with gt:", gt_cel_loss)
        print("KL(gt || prob)", kl_gt_prob)
        print("KL(prob || gt )", kl_prob_gt)
        print("KS error", ks_error)

    return error, auc, ece, mce, loss, brier, gt_brier, brier_approx, kl_gt_prob, kl_prob_gt, ks_error

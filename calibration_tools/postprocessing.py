import numpy as np
from scipy.optimize import minimize
from utils import softmax, mse_t, ll_t, mse_w, ll_w
from sklearn.metrics import log_loss


class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.

    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """

    def __init__(self, n_bins=15):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1.0 / n_bins  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+ self.bin_size, self.bin_size)  # Set bin bounds for intervals

    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range

        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered) / nr_elems  # Sums positive classes
            return conf

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.

        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs=probs, true=true)
            conf.append(temp_conf)
        self.conf = np.array(conf)

    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences

        Param:
            probs: the output from neural network for each class (shape [samples, classes])

        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.

        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs


class TemperatureScaling():

    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict_proba(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    # Find the temperature
    def fit(self, logtis, true):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        # true = true.flatten() # Flatten y_val
        true = np.eye(2)[true.astype(int)]
        opt = minimize(self._loss_fun, x0=1, args=(logtis, true), options={'maxiter': self.maxiter}, method=self.solver)
        self.temp = opt.x[0]

        return opt

    def predict_proba(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)


class mix_n_match():
    def __init__(self, n_class=2, temp=1, maxiter=50, solver="BFGS"):
        """
        Initialize class
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
        self.n_class = n_class
    @staticmethod
    def temperature_scaling(logit,label,loss):
        bnds = ((0.05, 5.0),)
        if loss == 'ce':
           t = minimize(ll_t, 1.0, args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
        if loss == 'mse':
            t = minimize(mse_t, 1.0, args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
        t = t.x
        return t
    @staticmethod
    def ensemble_scaling(logit, label,loss,t,n_class):
        p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        logit = logit/t
        p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        p2 = np.ones_like(p0)/n_class
        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        if loss == 'ce':
            w = minimize(ll_w, (1.0, 0.0, 0.0), args=(p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
        if loss == 'mse':
            w = minimize(mse_w, (1.0, 0.0, 0.0), args=(p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
        w = w.x
        return w

    def fit(self, logtis, label):
        '''
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            label: true labels.
        '''
        label =  np.eye(2)[label.astype(int)]
        t = self.temperature_scaling(logtis,label,loss='mse') # loss can change to 'ce'
        print("temperature = " +str(t))
        w = self.ensemble_scaling(logtis,label,'mse',t, self.n_class)
        print("weight = " +str(w))
        self.t = t
        self.w = w

    def predict_proba(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        p1 = softmax(logits)
        logits = logits/self.t
        p0 = softmax(logits)
        p2 = np.ones_like(p0)/self.n_class
        return self.w[0]*p0 + self.w[1]*p1 +self.w[2]*p2
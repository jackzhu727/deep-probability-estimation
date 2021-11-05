import pickle
import numpy as np
from matplotlib import pyplot as plt


def unpickle_probs(file, verbose=0):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)  # unpickle the content
    if verbose:
        print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels
    return (y_probs_val, y_val), (y_probs_test, y_test)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def plot_reliability_curve(y, output_prob, ece, mce, ax=None, n_bins=15):
    idx_sorted = np.argsort(output_prob[:, 1])
    sorted_prob = output_prob[idx_sorted, 1]
    sorted_labels = y[idx_sorted]
    n = sorted_prob.shape[0]
    all_probs = []
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(7, 7))

    for i in range(n_bins):
        avg_pred_prob = np.mean(sorted_prob[int(i*(n//n_bins)):int((n//n_bins)*(i+1))])
        avg_true_prob = np.mean(sorted_labels[int(i*(n//n_bins)):int((n//n_bins)*(i+1))])
        all_probs.append(avg_pred_prob)
        all_probs.append(avg_true_prob)
        ax.scatter(avg_pred_prob, avg_true_prob, color = 'r')
    max_prob = np.max(all_probs)
    min_prob = np.min(all_probs)
    ax.set_xlim([min_prob * 0.95, max_prob * 1.05])
    ax.set_ylim([min_prob * 0.95, max_prob * 1.05])
    ax.plot(np.linspace(min_prob*0.95, max_prob*1.05), np.linspace(min_prob*0.95, max_prob*1.05), '--k')
    ax.set_xlabel(r"Predicted Probability")
    ax.set_ylabel(r"Empirical Probability")
    ax.axis('equal')
    ax.set_title('ECE = {:.3f}, MCE = {:.3f}'.format(ece, mce))


def plot_histogram(prob, label, gt, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.hist(prob[:, 1], range=(0, 1), histtype="stepfilled", bins=10, alpha=0.6)
    ax.hist(label, range=(0, 1), histtype="stepfilled", bins=10, alpha=0.6)
    ax.hist(gt, range=(0, 1), histtype="stepfilled", bins=10, alpha=0.6)


def plot_scatter(prob, gt, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(gt, prob[:, 1], '.', alpha=0.6)
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Estimated probability')


def plot_empirical_distribution(y_probs, y_label, y_gt, ax=None, showplots=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    prob = y_probs[:, 1]
    order = prob.argsort()
    prob = prob[order]
    label = y_label[order]
    gt = y_gt[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len(prob)
    integrated_scores = np.cumsum(prob) / nsamples
    integrated_accuracy = np.cumsum(label) / nsamples
    integrated_gts = np.cumsum(gt) / nsamples
    percentile = np.linspace(0.0, 1.0, nsamples)
    spline_method = 'natural'
    splines = 6
    fitted_accuracy, fitted_error = spline.compute_accuracy(prob, label, spline_method, splines, showplots=showplots, ax=ax)

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute(integrated_scores - integrated_accuracy))
    if showplots:
        # Set up the graphs
        f, ax = plt.subplots(1, 4, figsize=(20, 5))
        size = 0.2
        f.suptitle(f"\nKS-error = {spline.str(float(KS_error_max) * 100.0)}%, "
                           f"Probability={spline.str(float(integrated_accuracy[-1]) * 100.0)}%"
                   , fontsize=18, fontweight="bold")

        # First graph, (accumualated) integrated_scores and integrated_accuracy vs sample number
        ax[0].plot(100.0 * percentile, integrated_scores, linewidth=3, label='Cumulative Score')
        ax[0].plot(100.0 * percentile, integrated_accuracy, linewidth=3, label='Cumulative Probability')
        ax[0].set_xlabel("Percentile", fontsize=16, fontweight="bold")
        ax[0].set_ylabel("Cumulative Score / Probability", fontsize=16, fontweight="bold")
        ax[0].legend(fontsize=13)
        ax[0].set_title('(a)', y=-size, fontweight="bold", fontsize=16)  # increase or decrease y as needed
        ax[0].grid()

        # Second graph, (accumualated) integrated_scores and integrated_accuracy versus
        # integrated_scores
        ax[1].plot(integrated_scores, integrated_scores, linewidth=3, label='Cumulative Score')
        ax[1].plot(integrated_scores, integrated_accuracy, linewidth=3,
                   label="Cumulative Probability")
        ax[1].set_xlabel("Cumulative Score", fontsize=16, fontweight="bold")
        ax[1].set_ylabel("Cumulative Score / Probability", fontsize=12)
        ax[1].legend(fontsize=13)
        ax[1].set_title('(b)', y=-size, fontweight="bold", fontsize=16)  # increase or decrease y as needed
        ax[1].grid()

        # Third graph, scores and accuracy vs percentile
        ax[2].plot(100.0 * percentile, prob, linewidth=3, label='Score')
        ax[2].plot(100.0 * percentile, fitted_accuracy, linewidth=3, label=f"Probability")
        ax[2].set_xlabel("Percentile", fontsize=16, fontweight="bold")
        ax[2].set_ylabel("Score / Probability", fontsize=16, fontweight="bold")
        ax[2].legend(fontsize=13)
        ax[2].set_title('(c)', y=-size, fontweight="bold", fontsize=16)  # increase or decrease y as needed
        ax[2].grid()

        # Fourth graph,
        # integrated_scores
        ax[3].plot(prob, prob, linewidth=3, label=f"Score")
        ax[3].plot(prob, fitted_accuracy, linewidth=3, label='Probability')
        ax[3].set_xlabel("Score", fontsize=16, fontweight="bold")
        ax[3].set_ylabel("Score / Probability", fontsize=12)
        ax[3].legend(fontsize=13)
        ax[3].set_title('(d)', y=-size, fontweight="bold", fontsize=16)  # increase or decrease y as needed
        ax[3].grid()
    return KS_error_max


class Spline():
    # Initializer
    def __init__(self, x, y, kx, runout='parabolic'):

        # This calculates and initializes the spline

        # Store the values of the knot points
        self.kx = kx
        self.delta = kx[1] - kx[0]
        self.nknots = len(kx)
        self.runout = runout

        # Now, compute the other matrices
        m_from_ky = self.ky_to_M()  # Computes second derivatives from knots
        my_from_ky = np.concatenate([m_from_ky, np.eye(len(kx))], axis=0)
        y_from_my = self.my_to_y(x)
        y_from_ky = y_from_my @ my_from_ky

        # print (f"\nmain:"
        #      f"\ny_from_my  = \n{utils.str(y_from_my)}"
        #      f"\nm_from_ky = \n{utils.str(m_from_ky)}"
        #      f"\nmy_from_ky = \n{utils.str(my_from_ky)}"
        #      f"\ny_from_ky = \n{utils.str(y_from_ky)}"
        #     )

        # Now find the least squares solution
        ky = np.linalg.lstsq(y_from_ky, y, rcond=-1)[0]

        # Return my
        self.ky = ky
        self.my = my_from_ky @ ky

    def my_to_y(self, vecx):
        # Makes a matrix that computes y from M
        # The matrix will have one row for each value of x

        # Make matrices of the right size
        ndata = len(vecx)
        nknots = self.nknots
        delta = self.delta

        mM = np.zeros((ndata, nknots))
        my = np.zeros((ndata, nknots))

        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / delta))
            if j >= self.nknots - 1: j = self.nknots - 2
            if j < 0: j = 0
            x = xx - j * delta

            # Fill in the values in the matrices
            mM[i, j] = -x ** 3 / (6.0 * delta) + x ** 2 / 2.0 - 2.0 * delta * x / 6.0
            mM[i, j + 1] = x ** 3 / (6.0 * delta) - delta * x / 6.0
            my[i, j] = -x / delta + 1.0
            my[i, j + 1] = x / delta

        # Now, put them together
        M = np.concatenate([mM, my], axis=1)

        return M

    # -------------------------------------------------------------------------------

    def my_to_dy(self, vecx):
        # Makes a matrix that computes y from M for a sequence of values x
        # The matrix will have one row for each value of x in vecx
        # Knots are at evenly spaced positions kx

        # Make matrices of the right size
        ndata = len(vecx)
        h = self.delta

        mM = np.zeros((ndata, self.nknots))
        my = np.zeros((ndata, self.nknots))

        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / h))
            if j >= self.nknots - 1: j = self.nknots - 2
            if j < 0: j = 0
            x = xx - j * h

            mM[i, j] = -x ** 2 / (2.0 * h) + x - 2.0 * h / 6.0
            mM[i, j + 1] = x ** 2 / (2.0 * h) - h / 6.0
            my[i, j] = -1.0 / h
            my[i, j + 1] = 1.0 / h

        # Now, put them together
        M = np.concatenate([mM, my], axis=1)

        return M

    # -------------------------------------------------------------------------------

    def ky_to_M(self):

        # Make a matrix that computes the
        A = 4.0 * np.eye(self.nknots - 2)
        b = np.zeros(self.nknots - 2)
        for i in range(1, self.nknots - 2):
            A[i - 1, i] = 1.0
            A[i, i - 1] = 1.0

        # For parabolic run-out spline
        if self.runout == 'parabolic':
            A[0, 0] = 5.0
            A[-1, -1] = 5.0

        # For cubic run-out spline
        if self.runout == 'cubic':
            A[0, 0] = 6.0
            A[0, 1] = 0.0
            A[-1, -1] = 6.0
            A[-1, -2] = 0.0

        # The goal
        delta = self.delta
        B = np.zeros((self.nknots - 2, self.nknots))
        for i in range(0, self.nknots - 2):
            B[i, i] = 1.0
            B[i, i + 1] = -2.0
            B[i, i + 2] = 1.0

        B = B * (6 / delta ** 2)

        # Now, solve
        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B

        # Now, add rows of zeros for M[0] and M[n-1]

        # This depends on the type of spline
        if (self.runout == 'natural'):
            z0 = np.zeros((1, self.nknots))  # for natural spline
            z1 = np.zeros((1, self.nknots))  # for natural spline

        if (self.runout == 'parabolic'):
            # For parabolic runout spline
            z0 = AinvB[0]
            z1 = AinvB[-1]

        if (self.runout == 'cubic'):
            # For cubic runout spline

            # First and last two rows
            z0 = AinvB[0]
            z1 = AinvB[1]
            zm1 = AinvB[-1]
            zm2 = AinvB[-2]

            z0 = 2.0 * z0 - z1
            z1 = 2.0 * zm1 - zm2

        # print (f"ky_to_M:"
        #       f"\nz0 = {utils.str(z0)}"
        #       f"\nz1 = {utils.str(z1)}"
        #       f"\nAinvB = {utils.str(AinvB)}"
        #      )

        # Reshape to (1, n) matrices
        z0 = z0.reshape((1, -1))
        z1 = z1.reshape((1, -1))

        AinvB = np.concatenate([z0, AinvB, z1], axis=0)

        # print (f"\ncompute_spline: "
        #       f"\n A     = \n{utils.str(A)}"
        #       f"\n B     = \n{utils.str(B)}"
        #       f"\n Ainv  = \n{utils.str(Ainv)}"
        #       f"\n AinvB = \n{utils.str(AinvB)}"
        #      )

        return AinvB

    # -------------------------------------------------------------------------------

    def evaluate(self, x):
        # Evaluates the spline at a vector of values
        y = self.my_to_y(x) @ self.my
        return y

    # -------------------------------------------------------------------------------

    def evaluate_deriv(self, x):

        # Evaluates the spline at a vector (or single) point
        y = self.my_to_dy(x) @ self.my
        return y

# ===============================================================================


def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a


def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a


def compute_accuracy(scores_in, labels_in, spline_method, splines, showplots=True, ax=None):

    # Computes the accuracy given scores and labels.
    # Also plots a graph of the spline fit

    # Change to numpy, then this will work
    scores = ensure_numpy (scores_in)
    labels = ensure_numpy (labels_in)

    # Sort them
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    #Accumulate and normalize by dividing by num samples
    nsamples = len(scores)
    integrated_accuracy = np.cumsum(labels) / nsamples
    integrated_scores   = np.cumsum(scores) / nsamples
    percentile = np.linspace (0.0, 1.0, nsamples)

    # Now, try to fit a spline to the accumulated accuracy
    nknots = splines
    kx = np.linspace (0.0, 1.0, nknots)

    error = integrated_accuracy - integrated_scores
    #error = integrated_accuracy

    spline = Spline (percentile, error, kx, runout=spline_method)

    # Now, compute the accuracy at the original points
    dacc = spline.evaluate_deriv (percentile)
    #acc = dacc
    acc = scores + dacc

    # Compute the error
    fitted_error = spline.evaluate (percentile)
    err = error - fitted_error
    stdev = np.sqrt(np.mean(err*err))
    print ("compute_error: fitted spline with accuracy {:.3e}".format(stdev))

    if showplots:
        # Set up the graphs
        if ax is None:
            f, ax = plt.subplots()
        # f.suptitle ("Spline-fitting")

        # (accumualated) integrated_scores and # integrated_accuracy vs sample number
        ax.plot(100.0*percentile, error, label='Error')
        ax.plot(100.0*percentile, fitted_error, label='Fitted error')
        ax.legend()
        # plt.savefig(os.path.join(outdir, plotname) + '_splinefit.png', bbox_inches="tight")
        # plt.close()
    return acc, -fitted_error


def str (A,
         form    = "{:6.3f}",
         iform   = "{:3d}",
         sep     = '  ',
         mbegin  = '  [',
         linesep = ',\n   ',
         mend    = ']',
         vbegin  = '[',
         vend    = ']',
         end     = '',
         nvals   = -1
         ) :
  # Prints a tensorflow or numpy vector nicely
  #
  # List
  if isinstance (A, list) :
    sstr = '[' + '\n'
    for i in A :
      sstr = sstr + str(i) + '\n'
    sstr = sstr + ']'
    return sstr

  elif isinstance (A, tuple) :
    sstr = '('
    for i in A :
      sstr = sstr + str(i) + ', '
    sstr = sstr + ')'
    return sstr

  # Scalar types and None
  elif A is None : return "None"
  elif isinstance (A, float) : return form.format(A)
  elif isinstance (A, int) : return iform.format(A)

  # Othewise, try to see if it is a numpy array, or can be converted to one
  elif isinstance (A, np.ndarray) :
    if A.ndim == 0 :

      sstr = form.format(A)
      return sstr

    elif A.ndim == 1 :

      sstr = vbegin

      count = 0
      for val in A :

        # Break once enough values have been written
        if count == nvals :
          sstr = sstr + sep + "..."
          break

        if count > 0 :
          sstr = sstr + sep
        sstr = sstr + form.format(val)
        count += 1

      sstr = sstr + vend
      return sstr

    else :
      sstr = '['
      for var in A :
        if var.ndim == 2 :
          sstr = sstr + '\n'
        sstr = sstr + str(var)
        if var.ndim == 2 :
          sstr = sstr + '\n'
      sstr = sstr + ']'
      return sstr

  # Now, try things that can be converted to numpy array
  else :
    try :
      temp = np.array (A)
      return str(temp,
                 form    = form,
                 sep     = sep,
                 mbegin  = mbegin,
                 linesep = linesep,
                 mend    = mend,
                 vbegin  = vbegin,
                 vend    = vend,
                 end     = end,
                 nvals   = nvals
                 )

    except :
      return f"{A}"

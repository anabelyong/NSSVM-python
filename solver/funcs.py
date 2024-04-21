# math
import numpy as np
# sparse
from scipy.sparse import csr_matrix
# plot
import matplotlib.pyplot as plt
# time
import time

def randomData(dim, m, n, r):
    """
    Generates random data.
    
    Parameters:
        dim (str): Type of data ('2D', '3D', or 'nD').
        m (int): Number of samples.
        n (int): Number of features.
        r (float): Flipping ratio.
        
    Returns:
        Atr (ndarray): Training samples data, shape (m/2, n).
        ctr (ndarray): Training samples classes, shape (m/2,).
        Ate (ndarray): Testing samples data, shape (m/2, n).
        cte (ndarray): Testing samples classes, shape (m/2,).
    """
    m2 = int(np.ceil(m / 2))

    if dim == '2D':
        A = np.vstack((np.hstack((0.5 + np.sqrt(0.5) * np.random.randn(m2, 1), -3 + np.sqrt(3) * np.random.randn(m2, 1))),
                       np.hstack((-0.5 + np.sqrt(0.5) * np.random.randn(m2, 1), 3 + np.sqrt(3) * np.random.randn(m2, 1)))))
        c = np.hstack((-np.ones(m2), np.ones(m2)))
    elif dim == '3D':
        rho = 0.5 + 0.03 * np.random.randn(m2)
        t = 2 * np.pi * np.random.rand(m2)
        data1 = np.column_stack((rho * np.cos(t), rho * np.sin(t), rho * rho))

        rho = 0.5 + 0.03 * np.random.randn(m2)
        t = 2 * np.pi * np.random.rand(m2)
        data2 = np.column_stack((rho * np.cos(t), rho * np.sin(t), -rho * rho))
        
        A = np.vstack((data1, data2))
        c = np.hstack((np.ones(m2), -np.ones(m2)))
    elif dim == 'nD':
        c = np.ones(m)
        I0 = np.random.permutation(m)
        I = I0[:int(np.ceil(m2))]
        c[I] = -1
        A = np.tile(c[:, np.newaxis] * np.random.rand(m, 1), (1, n)) + np.random.randn(m, n)
    
    np.random.seed(42)
    T = np.random.permutation(m)
    Atr = A[T[:m2], :]
    ctr = c[T[:m2]]
    ctr = filp(ctr, r)

    Ate = A[T[m2:m], :]
    cte = c[T[m2:m]]

    return Atr, ctr, Ate, cte

def filp(fc, r):
    """
    Flips a fraction of the values in fc.
    
    Parameters:
        fc (ndarray): Input array.
        r (float): Flipping ratio.
    
    Returns:
        ndarray: Flipped array.
    """
    if r > 0:
        mc = len(fc)
        np.random.seed(42)
        T0 = np.random.permutation(mc)
        fc[T0[:int(np.ceil(r * mc))]] = -fc[T0[:int(np.ceil(r * mc))]]
    return fc


def accuracy(X, x, y):
    """
    Calculates the accuracy of a linear classifier.
    
    Parameters:
        X (ndarray): Data matrix with shape (m, n), where m is the number of samples and n is the number of features.
        x (ndarray): Coefficients of the linear classifier.
        y (ndarray): True labels for the data samples.
        
    Returns:
        acc (float): Accuracy of the classifier.
        mis (int): Number of misclassified samples.
    """
    if X.size != 0:
        z = np.dot(X, x[:-1]) + x[-1]
        sz = np.sign(z)
        sz[sz == 0] = 1
        mis = np.count_nonzero(sz - y)
        acc = 1 - mis / len(y)
    else:
        acc = np.nan
        mis = np.nan
    
    return acc, mis


def plot2D(Atr, ctr, x, text, acc, ax=None):
    """
    Plot 2D data points with labels and decision boundary.

    Parameters:
        Atr (ndarray): Data points with shape (m, 2).
        ctr (ndarray): Labels for data points.
        x (ndarray): Coefficients of the linear classifier.
        text (str): Text for legend.
        acc (float): Accuracy of the classifier.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.

    Returns:
        None
    """
    siz = 50
    at1 = Atr[ctr == 1]
    at2 = Atr[ctr == -1]
    x0 = np.array([-2, 2])
    y0 = 2.5 * x0

    if ax is None:
        ax = plt.gca()

    ax.scatter(at1[:, 0], at1[:, 1], s=siz, marker='o', c='m', linewidth=1.5)
    ax.scatter(at2[:, 0], at2[:, 1], s=siz, marker='x', c='b', linewidth=1.5)
    ax.plot(x0, y0, color='black', linestyle=':', linewidth=1.5)
    ax.axis([-2, 2, min(np.concatenate([at1[:, 1], at2[:, 1]])), max(np.concatenate([at1[:, 1], at2[:, 1]]))])
    #ax.set_box_on(True)
    ax.grid(True)

    if x is not None:
        y = -x[0] / x[1] * x0 - x[2] / x[1]
        ax.plot(x0, y, color='g', linewidth=1.5)
        ax.legend(['Positive', 'Negative', 'Bayes', text], loc='upper right')
        ax.set_title(f'Accuracy: {acc*100:.2f}%')
        ax.grid(True)
        ax.axis([-2, 2, min(Atr[:, 1]) - 0.1, max(Atr[:, 1]) + 1.5])

def normalization(X, normal_type):
    """
    Normalize the input matrix X.

    Parameters:
        X (ndarray): An (m x n) order matrix to be normalized.
        normal_type (int): Type of normalization:
                           0: No normalization, NX = X
                           1: Sample-wise and then feature-wise normalization.
                           2: Feature-wise scaling to [-1, 1], typically for logistic regression.
                           3: Feature-wise scaling to unit norm columns, typically for CS problem.

    Returns:
        NX (ndarray): Normalized m x n order matrix.
    """
    t0 = time.time()

    if normal_type == 0:
        NX = X
    elif normal_type == 1:
        C = X - np.mean(X, axis=1)[:, np.newaxis]
        Yrow = C / np.std(X, axis=1)[:, np.newaxis]
        Y = Yrow.T
        D = Y - np.mean(Y, axis=1)[:, np.newaxis]
        Ycol = D / np.std(Y, axis=1)[:, np.newaxis]
        NX = Ycol.T
        if np.isnan(NX).any():
            nX = 1 / np.sqrt(np.sum(X * X, axis=0))
            lX = len(nX)
            NX = X @ csr_matrix((nX, (np.arange(lX), np.arange(lX))), shape=(lX, lX))
    else:
        if normal_type == 2:
            nX = 1 / np.max(np.abs(X), axis=0)
        else:
            nX = 1 / np.sqrt(np.sum(X * X, axis=0))

        lX = len(nX)
        if lX <= 10000:
            NX = X @ csr_matrix((nX, (np.arange(lX), np.arange(lX))), shape=(lX, lX))
        else:
            k = 5000
            if np.count_nonzero(X) / lX / lX < 1e-4:
                k = 10000
            K = int(np.ceil(lX / k))

            for i in range(1, K):
                T = slice((i - 1) * k, i * k)
                X[:, T] = X[:, T] @ csr_matrix((nX[T], (np.arange(k), np.arange(k))), shape=(k, k))

            T = slice((K - 1) * k, lX)
            k0 = len(nX[T])
            X[:, T] = csr_matrix(X[:, T]) @ csr_matrix((nX[T], (np.arange(k0), np.arange(k0))), shape=(k0, k0))

            NX = X

    NX[np.isnan(NX)] = 0
    print(f'Normalization used {time.time() - t0:.4f} seconds.')

    return NX
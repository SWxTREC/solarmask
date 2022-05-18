import torch
import numpy as np
import skimage
from scipy.stats import kurtosis, skew


mu0 = 4 * np.pi * 10**-3
k = 1e3
M = 1e6
ds = np.sqrt(1.33e105) * k / M


def norm(data):
    """The one norm of a set of data elements

    Args:
        data (A list, tuple or iterable): A List of elements that can be taken the absolute value of

    Returns:
        The result of summing the application of torch.abs() on all elements of data
    """
    n = 0
    for i in data:
        n += i*i

    return np.sqrt(n)

###### CONSTANTS
gpu_dev = torch.device("cpu") # Change if you have a gpu

def stat_moment_label(prefix: str):
    """Labels the statistical moment with an added prefix

    Args:
        prefix (string): Prefix to add to labels

    Returns:
        tuple: List of labeled elements
    """
    return prefix + "_mean", prefix + "_std", prefix + "_skew", prefix + "_kurt"

def stat_moment(data):
    """The statistical \"moment\" of a distribution

    Args:
        data (np.array): A (usually 2d) array containing a distribution of scalars

        For a dataset X, 

        The mean measures the "center point" of the data X
        $$mean(X) = \\frac{1}{|X|}\\sum_{x \\in X}x$$

        The standard deviation measures the "spread" of the data X
        $$std(X) = \\sqrt{\\frac{1}{|X|}\\sum_{x \\in X}(x - mean(x))^2}$$

        The skew measures the "shift" of the center point of X
        $$skew(X) = \\frac{1}{|X|}\\sum_{x \\in X}(\\frac{x - mean(x)}{std(x)})^3$$

        The kurtosis measures the "taildness" of a X
        $$kurtosis(X) = \\frac{1}{|X|}\\sum_{x \\in X}(\\frac{x - mean(x)}{std(x)})^4 - 3$$

        $$M(X) = (mean(X), std(X), skew(X), kurtosis(X))^T$$

    Returns:
        A pytorch tensor containing average, standard deviation, skew, kurtosis (in that order)
    """
    avg = np.mean(data)
    std = np.std(data)
    #skw = np.mean(((data - avg)/std)**3)
    #krt = np.mean(((data - avg)/std)**4) - 3.0
    krt = kurtosis(data)
    skw = skew(data)
    return np.array([avg, std, skw, krt])


# The kernel to use greens function (radius 10)
radius = 10
dz = 0.001
dist_kern = torch.zeros((2*radius + 1, 2*radius + 1))
for x0 in range(radius + 1):
    for y0 in range(radius + 1):
        if norm((x0, y0)) <= radius:
            v = dz / norm((x0, y0, dz))
            dist_kern[radius + x0][radius + y0] = v
            dist_kern[radius - x0][radius + y0] = v
            dist_kern[radius + x0][radius - y0] = v
            dist_kern[radius - x0][radius - y0] = v



def gradient(data):
    """Numerical Gradient of data (my own - not numpy's)

    Args:
        data (2d np.array): A 2d array of scalars

    Returns:
        (np.array, np.array): 2 arrays representing dy, dx respectively
    """
    return np.gradient(data)

    """
    retrows = np.zeros(data.shape)
    retcols = np.zeros(data.shape)

    retrows[1:-1,:] = data[2:,:] - data[:-2,:]

    retrows[0] = (-3 * data[0] + 4*data[1] - data[2])
    retrows[-1] = -(-3 * data[-1] + 4*data[-2] - data[-3])

    retcols[:,1:-1] = data[:,2:] - data[:,:-2]
    retcols[:,0] = (-3 * data[:,0] + 4*data[:,1] - data[:,2])
    retcols[:,-1] = -(-3 * data[:,-1] + 4*data[:,-2] - data[:,-3])

    return 0.5 * retrows, 0.5 * retcols
    """

def cov(m, rowvar=False):
    """Covariance of a dataset m

    Args:
        m (2d array): Two lists of data (x, y) to take the covariance from

    Returns:
        Covariance of the two data sets
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= np.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def unpad(Z):
    """Removes surrounding whitespace (False values) from a binary array Z

    Args:
        Z (2d array): A binary array

    Returns:
        Same type as Z: Z without the surrounding values False
    """
    x, y = np.where(Z)
    return Z[np.min(x):np.max(x)+1,np.min(y):np.max(y)+1]



def shannon_entropy(Z):
    """Computes shannon entropy of a binary array Z

    Args:
        Z (2d array): A binary array

    Returns:
        float: The shannon entropy of array Z
    """
    return skimage.measure.shannon_entropy(unpad(Z))


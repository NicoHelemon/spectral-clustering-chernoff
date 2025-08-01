# src/utils/utils.py
# Utility functions for spectral clustering and data processing.

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import scipy.sparse
from typing import Optional, Union, Sequence, Any, Tuple
from scipy.stats import rv_discrete, rv_continuous

def spectral_embedding(A: Union[np.ndarray, scipy.sparse.spmatrix], 
					   d: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Calculate the spectral embedding of the matrix A.\n
		Parameters
		----------
		A : NDArray[np.float64, scipy.sparse.spmatrix]
			The input matrix, shape (n, n), dense or sparse.
		d : int
			Number of dimensions for the embedding.\n
		Returns
		-------
		X : NDArray[np.float64]
			The spectral embedding of the matrix, a 2-D array of shape (n, d).
		Λ : NDArray[np.float64]
			The eigenvalues of the matrix, a 1-D array of length d."""

		n = A.shape[0]
		if d > n:
			raise ValueError(f'd must be ≤ number of nodes ({n}), got {d}.')

		# Choose solver depending on sparsity and spectrum size
		if scipy.sparse.issparse(A):
			A_sp = A.asfptype() #type: ignore
			if d < n:
				vals, vecs = eigsh(A_sp, k=d, which='LM')
			else:
				A_dense = A_sp.toarray()
				vals, vecs = eigh(A_dense)
		else:
			A_arr = A.astype(np.float32) #type: ignore
			if d < n:
				vals, vecs = eigsh(A_arr, k=d, which='LM')
			else:
				vals, vecs = eigh(A_arr)

		# #Ensure the eigenvectors are in the same direction
		# #by aligning them with the largest absolute value in each column
		# j = np.argmax(np.abs(vecs), axis=0)
		# signs = np.sign(vecs[j, np.arange(vecs.shape[1])])
		# vecs *= signs[None, :]

		return vecs * np.sqrt(np.abs(vals)), vals #type: ignore


def mask_outliers(X: NDArray[np.float64],
				  q_outliers: float) -> NDArray[np.bool_]:
	"""Return a boolean mask masking outliers (i.e., points that are in the top q_outliers quantile of distances from the mean).\n
	Parameters
	----------
	X : NDArray[np.float64]
		The data points, a 2-D array of shape (n, d).
	q_outliers : float
		The quantile threshold for outliers, between 0 and 1.\n
	Returns
	-------
	mask : NDArray[np.bool_]
		A boolean mask of shape (n,) where True indicates a point is not an outlier."""
	
	assert 0 <= q_outliers <= 1, "q_outliers must be between 0 and 1"

	distances = np.linalg.norm(X - X.mean(axis=0), axis=1)
	idx_outliers = np.argsort(distances)[-int(len(X) * q_outliers):] if q_outliers > 0 else []
	mask = np.ones(len(X), dtype=bool)
	mask[idx_outliers] = False

	return mask


def sigmoid(x: float, x0: float, k: float) -> float:
	"""Compute the sigmoid function with a given slope.\n
	Parameters
	----------
	x : float
		The input value.
	x0 : float
		The x-value at which the sigmoid function is centered.
	k : float
		The slope of the sigmoid function.\n
	Returns
	-------
	float
		The value of the sigmoid function at x."""
	z = k*(x - x0)
	z = np.clip(z, -700, +700)
	return 1/(1 + np.exp(-z))

def step(x: float, x0: float) -> float:
	"""Compute a step function that returns 1 if x > x0, else 0.\n
	Parameters
	----------
	x : float
		The input value.
	x0 : float
		The threshold value.\n
	Returns
	-------
	NDArray[np.int32]
		An array with 1 if x > x0, else 0."""

	return 1 if x > x0 else 0

def sigmoid_w95(x: float, x0: float, w: float) -> float:
	"""Compute a sigmoid function with a specific width at 95% of the maximum value.\n
	Parameters
	----------
	x : float
		The input value.
	x0 : float
		The x-value at which the sigmoid function is centered.
	w : float
		The width of the sigmoid function at 95% of its maximum value.\n
	Returns
	-------
	float
		The value of the sigmoid function at x, adjusted for the specified width."""
	if w == 0:
		return step(x, x0)
	else:
		return sigmoid(x, x0, np.log(19)/w)
	

def procrustes_align(Source: NDArray[np.float64],
					 Target: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
	"""Align the Source matrix to the Target matrix using Procrustes analysis.\n
	Parameters
	----------
	Source : NDArray[np.float64]
		The source matrix to be aligned, a 2-D array of shape (n, d).
	Target : NDArray[np.float64]
		The target matrix to align to, a 2-D array of shape (n, d).\n
	Returns
	-------
	R : NDArray[np.float64]
		The rotation matrix that aligns the Source to the Target.
	beta : float
		The scaling factor applied to the Source matrix during alignment."""
	
	S, T = Source, Target

	S0 = S - S.mean(axis=0)
	T0 = T - T.mean(axis=0)

	R, sigma = orthogonal_procrustes(S0, T0)
	beta = sigma / np.linalg.norm(S0, 'fro')**2

	return R, beta

class ConstantDist:
	"""The constant distribution."""

	def __init__(self, value: float):
		"""Initialize a constant distribution with a fixed value."""
		self.value = value
		xk = np.array([value])
		pk = np.array([1.0])
		self.dist = rv_discrete(name='constant', values=(xk, pk))

	def rvs(self, size=None, random_state=None):
		if size is None:
			return self.value
		return np.full(size, self.value)

	def support(self):
		return (self.value, self.value)

	def mean(self):
		return self.value

def summarize_dist(g: Union[rv_discrete, rv_continuous, ConstantDist]) -> Tuple[str, Any, Any]:
	"""Summarize a distribution by its name, shape arguments, and parameters.\n
	Parameters
	----------
	g : Union[rv_discrete, rv_continuous, ConstantDist]
		The distribution to summarize.\n
	Returns
	-------
		tuple[str, Any, Any]
			A tuple containing the distribution name, shape arguments, and parameters."""
	if isinstance(g, ConstantDist):
		return ("constant", g.value, ())

	# scipy rv_continuous and rv_discrete store their init kwargs in .kwds
	if hasattr(g, "name") and hasattr(g, "kwds"):
		# e.g. ("beta", (("a", 2.3), ("b", 1.0)), (("loc",0),("scale",1)))
		dist_params = tuple(sorted(g.kwds.items())) # type: ignore
		# some distributions also take shape arguments in g.args
		shape_args = tuple(g.args) if hasattr(g, "args") else () # type: ignore
		return (g.name, shape_args, dist_params)

	raise TypeError(f"Don't know how to summarize {g!r}")

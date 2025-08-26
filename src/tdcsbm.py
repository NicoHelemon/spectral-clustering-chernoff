# src/twsbm.py
# Contains the TWSBM class, which models transformed weighted stochastic block models
# and computes a range of structural and embedding-based metrics.

from .models.wsbm import WSBM
#from .models.wdcsbm import WDCSBM, GInputType
from .transformations import *
from .metrics import *
from .utils.utils import *
from .utils.EGMM import *

import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

#TODO Harmonize X vs X_A
#TODO Use μ rather than M for the mean of the embedding?

class TDCSBM:
	"""Class representing a transformed WDCSBM instance."""

	def __init__(self,
				 A: NDArray[np.float64], 
				 Z: NDArray[np.int32],
				 K: int,
				 X_transformation: Optional[str] = None):
		"""Initialize a TWCSBM instance.\n
		Parameters
		----------
		A : NDArray[np.float64]
			The transformed weight matrix sampled, a 2-D array of shape (n, n).
		Z : NDArray[np.int32]
			The community labels of the sample, a 1-D array of length n.
		K : Optional[int]
			The number of communities in the model. If None, it is inferred from the model.
		X_transformation : Optional[str]
			The transformation applied to the embedding. One of {None, 'normalised', 'theta', 'score'}.
		"""

		if X_transformation not in [None, 'normalised', 'theta', 'score']:
			raise ValueError("ValueError: 'transformation' is not in the list of admissible models.")

		self.A = A
		self.Z = Z
		self.K = K
		self.X_transformation = X_transformation

		if len(np.unique(Z)) > K:
			raise ValueError(f"Number of unique communities in Z ({len(np.unique(Z))}) exceeds K ({K})")

		self.X_A, _ = spectral_embedding(A, self.K)
		if np.isnan(self.X_A).any():
			print("Warning: NaN values found in the spectral embedding of A")
			self.X_A = np.nan_to_num(self.X_A)
			print("NaN values replaced with 0")
		self.Z_hat = self.fit_predict_egmm(self.X_A, self.K, self.X_transformation)

		# Metrics
		self.ARI = ARI(Z, self.Z_hat)
		

	def __eq__(self, other) -> bool:
		if not isinstance(other, TDCSBM):
			return False
		return (np.array_equal(self.A, other.A) and
				np.array_equal(self.Z, other.Z) and
				self.K == other.K and
				self.X_transformation == other.X_transformation)
	
	def __hash__(self) -> int:
		return hash((self.A.tobytes(), self.Z.tobytes(), self.K, self.X_transformation))
	
	def __reduce__(self) -> tuple[type, tuple]:
		return (TDCSBM, (self.A, self.Z, self.K, self.X_transformation))
	
	@staticmethod
	def fit_predict_egmm(X: NDArray[np.float64], 
					  	 K: int,
						 X_transformation: Optional[str] = None) -> NDArray[np.int32]:
		"""Fit and predict using the EGMM model.\n
		Parameters
		----------
		X : NDArray[np.float64]
			The spectral embedding of the weight matrix, a 2-D array of shape (n, d).
		K : int
			Number of components for the GMM.
		transformation : Optional[str]
			The transformation applied to the embedding. One of {None, 'normalised', 'theta', 'score'}.
		Returns
		-------
		Z_hat : NDArray[np.int32]
			1-D array of length n, where Z_hat[i] is the estimated community index (0 to K-1) for node i.
		"""

		egmm = EGMM(K)
		Z_hat = egmm.fit_predict_approximate(X, d = K, transformation=X_transformation)

		return Z_hat

		

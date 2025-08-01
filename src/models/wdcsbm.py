# src/models/wsbm.py
# Defines the base class for Weighted Stochastic Block Models (WSBM).

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Sequence, List
from scipy.stats import rv_continuous, rv_discrete
from abc import ABC
from abc import abstractmethod

from ..transformations import WeightTransform
from ..utils.utils import *

DistType = Union[rv_continuous, rv_discrete, "ConstantDist"]
GInputType = Union[float, Sequence[DistType]]

class WDCSBM(ABC):
	"""Base class for Weighted Degree-Corrected Stochastic Block Models (WDCSBM)."""

	def __init__(self,
				 K: int,
				 G: GInputType,
				 π: NDArray[np.float64], 
				 n: int):
		""" Initialize a WSBM instance.
		Parameters
		----------
		K : int
			Number of communities (blocks).
		G : Union[float, Sequence[DistType]]
			Probability of an edge existing between any two nodes or a sequence of K distributions for degree correction.
		π : NDArray[np.float64]
			1-D array of length K representing the block proportions.
		n : int
			Total number of nodes in the WSBM."""
		
		# K must be at least 2
		if K < 2:
			raise ValueError(f"K must be ≥ 2, got {K}")

		# G must be a float in [0,1] or a sequence of distributions
		if isinstance(G, float):
			if not (0.0 < G <= 1.0):
				raise ValueError(f"G must be between 0 and 1, got {G}")
			self.G: Sequence[DistType] = [ConstantDist(np.sqrt(G)) for _ in range(K)]
		else:
			if len(G) != K: # type: ignore
				raise ValueError(f"Must pass either one float or a sequence of length K={K}, got len(G)={len(G)}") # type: ignore
			for g in G: # type: ignore
				if not hasattr(g, "rvs"):
					raise ValueError("Each element of G must have an .rvs(...) method")
				low, high = g.support()
				if low < 0 or high > 1:
					raise ValueError(f"Each distribution in G must have support in [0, 1], got support {low}, {high}")
			self.G: Sequence[DistType] = list(G) # type: ignore

		# π must be a 1-dimensional array of length K
		if π.ndim != 1:
			raise ValueError(f"π must be 1-D, got {π.ndim}-D array")
		if π.shape[0] != K:
			raise ValueError(f"Length of π must equal K={K}, got {π.shape[0]}")

		# π must sum to 1 (within a tiny tolerance)
		total = π.sum()
		if not np.isclose(total, 1.0):
			raise ValueError(f"Entries of π must sum to 1, but sum is {total}")

		# n must be a non-negative integer
		if n < 0:
			raise ValueError(f"n must be non-negative, got {n}")

		self.K = K
		self.π = π
		self.n = n

		self.EG = np.array([g.mean() for g in self.G]) # type: ignore
		self.EGG = np.outer(self.EG, self.EG) # type: ignore

	def _sample_communities(self, 
							seed: Optional[int] = None) -> NDArray[np.int32]:
		"""Sample community assignments for n nodes.\n
		Parameters
		----------
		seed : Optional[int]
			Random seed for reproducibility. If None, uses the current random state.\n
		Returns
		-------
		Z : NDArray[np.int32]
			1-D array of length n, where Z[i] is the community index (0 to K-1) for node i."""

		np.random.seed(seed)
		Z = np.random.choice(np.arange(self.K), 
					   		 size=self.n, 
					   		 p=self.π)
		return Z

	def _sample_degree_corrections(self, 
								  Z: NDArray[np.int32], 
							 	  seed: Optional[int] = None) -> NDArray[np.float64]:
		"""Sample degree correction factors based on community assignments.\n
		Parameters
		----------
		Z : NDArray[np.int32]
			1-D array of length n, where Z[i] is the community index for node i.
		seed : Optional[int]
			Random seed for reproducibility. If None, uses the current random state.\n
		Returns
		-------
		W : NDArray[np.float64]
			1-D array of length n, where W[i] is the degree correction factor for node i."""
		
		np.random.seed(seed)
		W = np.array([self.G[z].rvs() for z in Z]) # type: ignore
		return W

	@abstractmethod
	def _sample_edge_weights(self, 
							 Z: NDArray[np.int32],
							 W: NDArray[np.float64],
							 seed: Optional[int] = None) -> NDArray[np.float64]:
		"""Sample edge weights based on community assignments.\n
		Parameters
		----------
		Z : NDArray[np.int32]
			1-D array of length n, where Z[i] is the community index for node i.
		W : NDArray[np.float64]
			1-D array of length n, where W[i] is the degree correction factor for node i.
		seed : Optional[int]
			Random seed for reproducibility. If None, uses the current random state.\n
		Returns
		-------
		A : NDArray[np.float64]
			2-D array of shape (n, n) representing the sampled edge weights.
			Diagonal entries are 0, and A[i, j] == A[j, i] is the weight of the edge between nodes i and j."""
		pass

	def sample(self, 
			   seed: Optional[int] = None) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
		"""Sample a WSBM instance.\n
		Parameters
		----------
		seed : Optional[int]
			Random seed for reproducibility. If None, uses the current random state.\n
		Returns
		-------
		A : NDArray[np.float64]
			2-D array of shape (n, n) representing the sampled edge weights.
			Diagonal entries are 0, and A[i, j] == A[j, i] is the weight of the edge between nodes i and j.\n
		Z : NDArray[np.int32]
			1-D array of length n, where Z[i] is the community index (0 to K-1) for node i.
		W : NDArray[np.float64]
			1-D array of length n, where W[i] is the degree correction factor for node i."""
		
		Z = self._sample_communities(seed)
		W = self._sample_degree_corrections(Z, seed)
		A = self._sample_edge_weights(Z, W, seed)
		return A, Z, W

	def theoretical_mean_variance(self,
					    T: WeightTransform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Calculate the theoretical block mean and variance of the model.\n
		Parameters
		----------
		T : WeightTransform
			The applied weight transformation.\n
		Returns
		-------
		B : NDArray[np.float64]
			Theoretical mean of the edge weights after transformation, a 2-D array of shape (K, K).
		C : NDArray[np.float64]
			Theoretical variance of the edge weights after transformation, a 2-D array of shape (K, K)."""
		from .bc_registry import theoretical_mean_variance
		return theoretical_mean_variance(self, T)
	
	def theoretical_mean(self,
					     T: WeightTransform) -> NDArray[np.float64]:
		"""Calculate the theoretical block mean of the model after applying a weight transformation.\n
		Parameters
		----------
		T : WeightTransform
			The applied weight transformation.\n
		Returns
		-------
		B : NDArray[np.float64]
			Theoretical mean of the edge weights after transformation, a 2-D array of shape (K, K)."""
		return self.theoretical_mean_variance(T)[0]

	def theoretical_variance(self,
						     T: WeightTransform) -> NDArray[np.float64]:
		"""Calculate the theoretical block variance of the model after applying a weight transformation.\n
		Parameters
		----------
		T : WeightTransform
			The applied weight transformation.\n
		Returns
		----------
		C : NDArray[np.float64]
			Theoretical variance of the edge weights after transformation, a 2-D array of shape (K, K)."""
		return self.theoretical_mean_variance(T)[1]

	@staticmethod
	def param_matrix_check(param_matrix : NDArray[np.float64], 
						   param_matrix_name: str, 
						   K: int,
						   must_be_positive: bool = True) -> None:
		"""Check the validity of a parameter matrix for WSBM.\n
		Parameters
		----------
		param_matrix : NDArray[np.float64]
			The parameter matrix to check, expected to be of shape (K, K).
		param_matrix_name : str
			Name of the parameter matrix for error messages.
		K : int
			Number of blocks (communities) in the WSBM.
		must_be_positive : bool
			If True, checks that all entries of the matrix are positive.
			Default is True.\n
		Raises
		------
		ValueError"""
		
		if K != 2:
			raise ValueError(f"Currently only K=2 is supported, got K={K}")

		if param_matrix.shape != (K, K):
			raise ValueError(f"{param_matrix_name} must be of shape ({K}, {K}), got {param_matrix.shape}")
		
		if not np.allclose(param_matrix, param_matrix.T):
			raise ValueError(f"{param_matrix_name} must be symmetric")

		if must_be_positive and np.any(param_matrix < 0):
			raise ValueError(f"All entries of {param_matrix_name} must be > 0")
		
	@abstractmethod
	def model_name_with_law_params(self) -> str:
		"""Return the model name with its law parameters."""
		pass
	
	@abstractmethod
	def param_matrix_str(self) -> str:
		"""Return a string representation of the parameter matrix."""
		pass
	
	@abstractmethod
	def __str__(self) -> str:
		"""Return a string representation of the WSBM instance."""
		pass
	
	@abstractmethod
	def __eq__(self, other) -> bool:
		"""Check equality with another WSBM instance."""
		pass
	
	@abstractmethod
	def __hash__(self) -> int:
		"""Return a hash value for the WSBM instance."""
		pass
	
	@abstractmethod
	def __reduce__(self) -> tuple[type, tuple]:
		"""Return a tuple for pickling the WSBM instance."""
		pass

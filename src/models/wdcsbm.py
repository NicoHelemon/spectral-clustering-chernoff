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

class WDCSBM(ABC):
	"""Base class for Weighted Degree-Corrected Stochastic Block Models (WDCSBM)."""

	def __init__(self,
				 K: int,
				 H: NDArray[np.object_],
				 G: Union[float, NDArray[np.object_]],
				 π: NDArray[np.float64], 
				 n: int):
		""" Initialize a WSBM instance.
		Parameters
		----------
		K : int
			Number of communities (blocks).
		H : NDArray[np.object_]
			2-D array of shape (K, K) where H[i, j] is the distribution for edge weights between communities i and j.
		G : Union[float, NDArray[np.object_]]
			Probability of an edge existing between any two nodes or a sequence of K distributions for degree correction.
		π : NDArray[np.float64]
			1-D array of length K representing the block proportions.
		n : int
			Total number of nodes in the WSBM."""
		
		# K must be at least 2
		if K < 2:
			raise ValueError(f"K must be ≥ 2, got {K}")
		
		if H.shape != (K, K):
			raise ValueError(f"H must be a 2-D array of shape ({K}, {K}), got {H.shape}")
		
		if any(not callable(getattr(h, "rvs", None)) for h in H.ravel()):
			raise ValueError("H must contain only distributions with an .rvs(...) method")

		# G must be a float in [0,1] or a sequence of distributions
		if isinstance(G, float):
			if not (0.0 < G <= 1.0):
				raise ValueError(f"G must be between 0 and 1, got {G}")
			self.G: NDArray[np.object_] = np.array([ConstantDist(np.sqrt(G)) for _ in range(K)])
		else:
			if len(G) != K: # type: ignore
				raise ValueError(f"Must pass either one float or a sequence of length K={K}, got len(G)={len(G)}") # type: ignore
			for g in G: # type: ignore
				if not hasattr(g, "rvs"):
					raise ValueError("Each element of G must have an .rvs(...) method")
				low, high = g.support()
				if low < 0 or high > 1:
					raise ValueError(f"Each distribution in G must have support in [0, 1], got support {low}, {high}")
			self.G: NDArray[np.object_] = list(G) # type: ignore

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
		self.H = H
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
		
		#np.random.seed(seed)
		
		#A = np.zeros((self.n, self.n))
		#sampled_edges_idx = np.random.rand(self.n, self.n) < np.outer(W, W)

		#A[sampled_edges_idx] = np.array(
		#	[d.rvs() for d in self.H[np.ix_(Z, Z)][sampled_edges_idx]], 
		#	dtype=float)

		np.random.seed(seed)
		n, K = self.n, self.K
		A = np.zeros((n, n), float)

		sampled_edges_idx  = np.random.rand(n, n) < np.outer(W, W)
		community_pair_idx = (Z[:, None] * K + Z[None, :]).ravel()
		I = np.flatnonzero(sampled_edges_idx.ravel())

		R = A.ravel()
		for c_i in np.unique(community_pair_idx[I]):
			idx = I[community_pair_idx[I] == c_i]
			k, l = divmod(c_i, K)
			R[idx] = self.H[k, l].rvs(size=idx.size) # type: ignore
		
		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)

		return A # type: ignore

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
	

	
	#@abstractmethod
	#def __str__(self) -> str:
	#	"""Return a string representation of the WSBM instance."""
	#	pass
	
	#@abstractmethod
	#def __eq__(self, other) -> bool:
	#	"""Check equality with another WSBM instance."""
	#	pass
	
	#@abstractmethod
	#def __hash__(self) -> int:
	#	"""Return a hash value for the WSBM instance."""
	#	pass
	
	#@abstractmethod
	#def __reduce__(self) -> tuple[type, tuple]:
	#	"""Return a tuple for pickling the WSBM instance."""
	#	pass

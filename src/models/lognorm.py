# src/models/lognorm.py
# Defines the LognormWSBM class, a specific implementation of WSBM with log-normal edge weights.

from .wsbm import WSBM
#from .wdcsbm import WDCSBM, GInputType
import numpy as np
from numpy.typing import NDArray
from scipy.stats import lognorm, norm
from typing import Optional

from ..utils.string_utils import *
from ..utils.utils import *


class LognormWSBM(WSBM):
	name = 'LogN'
	param_string = 'Σ'

	def __init__(self, 
				 K: int, 
				 ρ: float, 
				 π: NDArray[np.float64], 
				 n: int, 
				 Σ: NDArray[np.float64], 
				 μ: float):
		""" Initialize a Lognorm instance.
		Parameters
		----------
		K : int
			Number of communities (blocks).
		ρ : float
			Probability of an edge existing between any two nodes.
		π : NDArray[np.float64]
			1-D array of length K representing the block proportions.
		n : int
			Total number of nodes in the WSBM.
		Σ : NDArray[np.float64]
			2-D array of shape (K, K) representing the covariance of the underlying Gaussian distributions of the lognormal distribution.
			Σ[i, j] == Σ[j, i] is the parameter for edges between nodes in community i and community j.
		μ : float
			The mean of the underlying Gaussian distributions of the lognormal distribution.
			Note: The mean accross the communities is constant, i.e., M_{i,j} = μ for all i,j (with a slight adjustment to avoid singularities).
		"""
		
		super().__init__(K, ρ, π, n)
		
		WSBM.param_matrix_check(Σ, "Σ", K)
		
		self.Σ = Σ
		self.μ = μ
		# Avoid singularities in the lognormal distribution
		self.M = np.log(np.exp(μ) * np.ones((K, K)) - 1e-6 * (1 - np.eye(K)))

	def _sample_edge_weights(self, 
							 Z: NDArray[np.int32], 
							 seed: Optional[int] = None) -> NDArray[np.float64]:
		
		np.random.seed(seed)
		Z_i, Z_j = Z[:, None], Z[None, :]
		
		# Mixture model (1-ρ)δ_0 + ρLognorm(Σ_{Z_i,Z_j}, exp(M)_{Z_i,Z_j})
		A = np.zeros((self.n, self.n))
		sampled_edges_idx = np.random.rand(self.n, self.n) < self.ρ

		s = self.Σ[Z_i, Z_j][sampled_edges_idx]
		scale = (np.exp(self.M))[Z_i, Z_j][sampled_edges_idx]
		A[sampled_edges_idx] = lognorm.rvs(s=s, scale=scale, size=s.shape)
		
		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)

		return A
	
	def model_name_with_law_params(self):
		return f'{LognormWSBM.name}(μ = {self.μ:.2f}, σ)'
	
	def param_matrix_str(self) -> str:
		return param_matrix_str(self.Σ, LognormWSBM.param_string)
	
	def __str__(self) -> str:
		return f'{self.model_name_with_law_params()}, {model_base_parameters_str(self.n, self.ρ, self.π)}'
	
	def __eq__(self, other) -> bool:
		return (isinstance(other, LognormWSBM) and 
				self.K == other.K and 
				self.ρ == other.ρ and 
				np.array_equal(self.π, other.π) and 
				self.n == other.n and 
				np.array_equal(self.Σ, other.Σ) and 
				self.μ == other.μ)
	
	def __hash__(self) -> int:
		return hash((LognormWSBM, 
			   		 self.K, 
					 self.ρ, 
					 tuple(self.π), 
					 self.n, 
					 tuple(map(tuple, self.Σ)),
					 self.μ))
	
	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.K, self.ρ, self.π, self.n, self.Σ, self.μ))
	
	@staticmethod
	def mu_for_quantile_at_zero(σ: float, 
				   				quantile: float = 0.99) -> float:
		"""
		Compute the mean μ of a Normal(μ, σ) such that 0 is at the given upper quantile.\n
		Parameters
		----------
		σ : float
			Standard deviation of the normal distribution.
		quantile : float, optional
			Upper quantile for 0 (default is 0.99).\n
		Returns
		-------
		float
			The mean μ satisfying P(X ≤ 0) = quantile for X ~ Normal(μ, σ).
		"""
		
		return float(- norm.ppf(quantile) * σ)
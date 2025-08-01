# src/models/beta.py
# Defines the BetaWSBM class, a specific implementation of WSBM with Beta-distributed edge weights.

from .wsbm import WSBM
#from .wdcsbm import WDCSBM, GInputType
import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta
from typing import Optional

from ..utils.string_utils import *
from ..utils.utils import *

class BetaWSBM(WSBM):
	"""Weighted Stochastic Block Model with Beta-distributed edge weights."""

	name = 'Beta'
	param_string = 'α'

	def __init__(self, 
				 K: int, 
				 ρ: float, 
				 π: NDArray[np.float64], 
				 n: int, 
				 α: NDArray[np.float64]):
		""" Initialize a BetaWSBM instance.
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
		α : NDArray[np.float64]
			2-D array of shape (K, K) representing the parameters of the Beta distribution for edge weights.
			α[i, j] == α[j, i] is the shape parameter for edges between nodes in community i and community j.
			Note: β is fixed to 1 in this model."""


		super().__init__(K, ρ, π, n)
		
		WSBM.param_matrix_check(α, "α", K)
		#WSBM.param_matrix_check(np.array([[β]]), "β", 1)
		
		self.α = α
		self.β = 1

	def _sample_edge_weights(self, 
							 Z: NDArray[np.int32], 
							 seed: Optional[int] = None) -> NDArray[np.float64]:
		
		np.random.seed(seed)
		Z_i, Z_j = Z[:, None], Z[None, :]
		
		# Mixture model (1-ρ)δ_0 + ρBeta(α_{Z_i,Z_j}, β)
		A = np.zeros((self.n, self.n))
		sampled_edges_idx = np.random.rand(self.n, self.n) < self.ρ
		A[sampled_edges_idx] = beta.rvs(self.α[Z_i, Z_j][sampled_edges_idx], self.β)
		
		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)

		return A

	def model_name_with_law_params(self) -> str:
		return f'{BetaWSBM.name}(α, β = {self.β})'
	
	def param_matrix_str(self) -> str:
		return param_matrix_str(self.α, BetaWSBM.param_string)
	
	def __str__(self) -> str:
		return f'{self.model_name_with_law_params()}, {model_base_parameters_str(self.n, self.ρ, self.π)}'
	
	def __eq__(self, other) -> bool:
		return (isinstance(other, BetaWSBM) and 
				self.K == other.K and 
				self.ρ == other.ρ and 
				np.array_equal(self.π, other.π) and 
				self.n == other.n and 
				np.array_equal(self.α, other.α) and 
				self.β == other.β)
	
	def __hash__(self) -> int:
		return hash((BetaWSBM,
			   		 self.K, 
					 self.ρ,
					 tuple(self.π),
					 self.n,
					 tuple(map(tuple, self.α)),
					 self.β))
	
	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.K, self.ρ, self.π, self.n, self.α))
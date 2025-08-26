# src/transformations.py
# This file contains various edge weight transformations for the Weighted Stochastic Block Model (WSBM).

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.stats import rankdata
from matplotlib import colors
from scipy.sparse import csr_matrix, issparse
from typing import Union

cmap_blue  = colors.LinearSegmentedColormap.from_list("blue", ["#00FFFF", "#0050FF"])
norm_pow_γ = colors.Normalize(vmin=0.5, vmax=2.0) #magic numbers

cmap_pink  = colors.LinearSegmentedColormap.from_list("pink_red", ["#FFB2C6", "#FF0000"])
norm_qtl_q = colors.Normalize(vmin=0.01, vmax=0.5) #magic numbers

cmap_green = colors.LinearSegmentedColormap.from_list("green", ["#00FF00", "#006400"])
norm_logp  = colors.Normalize(vmin=0.5, vmax=2.0) #magic numbers

cmap_purple = colors.LinearSegmentedColormap.from_list("purple", ["#E6E6FA", "#800080"])
norm_qthr 	= colors.Normalize(vmin=0.01, vmax=0.5) #magic numbers

class WeightTransform(ABC):
	"""Abstract class for edge weight transformations in WSBM."""
	@abstractmethod
	def __init__(self):
		""" Initialize the weight transformation."""
		pass

	@abstractmethod
	def __call__(self, A: Union[NDArray[np.float64], csr_matrix]) -> Union[NDArray[np.float64], csr_matrix]:
		""" Apply the transformation to the weight matrix A.
		Parameters
		----------
		A : Union[NDArray[np.float64], csr_matrix]
			The input weight matrix, of shape (n, n), dense or sparse.\n
		Returns
		-------
		tA : NDArray[np.float64]
			The transformed weight matrix, of shape (n, n), dense or sparse."""
		pass
	
	@abstractmethod
	def __eq__(self, other) -> bool:
		""" Check equality with another transformation."""
		pass

	@abstractmethod
	def __hash__(self) -> int:
		""" Return a hash value for the transformation."""
		pass

	@abstractmethod
	def __reduce__(self) -> tuple[type, tuple]:
		""" Return a tuple for pickling the transformation."""
		pass

class ElementwiseTransform(WeightTransform):
	"""Abstract class for elementwise transformations of edge weights."""
	def __call__(self, A: Union[NDArray[np.float64], csr_matrix]) -> Union[NDArray[np.float64], csr_matrix]:
		""" Apply the elementwise transformation to the weight matrix A.
		Parameters
		----------
		A : Union[NDArray[np.float64], csr_matrix]
			The input weight matrix, of shape (n, n), dense or sparse.\n
		Returns
		-------
		tA : Union[NDArray[np.float64], csr_matrix]
			The transformed weight matrix, of shape (n, n), dense or sparse."""
		if issparse(A):
			A = A.copy().tocsr() # type: ignore
			A.data = self.f_data(A.data) # type: ignore
			A.eliminate_zeros() # type: ignore
			return A
		else:
			tA = A.copy()
			flat = tA.flat
			flat[:] = self.f_data(flat) # type: ignore
			return tA
		
	@abstractmethod
	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Apply the transformation to the data of the sparse matrix."""
		pass

class IdentityTransform(ElementwiseTransform):
	"""Identity transformation: returns the input matrix unchanged."""

	def __init__(self):
		self.name = "Identity"
		self.id = 'Id'
		self.color = cmap_blue(norm_pow_γ(1))

	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		return d.copy()
	
	def __eq__(self, other) -> bool:
		return isinstance(other, IdentityTransform)

	def __hash__(self) -> int:
		return hash(IdentityTransform)

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, ())
	
class OppositeTransform(ElementwiseTransform):
	"""Opposite transformation: returns 1 - A for positive elements, 0 otherwise."""

	def __init__(self):
		self.name = "Opposite"
		self.id = 'Opp'
		self.color = 'orange'
	
	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = np.clip(d, 0, None)
		td[td > 0] = 1 - td[td > 0]
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, OppositeTransform)

	def __hash__(self) -> int:
		return hash(OppositeTransform)

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, ())

class LogTransform(ElementwiseTransform):
	"""Logarithmic transformation: returns -log(A) for positive elements, 0 otherwise."""

	def __init__(self):
		self.name = "Logarithmic"
		self.id = 'Log'
		self.color = 'green'
	
	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = np.clip(d, 0, None)
		td[td > 0] = -np.log(td[td > 0])
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, LogTransform)

	def __hash__(self) -> int:
		return hash(LogTransform)

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, ())
	
class LogPowerTransform(ElementwiseTransform):
	"""Logarithmic power transformation: returns -log(A) ** γ 
	for positive elements, 0 otherwise."""

	def __init__(self, γ: float = 1.41):
		"""Initialize the logarithmic power transformation with a parameter γ.\n
		Parameters
		----------
		γ : float
			The power parameter in the range [0.5, 2.0]. Default is 1.41.
		"""

		assert 0.5 <= γ <= 2.0, "γ must be in [0.5, 2.0]"

		self.name = f"Log Power (γ = {γ})"
		self.id = f'LogP-{γ:.2f}'
		self.color = cmap_green(norm_logp(γ))

		self.γ = γ
		self.param_name = 'γ'
	
	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = np.clip(d, 0, 1)
		td[td > 0] = (-np.log(td[td > 0])) ** self.γ
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, LogPowerTransform) and self.γ == other.γ

	def __hash__(self) -> int:
		return hash((LogPowerTransform, self.γ))

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.γ,))


class ThresholdTransform(ElementwiseTransform):
	"""Threshold transformation: returns 1 if A <= τ, 0 otherwise."""

	def __init__(self, τ: float = 0.05):
		"""Initialize the threshold transformation with a threshold τ.\n
		Parameters
		----------
		τ : float
			The threshold value. Default is 0.05.
		"""

		self.name = f"Threshold (τ = {τ})"
		self.id = f'Thr-{τ}'
		self.color = 'black'

		self.τ = τ
		self.param_name = 'τ'
	
	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = np.clip(d, 0, None)
		td[td > 0] = (td[td > 0] <= self.τ).astype(float)
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, ThresholdTransform) and self.τ == other.τ

	def __hash__(self) -> int:
		return hash((ThresholdTransform, self.τ))

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.τ,))
	
class RankTransform(WeightTransform):
	"""Rank transformation: returns the normalized ordinal ranks of non-zero elements in A."""

	def __init__(self):
		self.name = "Rank"
		self.id = 'Rank'
		self.color = 'purple'

	def __call__(self, A: Union[NDArray[np.float64], csr_matrix]) -> Union[NDArray[np.float64], csr_matrix]:
		if issparse(A):
			coo = A.tocoo() # type: ignore
			mask = (coo.row<coo.col) & (coo.data>0)
			vals = coo.data[mask]
			ranks = rankdata(vals, method='ordinal')
			normed = ranks/(ranks.size+1)
			r, c = coo.row[mask], coo.col[mask]
			rows = np.r_[r, c]; cols = np.r_[c, r]
			data = np.r_[normed, normed]

			return csr_matrix((data,(rows,cols)), shape=A.shape)
		else:
			iu = np.triu_indices_from(A, k=1) # type: ignore
			p_iu = A[iu] > 0
			ranks = rankdata(A[iu][p_iu], method='ordinal')

			tA = np.zeros_like(A, dtype=np.float64)
			tA[iu[0][p_iu], iu[1][p_iu]] = ranks / (ranks.size + 1)
			tA = tA + tA.T
			np.fill_diagonal(tA, 0)
			
			return tA
	
	def __eq__(self, other) -> bool:
		return isinstance(other, RankTransform)

	def __hash__(self) -> int:
		return hash(RankTransform)

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, ())
	
class QuantileTransform(ElementwiseTransform):
	"""Quantile transformation: returns 1 if A <= τ, 0 otherwise, where τ is the q-th quantile of positive elements in A."""

	def __init__(self, q: float = 0.1):
		"""Initialize the quantile transformation with a quantile q.\n
		Parameters
		----------
		q : float
			The quantile value in the range [0.01, 0.5]. Default is 0.1.
		"""

		assert 0.01 <= q <= 0.5, "q must be in [0.01, 0.5]"

		self.name = f"Quantile (q = {q})"
		self.id = f'Q-{int(q*100):02d}'
		self.color = cmap_pink(norm_qtl_q(q))

		self.q = q
		self.param_name = 'q'

	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = d.copy()
		τ = np.quantile(td[td > 0], self.q)
		td[td > 0] = (td[td > 0] <= τ).astype(float)
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, QuantileTransform) and self.q == other.q

	def __hash__(self) -> int:
		return hash((QuantileTransform, self.q))

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.q,))
	
class QuantileThresholding(ElementwiseTransform):
	"""Quantile thresholding: returns 1 if A >= τ, 0 otherwise, 
	where τ is the (1-q)-th quantile of positive elements in A."""

	def __init__(self, q: float = 0.1):
		"""Initialize the quantile thresholding with a quantile q.\n
		Parameters
		----------
		q : float
			The quantile value in the range [0.01, 0.5]. Default is 0.1.
		"""

		assert 0.01 <= q <= 0.5, "q must be in [0.01, 0.5]"

		self.name = f"Quantile Thresholding (q = {q})"
		self.id = f'QThr-{int(q*100):02d}'
		self.color = cmap_purple(norm_qthr(q))

		self.q = q
		self.param_name = 'q'

	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = d.copy()
		τ = np.quantile(td[td > 0], 1 - self.q)
		td[td > 0] = (td[td > 0] >= τ).astype(float)
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, QuantileThresholding) and self.q == other.q

	def __hash__(self) -> int:
		return hash((QuantileThresholding, self.q))

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.q,))
	
class PowerTransform(ElementwiseTransform):
	"""Power transformation: returns A^γ."""

	def __init__(self, γ = 1.41):
		"""Initialize the power transformation with a parameter γ.\n
		Parameters
		----------
		γ : float
			The power parameter in the range [0.5, 2.0]. Default is 1.41.
		"""

		assert 0.5 <= γ <= 2.0, "γ must be in [0.5, 2.0]"

		self.name = f"Power (γ = {γ})"
		self.id = f'P-{γ:.2f}'
		self.color = cmap_blue(norm_pow_γ(γ))

		self.γ = γ
		self.param_name = 'γ'
	
	def f_data(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
		td = d.copy()
		td = td ** self.γ
		return td
	
	def __eq__(self, other) -> bool:
		return isinstance(other, PowerTransform) and self.γ == other.γ

	def __hash__(self) -> int:
		return hash((PowerTransform, self.γ))

	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.γ,))
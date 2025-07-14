# src/models/bc_registry.py
# This file registers the theoretical mean and variance functions for different WSBM models and transformations.

from .beta    import BetaWSBM
from .lognorm import LognormWSBM
from ..transformations import *

from multipledispatch import dispatch
import numpy as np
from functools import partial
from scipy.stats import norm, lognorm
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from itertools import product, combinations_with_replacement
from numpy.typing import NDArray


def edges_block_proportions(K: int,
							π: NDArray[np.float64],
							n: int) -> NDArray[np.float64]:
	"""	Compute the expected proportion of edges between blocks in a WSBM.	
		Parameters
		----------
		K : int
			Number of blocks.
		π : NDArray[np.float64]
			Block proportions, a vector of length K.
		n : int
			Total number of nodes in the WSBM.

		Returns
		-------
		P : NDArray[np.float64]
			Matrix of expected edge proportions between blocks."""

	n_edges = np.zeros((K, K))
	for i, j in product(range(K), repeat=2):
		if i == j:
			n_edges[i, j] = π[i]*n * (π[i]*n - 1) / 2
		else:
			n_edges[i, j] = π[i] * π[j] * n ** 2

	return n_edges / (n * (n - 1) / 2)


@dispatch(BetaWSBM, IdentityTransform)
def theoretical_mean_variance(model, 
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ

	B = ρ * α / (α + 1.0)
	C = ρ * α / (α + 2.0) - B**2
	return B, C

@dispatch(LognormWSBM, IdentityTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	Σ, M, ρ = model.Σ, model.M, model.ρ

	E1 = np.exp(M + Σ ** 2 / 2)
	E2 = np.exp(2 * M + 2 * Σ ** 2)

	B = ρ * E1
	C = ρ * E2 - B**2
	return B, C

@dispatch(BetaWSBM, OppositeTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ

	B = ρ / (α + 1.0)
	C = 2 * ρ / ((α + 1.0) * (α + 2.0)) - B**2
	return B, C

@dispatch(LognormWSBM, OppositeTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	Σ, M, ρ = model.Σ, model.M, model.ρ

	E1 = np.exp(M + Σ ** 2 / 2)
	E2 = np.exp(2 * M + 2 * Σ ** 2)

	B = ρ * (1 - E1)
	C = ρ * (1 - 2*E1 + E2) - B**2
	return B, C

@dispatch(BetaWSBM, LogTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ

	B = ρ / α
	C = ρ * (2.0 / α**2) - B**2
	return B, C

@dispatch(LognormWSBM, LogTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	Σ, M, ρ = model.Σ, model.M, model.ρ

	B = -ρ * M
	C = ρ * (Σ**2 + M**2) - B**2
	return B, C

@dispatch(BetaWSBM, ThresholdTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ

	B = ρ * (transform.τ ** α)
	C = B * (1 - B)
	return B, C

@dispatch(LognormWSBM, ThresholdTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	Σ, M, ρ = model.Σ, model.M, model.ρ

	B = ρ * norm.cdf((np.log(transform.τ) - M) / Σ)
	C = B * (1 - B)
	return B, C

@dispatch(BetaWSBM, RankTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ
	n, π, K = model.n, model.π, model.K

	P = edges_block_proportions(K, π, n)

	B = np.zeros((K, K))
	C = np.zeros((K, K))

	blocks = list(combinations_with_replacement(range(K), 2))

	S1 = sum(P[r,s] / (α + α[r,s]) for (r,s) in blocks)
	S2 = sum(P[r1,s1] * P[r2,s2] / (α + α[r1,s1] + α[r2,s2]) 
			 for (r1,s1) in blocks for (r2,s2) in blocks)

	B = ρ * α * S1
	C = ρ * α * S2 - B**2

	return B, C

@dispatch(LognormWSBM, RankTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:	
	Σ, M, ρ = model.Σ, model.M, model.ρ
	n, π, K = model.n, model.π, model.K

	P = np.triu(edges_block_proportions(K, π, n))

	CDF = partial(lognorm.cdf,  s=Σ, scale=np.exp(M))
	PDF = partial(lognorm.pdf,  s=Σ, scale=np.exp(M))

	def integrand(x):
		PDF_x = PDF(x)
		h_x = np.sum(P * CDF(x))

		return np.stack((h_x * PDF_x, h_x**2 * PDF_x))
	
	I, _ = quad_vec(integrand, 0, 1, epsabs = 1e-9, epsrel = 1e-7)

	B = ρ * I[0]
	C = ρ * I[1] - B**2

	return B, C

@dispatch(BetaWSBM, QuantileTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ
	π = model.π

	def CDF(τ: float) -> float:
		return (π[:, None] * π[None, :] * τ ** α).sum()
	
	τ_q = brentq(lambda τ: CDF(τ) - transform.q, 0, 1)

	B = ρ * (τ_q ** α) # type: ignore
	C = B * (1 - B)
	return B, C

@dispatch(LognormWSBM, QuantileTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	Σ, M, ρ = model.Σ, model.M, model.ρ
	π = model.π

	def CDF(τ: float) -> float:
		return (π[:, None] * π[None, :] * norm.cdf((np.log(τ) - M) / Σ)).sum()
	
	τ_q = brentq(lambda τ: CDF(τ) - transform.q, np.finfo(float).tiny, 1)

	B = ρ * norm.cdf((np.log(τ_q) - M) / Σ) # type: ignore
	C = B * (1 - B)
	return B, C

@dispatch(BetaWSBM, PowerTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	α, ρ = model.α, model.ρ

	B = ρ * α / (α + transform.γ)
	C = ρ * α / (α + 2.0*transform.γ) - B**2
	return B, C

@dispatch(LognormWSBM, PowerTransform)
def theoretical_mean_variance(model,
							  transform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
	Σ, M, ρ  = model.Σ, model.M, model.ρ

	γΣ = transform.γ * Σ
	γM = transform.γ * M
	γE1 = np.exp(γM + γΣ ** 2 / 2)
	γE2 = np.exp(2 * γM + 2 * γΣ ** 2)

	B = ρ * γE1
	C = ρ * γE2 - B**2
	return B, C
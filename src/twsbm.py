# src/twsbm.py
# Contains the TWSBM class, which models transformed weighted stochastic block models
# and computes a range of structural and embedding-based metrics.

from .models.wsbm import WSBM
from .transformations import *
from .metrics import *
from .utils.utils import *

import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

#TODO Harmonize X vs X_A
#TODO Use μ rather than M for the mean of the embedding?

class TWSBM:
	"""Class representing a transformed WSBM instance."""

	def __init__(self, 
				 model: WSBM, 
				 transformation: WeightTransform,
				 A: NDArray[np.float64], 
				 Z: NDArray[np.int32]):
		"""Initialize a TWSBM instance.\n
		Parameters
		----------
		model : WSBM
			The model of the sample.
		transformation : WeightTransform
			The weight transformation having been applied to the sample.
		A : NDArray[np.float64]
			The transformed weight matrix sampled, a 2-D array of shape (n, n).
		Z : NDArray[np.int32]
			The community labels of the sample, a 1-D array of length n.
		"""
		
		self.model = model
		self.transformation = transformation

		self.A = A
		self.Z = Z


		# Theoretical mean and variance of the model and its embedding
		self.B, self.C = model.theoretical_mean_variance(transformation)
		self.M, self.Σ = self.theoretical_embedding_mean_covariance(self.B, self.C, model.π, model.K)

		# Empirical mean and variance of the model and its embedding
		self.X_A, _ = spectral_embedding(A, model.K)
		self.GMM = self.fit_gmm(self.X_A, model.K)
		self.Z_hat, self.π_hat, self.M_hat, self.Σ_hat = self.get_gmm_estimates(self.GMM, self.X_A)

		mapper = self.best_permutation(Z, self.Z_hat)
		self.Z_hat = mapper[self.Z_hat]
		self.π_hat = self.π_hat[mapper]
		self.M_hat = self.M_hat[mapper, :]
		self.Σ_hat = self.Σ_hat[mapper, :, :]

		self.B_hat, self.C_hat = self.empirical_mean_variance(A, self.Z_hat, model.K)

		# Metrics
		self.ARI = ARI(Z, self.Z_hat)
		self.GMM_score = GMMScore(self.GMM, self.X_A)

		# Theoretical and empirical Chernoff information
		self.C_graph  = ChernoffGraph(self.B, self.C, model.K, model.π)
		self.Ĉ_graph  = ChernoffGraph(self.B_hat, self.C_hat, model.K, self.π_hat)
		self.gĈ_graph = GatedChernoffGraph(self.Ĉ_graph, self.GMM_score)

		self.C_embed  = ChernoffEmbedding(self.M, self.Σ, model.K)
		self.Ĉ_embed  = ChernoffEmbedding(self.M_hat, self.Σ_hat, model.K)
		self.gĈ_embed = GatedChernoffEmbedding(self.Ĉ_embed, self.GMM_score)

		assert np.allclose(self.C_graph, self.C_embed, rtol = 1e-5), "Chernoff graph and Chernoff embedding informations do not match."

		self.M, self.Σ = self.procrustes_align_theoretical_embedding_mean_covariance(self.M, self.Σ, self.X_A, self.Z, model.K)

		

	def __eq__(self, other) -> bool:
		if not isinstance(other, TWSBM):
			return False
		return (self.model == other.model and 
		  		self.transformation == other.transformation and
				np.array_equal(self.A, other.A) and 
				np.array_equal(self.Z, other.Z))
	
	def __hash__(self) -> int:
		return hash(self.model) ^ hash(self.transformation) ^ hash(tuple(map(tuple, self.A))) ^ hash(tuple(self.Z))
	
	def __reduce__(self) -> tuple[type, tuple]:
		return (self.__class__, (self.model, self.transformation, self.A, self.Z))
	
	
	
	@staticmethod
	def theoretical_embedding_mean_covariance(B: NDArray[np.float64], 
											  C: NDArray[np.float64], 
											  π: NDArray[np.float64],
											  d: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Calculate the theoretical mean and covariance of the spectral embedding.\n
		Parameters
		----------
		B : NDArray[np.float64]
			Theoretical block mean, a 2-D array of shape (K, K).
		C : NDArray[np.float64]
			Theoretical block variance, a 2-D array of shape (K, K).
		π : NDArray[np.float64]
			Theoretical community proportions, a 1-D array of length K.
		d : int
			Number of dimensions for the embedding.\n
		Returns
		-------
		M : NDArray[np.float64]
			Theoretical mean of the spectral embedding, a 2-D array of shape (K, d).
		Σ : NDArray[np.float64]
			Theoretical covariance of the spectral embedding, a 3-D array of shape (K, d, d)."""
		
		X_B, Λ_B = spectral_embedding(B, d)
		I_pq = np.sign(np.diag(Λ_B))
		Δ = X_B.T @ np.diag(π) @ X_B
		Δ_inv = np.linalg.inv(Δ)

		M = X_B
		Σ = ((I_pq @ Δ_inv)[None, :, :] @ np.einsum('l,kl,li,lj->kij', π, C, X_B, X_B) @ (Δ_inv @ I_pq)[None, :, :])

		return M, Σ
	
	@staticmethod
	def fit_gmm(X: NDArray[np.float64], 
				K: int,
				q_outliers: float = 0.01) -> GaussianMixture:
		"""Fit a Gaussian Mixture Model to the spectral embedding X.\n
		Parameters
		----------
		X : NDArray[np.float64]
			The spectral embedding of the weight matrix, a 2-D array of shape (n, d).
		K : int
			Number of components for the GMM.
		q_outliers : float, optional
			Proportion of outliers to remove from the data before fitting the GMM, by default 0.01.\n
		Returns
		-------
		GMM : GaussianMixture
			The fitted Gaussian Mixture Model.
		"""
		return GaussianMixture(n_components=K, covariance_type='full', random_state=42).fit(X[mask_outliers(X, q_outliers)])
	
	@staticmethod
	def get_gmm_estimates(GMM: GaussianMixture,
					      X: NDArray[np.float64]) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
		"""Get the estimates from the fitted GMM.\n
		Parameters
		----------
		GMM : GaussianMixture
			The fitted Gaussian Mixture Model.
		X : NDArray[np.float64]
			The spectral embedding of the weight matrix, a 2-D array of shape (n, d).\n
		Returns
		-------
		Z_hat : NDArray[np.int32]
			1-D array of length n, where Z_hat[i] is the estimated community index (0 to K-1) for node i.
		π_hat : NDArray[np.float64]
			1-D array of length K, where π_hat[i] is the estimated proportion of community i.
		M_hat : NDArray[np.float64]
			2-D array of shape (K, d), where M_hat[i] is the estimated mean vector of community i.
		Σ_hat : NDArray[np.float64]
			3-D array of shape (K, d, d), where Σ_hat[i] is the estimated covariance matrix of community i."""
		
		Z_hat = GMM.predict(X)
		π_hat = GMM.weights_
		M_hat = GMM.means_
		Σ_hat = GMM.covariances_ * X.shape[0] #type: ignore

		return (np.asarray(Z_hat, dtype=np.int32),
		  		np.asarray(π_hat, dtype=np.float64),
		        np.asarray(M_hat, dtype=np.float64), 
				np.asarray(Σ_hat, dtype=np.float64))
	
	@staticmethod
	def best_permutation(Z0: NDArray[np.int32], 
						 Z1: NDArray[np.int32]) -> NDArray[np.int32]:
		"""Find the best permutation of Z1 to match Z0 using the Hungarian algorithm.\n
		Parameters
		----------
		Z0 : NDArray[np.int32]
			The first clustering to match.
		Z1 : NDArray[np.int32]
			The second clustering to permute.\n
		Returns
		-------
		mapper : NDArray[np.int32]
			A mapping from the indices of Z1 to the indices of Z0 that maximizes the agreement between the two clusterings.
		"""
		
		C = contingency_matrix(Z0, Z1)
		row_ind, col_ind = linear_sum_assignment(-C) # type: ignore

		mapper = np.zeros(C.shape[0], dtype=np.int32)
		mapper[col_ind] = row_ind

		return mapper
	
	@staticmethod
	def empirical_mean_variance(A: NDArray[np.float64],
							    Z_hat: NDArray[np.int32], 
							    K: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Calculate the empirical block mean and variance of the model.\n
		Parameters
		----------
		A : NDArray[np.float64]
			The weight matrix, a 2-D array of shape (n, n).
		Z_hat : NDArray[np.int32]
			1-D array of length n, where Z_hat[i] is the estimated community index (0 to K-1) for node i.
		K : int
			Number of communities.\n
		Returns
		-------
		B_hat : NDArray[np.float64]
			Empirical mean of the edge weights, a 2-D array of shape (K, K).
		C_hat : NDArray[np.float64]
			Empirical variance of the edge weights, a 2-D array of shape (K, K)."""

		nodes = np.bincount(Z_hat, minlength=K)
		edges = nodes[:, None] * nodes[None, :] - np.diag(nodes)

		H  = np.eye(K)[Z_hat]
		S1 = H.T @ A @ H
		S2 = H.T @ A**2 @ H

		with np.errstate(divide='ignore', invalid='ignore'):
			B_hat = S1 / edges
			C_hat = (S2 - B_hat * S1) / (edges - 1)
			B_hat = np.nan_to_num(B_hat)
			C_hat = np.nan_to_num(C_hat)
		return B_hat, C_hat

	@staticmethod
	def procrustes_align_theoretical_embedding_mean_covariance(M: NDArray[np.float64],
															   Σ: NDArray[np.float64],
												 			   X: NDArray[np.float64], 
															   Z: NDArray[np.int32], 
															   K: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Align the theoretical embedding mean and covariance with the empirical embedding using Procrustes alignment.\n
		Parameters
		----------
		M : NDArray[np.float64]
			Theoretical mean of the spectral embedding, a 2-D array of shape (K, d).
		Σ : NDArray[np.float64]
			Theoretical covariance of the spectral embedding, a 3-D array of shape (K, d, d).
		X : NDArray[np.float64]
			The spectral embedding of the weight matrix, a 2-D array of shape (n, d).
		Z : NDArray[np.int32]
			1-D array of length n, where Z[i] is the community index (0 to K-1) for node i.
		K : int
			Number of communities.\n
		Returns
		-------
		M_fit : NDArray[np.float64]
			Aligned mean of the spectral embedding, a 2-D array of shape (K, d).
		Σ_fit : NDArray[np.float64]
			Aligned covariance of the spectral embedding, a 3-D array of shape (K, d, d)."""
		
		def empirical_embedding_mean_covariance(X: NDArray[np.float64], 
												Z: NDArray[np.int32], 
												K: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
			"""Calculate the empirical mean and covariance of the spectral embedding, provided the ground truth labels.\n
			Parameters
			----------
			X : NDArray[np.float64]
				The spectral embedding of the weight matrix, a 2-D array of shape (n, d).
			Z : NDArray[np.int32]
				1-D array of length n, where Z[i] is the community index (0 to K-1) for node i.
			K : int
				Number of communities.\n
			Returns
			-------
			M_hat_Z : NDArray[np.float64]
				Empirical mean of the spectral embedding, a 2-D array of shape (K, d).
			Σ_hat_Z : NDArray[np.float64]
				Empirical covariance of the spectral embedding, a 3-D array of shape (K, d, d)."""
		
			M_hat_Z = np.zeros((K, X.shape[1]), dtype=np.float64)
			for k in range(K):
				M_hat_Z[k] = X[Z == k].mean(axis=0)

			X_centered = X - M_hat_Z[Z]
			Σ_hat_Z = np.zeros((K, X.shape[1], X.shape[1]), dtype=np.float64)
			
			for k in range(K):
				X_k = X_centered[Z == k]
				if len(X_k) > 0:
					Σ_hat_Z[k] = np.cov(X_k.T)

			Σ_hat_Z = Σ_hat_Z * X.shape[0]  # Scale by number of samples

			return M_hat_Z, Σ_hat_Z
		
		M_hat_Z, Σ_hat_Z = empirical_embedding_mean_covariance(X, Z, K)
		R, beta = procrustes_align(M, M_hat_Z)

		M_fit = beta * (M - M.mean(axis=0)) @ R  + M_hat_Z.mean(axis=0)
		Σ_fit = beta**2 * (R @ Σ @ R.T) #TODO Does not fit the data well
		# TODO
		# ChernoffEmbedding(M_hat_Z, Σ_hat_Z, K)
		# ChernoffEmbedding(self.M_hat, self.Σ_hat, model.K) (when ARI is high)
		# ChernoffEmbedding(self.M, self.Σ, self.K) after procrustes alignment
		# do not seem to match with the following 3 (which are matching)
		# ChernoffGraph(self.B, self.C, model.K, model.π)
		# ChernoffGraph(self.B_hat, self.C_hat, model.K, self.π_hat) (when ARI is high)
		# ChernoffEmbedding(self.M, self.Σ, self.K) before procrustes alignment
		# Why?

		return M_fit, Σ_fit
		

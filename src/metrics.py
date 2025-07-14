from .utils.string_utils import *
from .utils.utils import *

from sklearn.mixture import GaussianMixture
from numpy.typing import NDArray
import numpy as np
from sklearn.metrics import adjusted_rand_score
from itertools import combinations
from scipy.optimize import minimize_scalar

def gating_function(s: float,
                    s0: float = 1.0) -> float:
    """Apply a gating function to the input array.\n
    Parameters
    ----------
    s : float
        The input score to be gated.
    s0 : float, optional
        The threshold score for gating, by default 1.0.\n
    Returns
    -------
    int
        Returns 1 if the score is greater than the threshold, otherwise returns 0.
    """
    return step(s, s0)


def ARI(Z1: NDArray[np.int32], 
        Z2: NDArray[np.int32]) -> float:
    """Calculate the Adjustd Rand Index between two clusterings.\n
    Parameters
    ----------
    Z1 : NDArray[np.int32]
        First clustering labels.
    Z2 : NDArray[np.int32]
        Second clustering labels.\n
    Returns
    -------
    float
        The Rand index between the two clusterings.
    """
    return adjusted_rand_score(Z1, Z2)

ARI.id = 'ARI'                     # type: ignore
ARI.name = "Adjusted Rand Index"   # type: ignore
ARI.pretty_name = 'ARI'            # type: ignore
ARI.color = 'red'                  # type: ignore 
	

def GMMScore(GMM: GaussianMixture,
             X: NDArray[np.float64]) -> float:
    """Calculate the GMM score for a given Gaussian Mixture Model and data.\n
    Parameters
    ----------
    GMM : GaussianMixture
        The Gaussian Mixture Model to evaluate.
    X : NDArray[np.float64]
        The data points to score.\n
    Returns
    -------
    float
        The GMM score for the data points.
    """
    return GMM.score(X)  # type: ignore

GMMScore.id = 'GMM_score'           # type: ignore
GMMScore.name = "GMM score"         # type: ignore
GMMScore.pretty_name = 'GMM score'  # type: ignore
GMMScore.color = 'green'            # type: ignore


def ChernoffGraph(B: NDArray[np.float64], 
                  C: NDArray[np.float64],
                  K: int,
                  π: NDArray[np.float64]) -> float:
    """Calculate the Chernoff graph-information.\n
    Parameters
    ----------
    B : NDArray[np.float64]
        Theoretical block mean, a 2-D array of shape (K, K).
    C : NDArray[np.float64]
        Theoretical block variance, a 2-D array of shape (K, K).
    K : int
        Number of communities.
    π : NDArray[np.float64]
        Theoretical community proportions, a 1-D array of length K.\n
    Returns
    -------
    float
        The Chernoff graph-information value.
    """

    def objective(t: float, k: int, l: int) -> float:
        e = np.eye(K)
        S_kl_t = (1 - t) * np.diag(C[k]) + t * np.diag(C[l])
        u = B @ (e[k] - e[l])
        matrix_res = u.T @ np.diag(π) @ np.linalg.lstsq(S_kl_t, u, rcond=None)[0]
        return 0.5 * t * (1 - t) * matrix_res.item()
    
    def neg_objective(t: float, k: int, l: int) -> float:
        return -objective(t, k, l)
    
    c = np.inf
    for k, l in combinations(range(K), 2):
        res = minimize_scalar(neg_objective, bounds=(0, 1), method='bounded', args=(k, l))
        c = min(c, -res.fun) # type: ignore

    return c

ChernoffGraph.id = 'C_graph'                            # type: ignore
ChernoffGraph.name = "Chernoff graph-information"       # type: ignore
ChernoffGraph.pretty_name = 'C' + sup('graph')          # type: ignore
ChernoffGraph.color = 'gold'                            # type: ignore


def ChernoffGraphEstimate(B_hat: NDArray[np.float64], 
                          C_hat: NDArray[np.float64],
                          K: int,
                          π_hat: NDArray[np.float64]) -> float:
    """Calculate the Chernoff graph-information estimate.\n
    Parameters
    ----------
    B_hat : NDArray[np.float64]
        Estimated block mean, a 2-D array of shape (K, K).
    C_hat : NDArray[np.float64]
        Estimated block variance, a 2-D array of shape (K, K).
    K : int
        Number of communities.
    π_hat : NDArray[np.float64]
        Estimated community proportions, a 1-D array of length K.\n
    Returns
    -------
    float
        The Chernoff graph-information estimate value.
    """
    return ChernoffGraph(B_hat, C_hat, K, π_hat)

ChernoffGraphEstimate.id = 'Ĉ_graph'                        # type: ignore
ChernoffGraph.name = "Chernoff graph-information estimate"  # type: ignore
ChernoffGraphEstimate.pretty_name = 'Ĉ' + sup('graph')      # type: ignore
ChernoffGraphEstimate.color = 'teal'                        # type: ignore


def GatedChernoffGraph(ĉ_graph: float,
                       score: float) -> float:
    """Calculate the gated Chernoff graph-information estimate.\n
    Parameters
    ----------
    ĉ_graph : float
        The Chernoff graph-information estimate value.
    score : float
        The score to apply the gating function to.\n
    Returns
    -------
    float
        The gated Chernoff graph-information estimate value.
    """
    return gating_function(score) * ĉ_graph

GatedChernoffGraph.id = 'gĈ_graph'                                      # type: ignore
GatedChernoffGraph.name = "Gated Chernoff graph-information estimate"   # type: ignore
GatedChernoffGraph.pretty_name = 'gĈ' + sup('graph')                    # type: ignore
GatedChernoffGraph.color = 'teal'                                       # type: ignore


def ChernoffEmbedding(M: NDArray[np.float64], 
                      Σ: NDArray[np.float64],
                      K: int) -> float:
    """Calculate the Chernoff embedding-information.\n
    Parameters
    ----------
    M : NDArray[np.float64]
        The mean of the Gaussian mixture model, a 2-D array of shape (K, d).
    Σ : NDArray[np.float64]
        The covariance matrix of the Gaussian mixture model, a 2-D array of shape (K, d, d).
    K : int
        Number of communities.\n
    Returns
    -------
    float
        The Chernoff embedding-information value.
    """

    def objective(t: float, k: int, l: int) -> float:
        Σ_kl_t = (1 - t) * Σ[k] + t * Σ[l]
        mk_ml = M[k] - M[l]
        matrix_res = mk_ml.T @ np.linalg.inv(Σ_kl_t) @ mk_ml
        return 0.5 * t * (1 - t) * matrix_res.item()
    
    def neg_objective(t: float, k: int, l: int) -> float:
        return -objective(t, k, l)
    
    c = np.inf
    for k, l in combinations(range(K), 2):
        res = minimize_scalar(neg_objective, bounds=(0, 1), method='bounded', args=(k, l))
        c = min(c, -res.fun) # type: ignore

    return c

ChernoffEmbedding.id = 'C_embed'                            # type: ignore
ChernoffEmbedding.name = "Chernoff embedding-information"   # type: ignore
ChernoffEmbedding.pretty_name = 'C' + sup('embed')          # type: ignore
ChernoffEmbedding.color = 'brown'                           # type: ignore


def ChernoffEmbeddingEstimate(M_hat: NDArray[np.float64],
                              Σ_hat: NDArray[np.float64],
                              K: int) -> float:
        """Calculate the Chernoff embedding-information estimate.\n
        Parameters
        ----------
        M_hat : NDArray[np.float64]
            Estimated mean of the Gaussian mixture model, a 2-D array of shape (K, d).
        Σ_hat : NDArray[np.float64]
            Estimated covariance matrix of the Gaussian mixture model, a 2-D array of shape (K, d, d).
        K : int
            Number of communities.\n
        Returns
        -------
        float
            The Chernoff embedding-information estimate value.
        """
        return ChernoffEmbedding(M_hat, Σ_hat, K)

ChernoffEmbeddingEstimate.id = 'Ĉ_embed'                                    # type: ignore
ChernoffEmbeddingEstimate.name = "Chernoff embedding-information estimate"  # type: ignore
ChernoffEmbeddingEstimate.pretty_name = 'Ĉ' + sup('embed')                  # type: ignore
ChernoffEmbeddingEstimate.color = 'magenta'                                 # type: ignore


def GatedChernoffEmbedding(ĉ_embed: float,
                           score: float) -> float:
    """Calculate the gated Chernoff embedding-information estimate.\n
    Parameters
    ----------
    ĉ_embed : float
        The Chernoff embedding-information estimate value.
    score : float
        The score to apply the gating function to.\n
    Returns
    -------
    float
        The gated Chernoff embedding-information estimate value.
    """
    return gating_function(score) * ĉ_embed

GatedChernoffEmbedding.id = 'gĈ_embed'                                          # type: ignore
GatedChernoffEmbedding.name = "Gated Chernoff embedding-information estimate"   # type: ignore
GatedChernoffEmbedding.pretty_name = 'gĈ' + sup('embed')                        # type: ignore
GatedChernoffEmbedding.color = 'magenta'                                        # type: ignore
# src/visualization/plot_embeddings.py
# Provides functions to sample instances of the WSBM model with different parameters and transformations,
# and to plot the embeddings of the various TWSBM instances.

from ..models.wsbm import WSBM
from ..transformations import *
from ..twsbm import *

from typing import Union, Any, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def sample_instances(model: WSBM, 
					 model_parameters: Sequence[Tuple[Union[NDArray[Any], float], ...]],
					 transformations: Sequence[WeightTransform]) -> dict[WSBM, dict[WeightTransform, TWSBM]]:
	"""Sample instances of the WSBM model with different parameters and transformations.\n
	Parameters
	----------
	model : WSBM
		The base WSBM model to sample from.
	model_parameters : list[NDArray[np.float64]]
		A list of parameter sets for the WSBM model.
	transformations : list[WeightTransform]
		A list of weight transformations to apply to the sampled adjacency matrix.\n
	Returns
	-------
	dict[WSBM, dict[WeightTransform, TWSBM]]
		A dictionary where keys are WSBM instances and values are dictionaries mapping transformations to TWSBM instances."""

	K, ρ, π, n = model.K, model.ρ, model.π, model.n
	instances = {}
	for params in model_parameters:
		m = model.__class__(K, ρ, π, n, *params) #type: ignore
		instances[m] = {}
		A, Z = m.sample(seed=42)
		for t in transformations:
			instances[m][t] = TWSBM(model=m, transformation=t, A=t(A), Z=Z)

	return instances


def plot_embeddings(instances: dict[WSBM, dict[WeightTransform, TWSBM]],
					Z_display: Union[str, list[str], set[str], None] = None,
					M_display: Union[str, list[str], set[str], None] = None,
					Σ_display: Union[str, list[str], set[str], None] = None,
					stats_display: bool = True,
					q_outliers: float = 0.01) -> None:
	"""Plot the embeddings of the TWSBM instances in a grid for various parameters and transformations.\n
	Parameters
	----------
	instances : dict[WSBM, dict[WeightTransform, TWSBM]]
		A dictionary where keys are WSBM instances and values are dictionaries mapping transformations to TWSBM instances.
	Z_display : Union[str, list[str], set[str], None] optional
		The modes for displaying the Z values, can be 'T', 'P', or None (default).
		'T' for true labels, 'P' for predicted labels, None for black.
	M_display : Union[str, list[str], set[str], None] optional
		The modes for displaying the Mean values, can be 'T', 'P', or None (default).
		'T' for true means, 'P' for predicted means, None for no display.
		Both 'T' and 'P' can be activated simultaneously.
	Σ_display : Union[str, list[str], set[str], None] optional
		The modes for displaying the Covariance values, can be 'T', 'P', or None (default).
		'T' for true covariance, 'P' for predicted covariance, None for no display.
		Both 'T' and 'P' can be activated simultaneously.
	stats_display : bool, optional
		Whether to display statistics on the plot, by default True.
	q_outliers : float, optional
		The quantile threshold for outliers, by default 0.01.\n
	Returns
	-------
	None"""
	
	def _normalize_modes(arg, *, name):
		"""Turn a string like "TP", a list like ["P","T"], or None
		into a set {"T","P"}.  Raises ValueError if any element is invalid.
		Parameters
		----------
		arg : Union[str, list[str], set[str], None]
			The input modes to normalize.
		name : str
			The name of the argument for error messages.\n
		Returns
		-------
		set[str]
			A set of normalized modes."""
		
		allowed = {"T","P"}
		if arg is None:
			modes = set()
		elif isinstance(arg, str):
			modes = set(arg)
		elif isinstance(arg, (list, tuple, set)):
			modes = set(arg)
		else:
			raise TypeError(f"{name} must be a string or sequence of strings, got {type(arg)}")
		bad = modes - allowed
		if bad:
			raise ValueError(f"{name!r} contains invalid mode(s): {bad}. "
							f"Allowed modes are {allowed!r}.")
		return modes

	def _draw_ellipse(ax, M, Σ, cmap):
		"""Draw ellipses representing the covariance matrices at the means.\n
		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes to draw the ellipses on.
		M : NDArray[np.float64]
			The means of the Gaussian distributions, a 2-D array of shape (K, d).
		Σ : NDArray[np.float64]
			The covariance matrices, a 3-D array of shape (K, d, d).
		cmap : matplotlib.colors.ListedColormap
			The colormap to use for the ellipses.\n
		Returns
		-------
		None
		"""
		for c, mean, cov in zip(cmap(np.array([0, 1])), M, Σ):
			eigenvalues, eigenvectors = np.linalg.eigh(cov)
			angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
			width, height = 2 * np.sqrt(6 * eigenvalues)
			ellip = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, edgecolor=c, facecolor='none', linestyle='solid') #type: ignore
			ax.add_patch(ellip)
	
	Zmodes = _normalize_modes(Z_display, name="Z_display")
	Mmodes = _normalize_modes(M_display, name="M_display")
	Σmodes = _normalize_modes(Σ_display, name="Σ_display")

	cmap_true = ListedColormap(["blue", "red"])
	cmap_pred = ListedColormap(["deepskyblue", "hotpink"])

	if 'T' in Zmodes and 'P' in Zmodes:
		raise ValueError("Cannot color nodes by both Z and Ẑ simultaneously.")
	
	model = list(instances.keys())[0]

	if model.K != 2:
		raise NotImplementedError("This function currently only supports K=2.")

	n_rows, n_cols = len(instances), len(list(instances.values())[0].values())
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 + 3*n_cols, 2 + 3*n_rows), squeeze=False)
	title = 'Embeddings of t(A) for various transforms t on Model:\n' + str(model)
	fig.suptitle(title, fontsize=22)

	for i, (m, m_instances) in enumerate(instances.items()):
		axes[i, 0].set_ylabel(m.param_matrix_str() + '\n', fontsize=8)

		argmax_j = {}
		for c in ['C_graph', 'Ĉ_graph', 'Ĉ_embed']:
			argmax_j[c] = max(enumerate(m_instances.values()), key=lambda ji: getattr(ji[1], c))[0]

		for j, (t, I) in enumerate(m_instances.items()):
			ax = axes[i][j]
			plt.sca(ax)

			t = np.clip(I.ARI, 0, 1)**2
			u = np.clip(sigmoid_w95(I.GMM_score, -5, 10), 0, 1)
			base = np.array([1 - t, 1, 1 - t])
			grey = np.array([0.5, 0.5, 0.5])
			facecolor = u * base + (1 - u) * grey

			mask = mask_outliers(I.X_A, q_outliers)
			X_A, Z, Z_hat = I.X_A[mask], I.Z[mask], I.Z_hat[mask]

			if 'T' in Zmodes:
				ax.scatter(X_A[:, 0], X_A[:, 1], c=Z, cmap = cmap_true,
			   			   marker='.', alpha=0.2 + (1 - u) * 0.4 if stats_display else 1)
			elif 'P' in Zmodes:
				ax.scatter(X_A[:, 0], X_A[:, 1], c=Z_hat, cmap = cmap_pred,
			   			   marker='.', alpha=0.2 + (1 - u) * 0.4 if stats_display else 1)
			else:
				ax.scatter(X_A[:, 0], X_A[:, 1], c='black',
						   marker='.', alpha=0.2 + (1 - u) * 0.4 if stats_display else 1)
				
			if 'T' in Mmodes:
				ax.scatter(I.M[:, 0], I.M[:, 1], c=[0, 1], cmap = cmap_true,
						   marker='o', s = 100, linewidth=1, edgecolor='black')
				
			if 'P' in Mmodes:
				ax.scatter(I.M_hat[:, 0], I.M_hat[:, 1], c=[0, 1], cmap = cmap_pred,
						   marker='o', s = 100, linewidth=1, edgecolor='black')
				
			if 'T' in Σmodes:
				raise NotImplementedError("Drawing ellipses for theoretical embeddings is not implemented yet.")
				#_draw_ellipse(ax, I.M, I.Σ / I.model.n, cmap_true)
			
			if 'P' in Σmodes:
				_draw_ellipse(ax, I.M_hat, I.Σ_hat / I.model.n, cmap_pred)


			ax.set_xticks([])
			ax.set_yticks([])

			stats = [f"ARI:  {I.ARI:.2f}",
					 f"πẐ:   {list(np.sort(I.π_hat).round(2))}",
					 f"GS:   {I.GMM_score:.2f}",
					 f"C:     {I.C_graph:.5f}",
					 f"ĈG:   {I.Ĉ_graph:.5f}",
					 f"ĈE:   {I.Ĉ_embed:.5f}",]
			
			if stats_display:
				ax.set_facecolor(facecolor)
				invisible_handles = [Line2D([], [], color='none') for _ in stats]
				leg = ax.legend(
					handles=invisible_handles,
					labels=stats,
					loc="upper left",
					fontsize=8,
					handlelength=0,
					handletextpad=0,)
				for h in invisible_handles:
					h.set_visible(False)
				for txt in leg.get_texts():
					s = txt.get_text()

					for c, c_string in zip(['C_graph', 'Ĉ_graph', 'Ĉ_embed'], ['C', 'ĈG', 'ĈE']):
						if j == argmax_j[c] and s.startswith(c_string):
							txt.set_fontweight("bold")

			if i == 0:
				ax.set_title(I.transformation.name + "\n", fontsize=12) #type: ignore
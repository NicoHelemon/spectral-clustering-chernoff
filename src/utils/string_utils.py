# src/utils/string_utils.py
# Utility functions for string manipulation in the context of WSBM and TWSBM visualizations.

import numpy as np
from numpy.typing import NDArray
from typing import Optional

superscript_map = {
	"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
	"7": "⁷", "8": "⁸", "9": "⁹", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ",
	"e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ᶦ", "j": "ʲ", "k": "ᵏ",
	"l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "q": "۹", "r": "ʳ",
	"s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
	"z": "ᶻ", "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "E": "ᴱ", "F": "ᶠ",
	"G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ",
	"N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "Q": "Q", "R": "ᴿ", "S": "ˢ", "T": "ᵀ",
	"U": "ᵁ", "V": "ⱽ", "W": "ᵂ", "X": "ˣ", "Y": "ʸ", "Z": "ᶻ", "+": "⁺",
	"-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", " " : "\u202F"}

subscript_map = {
	"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
	"7": "₇", "8": "₈", "9": "₉", "a": "ₐ", "b": "♭", "c": "꜀", "d": "ᑯ",
	"e": "ₑ", "f": "բ", "g": "₉", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
	"l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "q": "૧", "r": "ᵣ",
	"s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "w": "w", "x": "ₓ", "y": "ᵧ",
	"z": "₂", "A": "ₐ", "B": "₈", "C": "C", "D": "D", "E": "ₑ", "F": "բ",
	"G": "G", "H": "ₕ", "I": "ᵢ", "J": "ⱼ", "K": "ₖ", "L": "ₗ", "M": "ₘ",
	"N": "ₙ", "O": "ₒ", "P": "ₚ", "Q": "Q", "R": "ᵣ", "S": "ₛ", "T": "ₜ",
	"U": "ᵤ", "V": "ᵥ", "W": "w", "X": "ₓ", "Y": "ᵧ", "Z": "Z", "+": "₊",
	"-": "₋", "=": "₌", "(": "₍", ")": "₎", " " : "\u202F"}

SUP = str.maketrans(''.join(superscript_map.keys()), ''.join(superscript_map.values()))
SUB = str.maketrans(''.join(subscript_map.keys()), ''.join(subscript_map.values()))

def sup(text: str) -> str:
	"""Convert characters in text to their superscript equivalents.\n
	Parameters
	----------
	text : str
		The input string to be converted.\n
	Returns
	-------
	str
		A string with characters replaced by their superscript equivalents."""
	return text.translate(SUP)

def sub(text: str) -> str:
	"""Convert characters in text to their subscript equivalents.\n
	Parameters
	----------
	text : str
		The input string to be converted.\n
	Returns
	-------
	str
		A string with characters replaced by their subscript equivalents."""
	return text.translate(SUB)

def model_base_parameters_str(n: int, ρ: float, π: NDArray[np.float64]) -> str:
	"""Generate a string representation of the base parameters of a WSBM model.\n
	Parameters
	----------
	n : int
		Total number of nodes in the WSBM.
	ρ : float
		The mixing parameter of the WSBM.
	π : NDArray[np.float64]
		Block proportions, a vector of length K.\n
	Returns
	-------
	str
		A string representation of the model's base parameters in the format:\n
		"n = {n}, ρ = {ρ}, π = {π.tolist()}"."""
	return f"n = {n}, ρ = {ρ}, π = {π.tolist()}"

def fancy_matrix_str(arr: NDArray[np.float64]) -> str:
	"""Generate a string representation of a 2x2 matrix with fancy brackets.\n
	Parameters
	----------
	arr : NDArray[np.float64]
		A 2D numpy array with exactly two rows.\n
	Returns
	-------
	str
		A string representation of the matrix in the format:\n
		⎡ a₁₁  a₁₂ ⎤\n
		⎣ a₂₁  a₂₂ ⎦\n"""

	assert arr.ndim == 2, "Input must be a 2D array"
	assert arr.shape[0] == 2, "Input must have exactly two rows"
	rows = arr.astype(str).tolist()
	
	top = "  ".join(rows[0])
	bot = "  ".join(rows[1])
	return f"⎡ {top} ⎤\n⎣ {bot} ⎦"

def param_matrix_str(P: NDArray[np.float64], 
					 param_string: str, 
					 varying: Optional[tuple[int, int]] = None) -> str:
	"""Generate a string representation of a 2x2 parameter matrix with fancy formatting.\n
	Parameters
	----------
	P : NDArray[np.float64]
		A 2D numpy array with exactly two rows.
	param_string : str
		The string to be used as a prefix for each parameter in the matrix.
	varying : Optional[tuple[int, int]]
		A tuple indicating which element in the matrix is varying, if any.\n
	Returns
	-------
	str
		A string representation of the parameter matrix in the format:\n
		⎡ α₁₁ = P₁₁  α₁₂ = P₂₁ ⎤\n
		⎣ α₂₁ = P₂₁  α₂₂ = P₂₂ ⎦\n
		Where α is the parameter string.\n
		If coordinates (i, j) are provided in varying, the corresponding entry will be αᵢⱼ = αᵢⱼ.\n"""
	  
	assert P.ndim == 2, "Input must be a 2D array"
	assert P.shape[0] == 2, "Input must have exactly two rows"

	arr = np.full(P.shape, '', dtype=object)

	for i, j in np.ndindex(P.shape):
		p_str = param_string + sub(f'{i+1}{j+1}')
		arr[i][j] = f'{p_str} = {P[i, j]:.2f}'
		if varying is not None and ((i, j) == varying or (j, i) == varying):
			arr[i][j] = f' {p_str} = {p_str} '

	return fancy_matrix_str(arr)
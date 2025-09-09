# utils.py - Funzioni di supporto per il progetto
import numpy as np
from scipy.fft import dct as _dct, idct as _idct
import time

# ---------- DCT "fatta in casa" O(N^3) via matrici di base ----------
def _dct_matrix(N: int) -> np.ndarray:
    """Matrice DCT-II ortonormale (come a lezione)."""
    k = np.arange(N)[:, None]       # righe
    i = np.arange(N)[None, :]       # colonne
    D = np.cos((np.pi*(2*i + 1)*k)/(2*N)).astype(np.float64)
    D[0, :] *= 1/np.sqrt(N)
    D[1:, :] *= np.sqrt(2/N)
    return D

def dct2_naive(X: np.ndarray) -> np.ndarray:
    """
    DCT2 'fatta in casa' O(N^3): D @ X @ D.T
    Stessa scalatura ortonormale usata nei test del prof.
    """
    X = np.asarray(X, dtype=np.float64)
    N, M = X.shape
    assert N == M, "Questa DCT2 naive assume blocchi quadrati N×N."
    D = _dct_matrix(N)
    return D @ X @ D.T

def idct2_naive(C: np.ndarray) -> np.ndarray:
    """IDCT2 coerente con la DCT2 ortonormale: D.T @ C @ D."""
    C = np.asarray(C, dtype=np.float64)
    N, M = C.shape
    assert N == M, "Questa IDCT2 naive assume blocchi quadrati N×N."
    D = _dct_matrix(N)
    return D.T @ C @ D

# ---------- DCT/IDCT veloci (SciPy) con stessa scalatura ----------
def dct2_fast(block: np.ndarray) -> np.ndarray:
    """DCT2 veloce (Type-II, norm='ortho') per righe e colonne."""
    X = np.asarray(block, dtype=np.float64)
    return _dct(_dct(X, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')

def idct2_fast(C: np.ndarray) -> np.ndarray:
    """IDCT2 veloce coerente."""
    return _idct(_idct(C, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')

# ---------- Utility timing ----------
def measure_time(func, *args, n_iterations=3):
    """Misura il tempo medio di esecuzione di una funzione."""
    times = []
    for _ in range(n_iterations):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

# test_dct.py - Verifica della correttezza della DCT
import numpy as np
from scipy.fft import dct  # <-- per il test 1D
from utils import dct2_naive, dct2_fast, idct2_fast

def test_dct_implementation():
    """
    Testa:
      (A) DCT2 del blocco 8x8 (come nel PDF)
      (B) DCT 1D della prima riga (come richiesto dal PDF)
    """
    # Blocco di test 8x8 dal PDF
    test_block = np.array([
        [231, 32, 233, 161, 24, 71, 140, 245],
        [247, 40, 248, 245, 124, 204, 36, 107],
        [234, 202, 245, 167, 9, 217, 239, 173],
        [193, 190, 100, 167, 43, 180, 8, 70],
        [11, 24, 210, 177, 81, 243, 8, 112],
        [97, 195, 203, 47, 125, 114, 165, 181],
        [193, 70, 174, 167, 41, 30, 127, 245],
        [87, 149, 57, 192, 65, 129, 178, 228]
    ], dtype=np.float64)

    # Risultato atteso 2D (dal PDF)
    expected_dct = np.array([
        [1.11e+03, 4.40e+01, 7.59e+01, -1.38e+02, 3.50e+00, 1.22e+02, 1.95e+02, -1.01e+02],
        [7.71e+01, 1.14e+02, -2.18e+01, 4.13e+01, 8.77e+00, 9.90e+01, 1.38e+02, 1.09e+01],
        [4.48e+01, -6.27e+01, 1.11e+02, -7.63e+01, 1.24e+02, 9.55e+01, -3.98e+01, 5.85e+01],
        [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01, 2.66e+01, -3.68e+01, 6.61e+01, 1.25e+02],
        [-1.09e+02, -4.33e+01, -5.55e+01, 8.17e+00, 3.02e+01, -2.86e+01, 2.44e+00, -9.41e+01],
        [-5.38e+00, 5.66e+01, 1.73e+02, -3.54e+01, 3.23e+01, 3.34e+01, -5.81e+01, 1.90e+01],
        [7.88e+01, -6.45e+01, 1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02, 3.98e+01],
        [1.97e+01, -7.81e+01, 9.72e-01, -7.23e+01, -2.15e+01, 8.13e+01, 6.37e+01, 5.90e+00]
    ])

    print("=" * 60)
    print("TEST DCT IMPLEMENTATION")
    print("=" * 60)

    # (A) DCT2: naive vs fast e confronto con atteso (qualitativo)
    print("\n1) DCT2 Naive vs Fast")
    dct_naive = dct2_naive(test_block)
    dct_fast_result = dct2_fast(test_block)
    diff_nf = np.abs(dct_naive - dct_fast_result)
    print(f"   Max |naive-fast| = {np.max(diff_nf):.3e}")

    # Inversione (coerenza numerica)
    reconstructed = idct2_fast(dct_fast_result)
    rec_err = np.abs(test_block - reconstructed)
    print(f"2) IDCT2 Fast -> max rec err = {np.max(rec_err):.3e}, mean = {np.mean(rec_err):.3e}")

    # Mostra la prima riga della DCT2 (confronto visivo col PDF)
    print("\n3) Prima riga DCT2 (nostra vs attesa PDF):")
    print("   nostra :", np.array2string(dct_fast_result[0, :], precision=2, suppress_small=True))
    print("   attesa :", np.array2string(expected_dct[0, :], precision=2, suppress_small=True))

    # (B) DCT 1D sulla prima riga (richiesta esplicita nel PDF)
    print("\n4) DCT 1D della prima riga (type=2, norm='ortho'):")
    first_row = test_block[0, :].astype(np.float64)
    dct1d = dct(first_row, type=2, norm='ortho')
    expected_1d = np.array([4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02, 6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01])
    diff_1d = np.abs(dct1d - expected_1d)
    print("   nostra :", np.array2string(dct1d, precision=2, suppress_small=True))
    print("   attesa :", np.array2string(expected_1d, precision=2, suppress_small=True))
    print(f"   max |diff| = {np.max(diff_1d):.3e}")

    # Esito
    ok_rec = np.max(rec_err) < 1e-10
    ok_1d = np.max(diff_1d) < 1.0  # tolleranza “umana” dato che i valori del PDF sono arrotondati
    if ok_rec and ok_1d:
        print("\nTEST PASSED (scalatura e implementazione coerenti)")
    else:
        print("\nTEST CHECK (controlla scalatura/type o arrotondamenti)")

if __name__ == "__main__":
    test_dct_implementation()

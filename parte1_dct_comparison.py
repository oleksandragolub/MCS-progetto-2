# parte1_dct_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from utils import dct2_naive, dct2_fast, measure_time
import time

np.random.seed(0)

def benchmark_dct_implementations():
    """
    Confronta i tempi di esecuzione della DCT naive vs fast
    per diverse dimensioni di matrici
    """
    print("=" * 60)
    print("PARTE 1: BENCHMARK DCT IMPLEMENTATIONS")
    print("=" * 60)
    
    # Dimensioni da testare (parti da piccole per la DCT naive)
    # Per la DCT naive non andare oltre 128 perché diventa molto lenta
    sizes_naive = [4, 8, 16, 32, 64, 128]
    sizes_fast = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    times_naive = []
    times_fast_for_naive_sizes = []
    times_fast_all = []
    
    print("\n Testing DCT Naive (questo richiederà qualche minuto)...")
    
    # Test DCT Naive
    for N in sizes_naive:
        # Crea matrice random NxN
        matrix = np.random.randn(N, N)
        
        # Misura tempo DCT naive
        print(f"   Dimensione {N}x{N}...")
        
        # Per matrici piccole facciamo più iterazioni
        n_iter = max(1, 100 // N)
        
        start = time.time()
        for _ in range(n_iter):
            _ = dct2_naive(matrix)
        elapsed_naive = (time.time() - start) / n_iter
        times_naive.append(elapsed_naive)
        
        # Misura anche la fast per confronto diretto
        start = time.time()
        for _ in range(100):  # Più iterazioni per la fast che è veloce
            _ = dct2_fast(matrix)
        elapsed_fast = (time.time() - start) / 100
        times_fast_for_naive_sizes.append(elapsed_fast)
        
        print(f"      Naive: {elapsed_naive:.6f}s | Fast: {elapsed_fast:.6f}s")
    
    print("\n Testing DCT Fast per dimensioni maggiori...")
    
    # Test DCT Fast per tutte le dimensioni
    for N in sizes_fast:
        matrix = np.random.randn(N, N)
        print(f"   Dimensione {N}x{N}...")
        
        start = time.time()
        for _ in range(100):
            _ = dct2_fast(matrix)
        elapsed = (time.time() - start) / 100
        times_fast_all.append(elapsed)
        print(f"      Fast: {elapsed:.6f}s")
    
    # Creazione del grafico
    create_comparison_plot(sizes_naive, times_naive, times_fast_for_naive_sizes, 
                          sizes_fast, times_fast_all)
    
    # Analisi della complessità
    analyze_complexity(sizes_naive, times_naive, times_fast_for_naive_sizes)
    
    return sizes_naive, times_naive, sizes_fast, times_fast_all

def create_comparison_plot(sizes_naive, times_naive, times_fast_comp, sizes_fast, times_fast_all):
    """
    Crea il grafico di confronto in scala semilogaritmica
    """
    plt.figure(figsize=(12, 8))
    
    # Plot principale con scala semilogaritmica sull'asse y
    plt.semilogy(sizes_naive, times_naive, 'ro-', linewidth=2, markersize=8, 
                label='DCT Naive (O(N³))')
    plt.semilogy(sizes_fast, times_fast_all, 'bs-', linewidth=2, markersize=6,
                label='DCT Fast/FFT (O(N²logN))')
    
    # Aggiungi linee di riferimento teoriche
    # Per N³
    n3_theoretical = [(n/sizes_naive[0])**3 * times_naive[0] for n in sizes_naive]
    plt.semilogy(sizes_naive, n3_theoretical, 'r--', alpha=0.3, 
                label='N³ teorico')
    
    # Per N²log(N)
    n2log_theoretical = [(n/sizes_fast[0])**2 * np.log2(n/sizes_fast[0]+1) * times_fast_all[0] 
                        for n in sizes_fast]
    plt.semilogy(sizes_fast, n2log_theoretical, 'b--', alpha=0.3,
                label='N²log(N) teorico')
    
    plt.xlabel('Dimensione matrice (N)', fontsize=12)
    plt.ylabel('Tempo di esecuzione (s) - scala log', fontsize=12)
    plt.title('Confronto tempi di esecuzione: DCT Naive vs DCT Fast', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which="both", linestyle='--')
    plt.legend(loc='upper left', fontsize=10)
    
    # Aggiungi annotazioni
    plt.text(0.02, 0.98, 'Nota: Asse Y in scala logaritmica', 
            transform=plt.gca().transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Salva il grafico
    plt.savefig('risultati/confronto_dct_tempi.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n Grafico salvato in: risultati/confronto_dct_tempi.png")

def analyze_complexity(sizes, times_naive, times_fast):
    """
    Analizza e stampa la complessità computazionale
    """
    print("\n" + "=" * 60)
    print("ANALISI COMPLESSITÀ COMPUTAZIONALE")
    print("=" * 60)
    
    # Calcola i rapporti per verificare O(N³) per naive
    print("\n Verifica O(N³) per DCT Naive:")
    print("   Se l'algoritmo è O(N³), raddoppiando N il tempo dovrebbe")
    print("   aumentare di circa 8 volte (2³ = 8)")
    
    for i in range(1, len(sizes)):
        size_ratio = sizes[i] / sizes[i-1]
        time_ratio = times_naive[i] / times_naive[i-1]
        expected_ratio = size_ratio ** 3
        print(f"\n   N: {sizes[i-1]}→{sizes[i]} (×{size_ratio:.1f})")
        print(f"   Tempo: ×{time_ratio:.2f} (atteso: ×{expected_ratio:.1f})")
    
    # Calcola i rapporti per verificare O(N²logN) per fast
    print("\n\n Verifica O(N²log(N)) per DCT Fast:")
    print("   Raddoppiando N, il tempo dovrebbe aumentare di")
    print("   circa 4-5 volte (considerando il fattore logaritmico)")
    
    for i in range(1, min(len(sizes), len(times_fast))):
        size_ratio = sizes[i] / sizes[i-1]
        time_ratio = times_fast[i] / times_fast[i-1]
        n1, n2 = sizes[i-1], sizes[i]
        expected_ratio = (n2**2 * np.log2(n2)) / (n1**2 * np.log2(n1))
        print(f"\n   N: {sizes[i-1]}→{sizes[i]} (×{size_ratio:.1f})")
        print(f"   Tempo: ×{time_ratio:.2f} (atteso: ×{expected_ratio:.2f})")
    
    # Calcolo speedup
    print("\n\n SPEEDUP (Fast vs Naive):")
    for i, n in enumerate(sizes[:len(times_naive)]):
        speedup = times_naive[i] / times_fast[i]
        print(f"   N={n:4d}: {speedup:8.1f}× più veloce")

def create_results_folder():
    """Crea la cartella risultati se non esiste"""
    import os
    if not os.path.exists('risultati'):
        os.makedirs('risultati')
        print(" Creata cartella 'risultati/'")

if __name__ == "__main__":
    create_results_folder()
    
    # Esegui il benchmark
    sizes_n, times_n, sizes_f, times_f = benchmark_dct_implementations()
    
    print("\n" + "=" * 60)
    print(" PARTE 1 COMPLETATA!")
    print("=" * 60)
    print("\nRisultati salvati nella cartella 'risultati/'")
    print("Il grafico mostra chiaramente la differenza di complessità:")
    print("- DCT Naive: O(N³) - crescita cubica")
    print("- DCT Fast:  O(N²logN) - crescita molto più lenta")
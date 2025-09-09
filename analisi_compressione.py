# analisi_compressione.py 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import dct2_fast, idct2_fast
import os

def compress_image_dct(image, F, d):
    """Funzione di compressione DCT con gestione dimensioni"""
    h, w = image.shape
    num_blocks_v = h // F
    num_blocks_h = w // F
    
    # Crea immagine compressa con dimensioni ridotte (scarta avanzi)
    h_new = num_blocks_v * F
    w_new = num_blocks_h * F
    compressed = np.zeros((h_new, w_new))
    
    for i in range(num_blocks_v):
        for j in range(num_blocks_h):
            
            block = image[i*F:(i+1)*F, j*F:(j+1)*F].astype(np.float64)
            dct_block = dct2_fast(block)
            
            # Elimina frequenze alte
            for k in range(F):
                for l in range(F):
                    if k + l >= d:
                        dct_block[k, l] = 0
            
            idct_block = idct2_fast(dct_block)
            compressed[i*F:(i+1)*F, j*F:(j+1)*F] = np.clip(np.round(idct_block), 0, 255)
    
    return compressed.astype(np.uint8)

def analyze_compression_effects():
    """Analizza gli effetti della compressione su diverse immagini"""
    
    # Lista di immagini da testare
    test_images = [
        'bridge.bmp',
        'cathedral.bmp', 
        'gradient.bmp',
        '640x640.bmp'
    ]
    
    # Controlla quali immagini esistono realmente
    available_images = []
    for img_name in test_images:
        img_path = f'immagini/{img_name}'
        if os.path.exists(img_path):
            available_images.append(img_name)
            print(f" {img_name} trovata")
        else:
            print(f" {img_name} non trovata, skip...")
    
    # Se non ci sono immagini disponibili, esci
    if not available_images:
        print(" Nessuna immagine trovata nella cartella 'immagini/'")
        print("   Assicurati di avere almeno una delle seguenti immagini:")
        for img in test_images:
            print(f"   - {img}")
        return
    
    # Parametri da testare
    F = 8  # Dimensione blocco standard JPEG
    d_values = [1, 2, 4, 8, 12, 14]  # Diverse soglie
    
    fig, axes = plt.subplots(len(available_images), len(d_values) + 1, 
                             figsize=(20, 12))
    fig.suptitle(f'Effetti della Compressione DCT (F={F})', fontsize=16)
    
    # Se c'è solo una immagine, axes potrebbe non essere un array 2D
    if len(available_images) == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx, img_name in enumerate(available_images):
        img_path = f'immagini/{img_name}'
        print(f" Elaborando {img_name}...")
        
        try:
            # Carica immagine
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            
            # Mostra originale
            axes[img_idx, 0].imshow(img_array, cmap='gray')
            axes[img_idx, 0].set_title(f'Originale\n{img_name}\n{img_array.shape}')
            axes[img_idx, 0].axis('off')
            
            # Testa diverse soglie
            for d_idx, d in enumerate(d_values):
                print(f"    Compressione con d={d}...")
                
                # Comprimi
                compressed = compress_image_dct(img_array, F, d)
                
                # Per calcolo MSE, usa solo la parte dell'immagine che è stata compressa
                h_comp, w_comp = compressed.shape
                img_cropped = img_array[:h_comp, :w_comp]
                
                # Calcola metriche
                mse = np.mean((img_cropped.astype(float) - compressed.astype(float)) ** 2)
                psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
                
                # Calcola compressione
                kept = sum(1 for k in range(F) for l in range(F) if k+l < d)
                compression = (1 - kept/(F*F)) * 100
                
                # Mostra immagine compressa
                axes[img_idx, d_idx + 1].imshow(compressed, cmap='gray')
                axes[img_idx, d_idx + 1].set_title(
                    f'd={d}\nComp:{compression:.0f}%\nPSNR:{psnr:.1f}dB',
                    fontsize=9
                )
                axes[img_idx, d_idx + 1].axis('off')
                
        except Exception as e:
            print(f" Errore nell'elaborazione di {img_name}: {e}")
            # Nascondi gli assi in caso di errore
            for j in range(len(d_values) + 1):
                axes[img_idx, j].axis('off')
    
    plt.tight_layout()
    
    try:
        plt.savefig('risultati/analisi_compressione_completa.png', dpi=150, bbox_inches='tight')
        print(" Analisi salvata in: risultati/analisi_compressione_completa.png")
    except Exception as e:
        print(f" Errore nel salvare l'analisi: {e}")
    
    plt.show()

def plot_frequency_spectrum():
    """Visualizza lo spettro delle frequenze"""
    print(" Generando visualizzazione frequenze...")
    
    F = 8
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Visualizzazione del Taglio delle Frequenze (F=8)', fontsize=14)
    
    d_values = [2, 4, 6, 8, 12, 14]
    
    for idx, d in enumerate(d_values):
        ax = axes[idx // 3, idx % 3]
        
        # Crea matrice che mostra quali frequenze vengono mantenute
        mask = np.zeros((F, F))
        for k in range(F):
            for l in range(F):
                if k + l < d:
                    mask[k, l] = 1
        
        im = ax.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'd={d}\nMantenute: {np.sum(mask):.0f}/{F*F} coefficienti')
        ax.set_xlabel('Frequenza orizzontale')
        ax.set_ylabel('Frequenza verticale')
        
        # Aggiungi griglia
        for i in range(F+1):
            ax.axhline(i-0.5, color='gray', linewidth=0.5)
            ax.axvline(i-0.5, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    
    try:
        plt.savefig('risultati/frequenze_tagliate.png', dpi=150, bbox_inches='tight')
        print(" Visualizzazione frequenze salvata in: risultati/frequenze_tagliate.png")
    except Exception as e:
        print(f" Errore nel salvare le frequenze: {e}")
    
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("ANALISI AVANZATA COMPRESSIONE DCT")
    print("=" * 60)
    
    # Verifica struttura directory
    print(f" Directory corrente: {os.getcwd()}")
    print(f" Cartella 'immagini' esiste: {os.path.exists('immagini')}")
    print(f" Cartella 'risultati' esiste: {os.path.exists('risultati')}")
    
    # Crea cartella risultati se non esiste
    os.makedirs('risultati', exist_ok=True)
    
    print("\n Fase 1: Visualizzazione spettro frequenze")
    plot_frequency_spectrum()
    
    print("\n Fase 2: Analisi effetti compressione")
    analyze_compression_effects()
    
    print("\n Analisi completata!")
    print(" Per continuare, premi ENTER nel terminale...")
    input()  # Pausa finale per vedere tutti i messaggi
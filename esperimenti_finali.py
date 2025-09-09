# esperimenti_finali.py - VERSIONE CORRETTA
import os
import numpy as np
from PIL import Image
from utils import dct2_fast, idct2_fast
import pandas as pd

def compress_image(image, F, d):
    """Comprimi immagine con DCT gestendo dimensioni"""
    h, w = image.shape
    num_blocks_v = h // F
    num_blocks_h = w // F
    
    # Dimensioni finali dopo scarto avanzi
    h_new = num_blocks_v * F
    w_new = num_blocks_h * F
    compressed = np.zeros((h_new, w_new))
    
    for i in range(num_blocks_v):
        for j in range(num_blocks_h):
            block = image[i*F:(i+1)*F, j*F:(j+1)*F].astype(np.float64)
            dct_block = dct2_fast(block)
            
            for k in range(F):
                for l in range(F):
                    if k + l >= d:
                        dct_block[k, l] = 0
            
            idct_block = idct2_fast(dct_block)
            compressed[i*F:(i+1)*F, j*F:(j+1)*F] = np.clip(np.round(idct_block), 0, 255)
    
    return compressed.astype(np.uint8)

def run_experiments():
    """Esegui esperimenti sistematici per la relazione"""
    
    # Crea cartella risultati se non esiste
    os.makedirs('risultati', exist_ok=True)
    
    images = ['bridge.bmp', 'cathedral.bmp', 'gradient.bmp', '640x640.bmp', 'shoe.bmp']
    F_values = [4, 8, 16]
    d_percentages = [0.25, 0.5, 0.75, 1.0]  # Percentuale di d_max
    
    results = []
    
    for img_name in images:
        try:
            img = Image.open(f'immagini/{img_name}').convert('L')
            img_array = np.array(img)
            print(f"\nProcessando {img_name} - Dimensioni: {img_array.shape}")
            
            for F in F_values:
                d_max = 2 * F - 2
                
                for d_perc in d_percentages:
                    d = int(d_perc * d_max)
                    if d == 0:
                        d = 1
                    
                    # Comprimi
                    compressed = compress_image(img_array, F, d)
                    
                    # Calcola MSE usando solo la parte compressa
                    h_comp, w_comp = compressed.shape
                    img_cropped = img_array[:h_comp, :w_comp]
                    
                    mse = np.mean((img_cropped.astype(float) - compressed.astype(float)) ** 2)
                    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100
                    
                    kept = sum(1 for k in range(F) for l in range(F) if k+l < d)
                    compression_ratio = (1 - kept/(F*F)) * 100
                    
                    # Calcola pixel scartati
                    pixels_lost = (img_array.shape[0] * img_array.shape[1]) - (h_comp * w_comp)
                    pixels_lost_perc = (pixels_lost / (img_array.shape[0] * img_array.shape[1])) * 100
                    
                    results.append({
                        'Immagine': img_name.replace('.bmp', ''),
                        'Dim. Orig.': f"{img_array.shape[0]}x{img_array.shape[1]}",
                        'F': F,
                        'd': d,
                        'd/d_max': f"{d_perc:.0%}",
                        'Compressione %': f"{compression_ratio:.1f}",
                        'PSNR (dB)': f"{psnr:.2f}",
                        'Pixel Scartati %': f"{pixels_lost_perc:.1f}",
                        'Qualità': 'Eccellente' if psnr > 40 else 'Buona' if psnr > 30 else 'Accettabile' if psnr > 20 else 'Scarsa'
                    })
                    
        except FileNotFoundError:
            print(f"   File {img_name} non trovato, skip...")
            continue
    
    # Crea DataFrame e salva
    df = pd.DataFrame(results)
    
    # Salva in diversi formati
    df.to_csv('risultati/tabella_esperimenti.csv', index=False)
    df.to_html('risultati/tabella_esperimenti.html', index=False)
    
    print("\n" + "=" * 100)
    print(" TABELLA RISULTATI ESPERIMENTI")
    print("=" * 100)
    
    # Mostra alcune righe significative
    print("\n Esempi con F=8 (standard JPEG):")
    df_f8 = df[df['F'] == 8]
    print(df_f8[['Immagine', 'd', 'Compressione %', 'PSNR (dB)', 'Qualità']].to_string(index=False))
    
    print("\n Statistiche Generali:")
    print(f"  - Numero totale esperimenti: {len(df)}")
    print(f"  - PSNR medio: {df['PSNR (dB)'].apply(lambda x: float(x)).mean():.2f} dB")
    print(f"  - Migliore PSNR: {df['PSNR (dB)'].apply(lambda x: float(x)).max():.2f} dB")
    print(f"  - Peggiore PSNR: {df['PSNR (dB)'].apply(lambda x: float(x)).min():.2f} dB")
    
    return df

if __name__ == "__main__":
    # Installa pandas se non presente
    try:
        import pandas as pd
    except ImportError:
        import subprocess
        import sys
        print("Installazione pandas...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd
    
    df = run_experiments()
    print("\n Esperimenti completati! Risultati salvati in risultati/tabella_esperimenti.csv")
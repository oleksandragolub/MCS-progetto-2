# parte2_compressor.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
# Rimossi import matplotlib non utilizzati
from utils import dct2_fast, idct2_fast
import os

class DCTImageCompressor:
    def __init__(self, root):
        self.root = root
        self.root.title("DCT Image Compressor - MCS Progetto 2")
        self.root.geometry("1200x700")
        
        # Variabili
        self.original_image = None
        self.compressed_image = None
        self.image_path = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Crea l'interfaccia grafica"""
        
        # Frame principale
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame controlli (in alto)
        control_frame = tk.LabelFrame(main_frame, text="Controlli", bg='#f0f0f0', font=('Arial', 12, 'bold'))
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Bottone per caricare immagine
        tk.Button(control_frame, text=" Carica Immagine BMP", 
                 command=self.load_image, bg='#4CAF50', fg='white',
                 font=('Arial', 10, 'bold'), padx=20, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Separatore
        tk.Label(control_frame, text="  |  ", bg='#f0f0f0').pack(side=tk.LEFT)
        
        # Parametro F (dimensione blocco)
        tk.Label(control_frame, text="Dimensione Blocco (F):", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.F_var = tk.IntVar(value=8)
        # Aggiungi callback per aggiornare d_max quando F cambia
        self.F_var.trace_add("write", lambda *_: self._update_d_max())
        F_spinbox = tk.Spinbox(control_frame, from_=4, to=32, increment=4, 
                               textvariable=self.F_var, width=5)
        F_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Parametro d (soglia frequenze)
        tk.Label(control_frame, text="Soglia Frequenze (d):", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.d_var = tk.IntVar(value=8)
        self.d_spinbox = tk.Spinbox(control_frame, from_=0, to=14, 
                                    textvariable=self.d_var, width=5)
        self.d_spinbox.pack(side=tk.LEFT, padx=5)

        hint = "d=0 → blocchi neri   |   d=2F−2 → elimina solo (F−1,F−1)"
        tk.Label(control_frame, text=hint, bg='#f0f0f0', fg='#555').pack(side=tk.LEFT, padx=8)
        
        # Label per mostrare la percentuale di compressione
        self.compression_label = tk.Label(control_frame, text="", bg='#f0f0f0', fg='blue')
        self.compression_label.pack(side=tk.LEFT, padx=20)
        
        # Bottone per comprimere
        tk.Button(control_frame, text=" Comprimi", 
                 command=self.compress_image, bg='#2196F3', fg='white',
                 font=('Arial', 10, 'bold'), padx=20, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Bottone per salvare
        tk.Button(control_frame, text=" Salva Compressa", 
                 command=self.save_compressed, bg='#FF9800', fg='white',
                 font=('Arial', 10, 'bold'), padx=20, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Frame per le immagini
        images_frame = tk.Frame(main_frame, bg='#f0f0f0')
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Frame immagine originale
        self.original_frame = tk.LabelFrame(images_frame, text="Immagine Originale", 
                                           bg='white', font=('Arial', 10, 'bold'))
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_label = tk.Label(self.original_frame, bg='white', 
                                       text="Nessuna immagine caricata")
        self.original_label.pack(expand=True)
        
        # Frame immagine compressa
        self.compressed_frame = tk.LabelFrame(images_frame, text="Immagine Compressa", 
                                             bg='white', font=('Arial', 10, 'bold'))
        self.compressed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.compressed_label = tk.Label(self.compressed_frame, bg='white',
                                        text="Premi 'Comprimi' per vedere il risultato")
        self.compressed_label.pack(expand=True)
        
        # Status bar
        self.status_bar = tk.Label(main_frame, text="Pronto", bd=1, 
                                  relief=tk.SUNKEN, anchor=tk.W, bg='#e0e0e0')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _update_d_max(self):
        """Aggiorna il valore massimo di d quando F cambia"""
        F = self.F_var.get()
        max_d = max(0, 2 * F - 2)
        self.d_spinbox.config(to=max_d)
        if self.d_var.get() > max_d:
            self.d_var.set(max_d)
        
    def load_image(self):
        """Carica un'immagine BMP"""
        file_path = filedialog.askopenfilename(
            title="Seleziona un'immagine BMP",
            initialdir="immagini/",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Carica l'immagine
                img = Image.open(file_path)
                
                # Converti in scala di grigi se necessario
                if img.mode != 'L':
                    img = img.convert('L')
                
                self.original_image = np.array(img)
                self.image_path = file_path
                
                # Mostra l'immagine originale
                self.display_image(self.original_image, self.original_label, "originale")
                
                # Aggiorna status
                h, w = self.original_image.shape
                self.status_bar.config(text=f"Immagine caricata: {os.path.basename(file_path)} - Dimensioni: {w}x{h}")
                
                # Aggiorna il massimo valore di d
                self._update_d_max()
                
            except Exception as e:
                messagebox.showerror("Errore", f"Impossibile caricare l'immagine:\n{str(e)}")
    
    def compress_image(self):
        """Comprimi l'immagine usando DCT"""
        if self.original_image is None:
            messagebox.showwarning("Attenzione", "Prima carica un'immagine!")
            return
        
        F = self.F_var.get()
        d = self.d_var.get()

        h, w = self.original_image.shape
        if h < F or w < F:
            messagebox.showwarning(
                "Attenzione",
                "L'immagine è più piccola di F. Riduci F o usa un'immagine più grande."
            )
            return
        
        # Verifica che d sia valido
        max_d = 2 * F - 2
        if d > max_d:
            d = max_d
            self.d_var.set(d)
        
        self.status_bar.config(text=f"Compressione in corso... F={F}, d={d}")
        self.root.update()
        
        try:
            # Esegui la compressione
            self.compressed_image = self.dct_compress(self.original_image, F, d)
            
            # Mostra l'immagine compressa
            self.display_image(self.compressed_image, self.compressed_label, "compressa")
            
            # Calcola e mostra le statistiche
            self.show_compression_stats(F, d)
            
            self.status_bar.config(text=f"Compressione completata! F={F}, d={d}")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la compressione:\n{str(e)}")
            self.status_bar.config(text="Errore nella compressione")
    
    def dct_compress(self, image, F, d):
        """
        Algoritmo di compressione DCT
        """
        h, w = image.shape
        
        # Calcola il numero di blocchi
        num_blocks_v = h // F
        num_blocks_h = w // F
        
        # Crea l'immagine di output (scarta gli avanzi)
        compressed = np.zeros((num_blocks_v * F, num_blocks_h * F))
        
        # Processa ogni blocco
        for i in range(num_blocks_v):
            for j in range(num_blocks_h):
                # Estrai il blocco
                block = image[i*F:(i+1)*F, j*F:(j+1)*F].astype(np.float64)
                
                # Applica DCT2
                dct_block = dct2_fast(block)
                
                # Elimina le frequenze alte (k + l >= d)
                for k in range(F):
                    for l in range(F):
                        if k + l >= d:
                            dct_block[k, l] = 0
                
                # Applica DCT2 inversa
                idct_block = idct2_fast(dct_block)
                
                # Arrotonda e limita i valori
                idct_block = np.round(idct_block)
                idct_block = np.clip(idct_block, 0, 255)
                
                # Inserisci il blocco nell'immagine compressa
                compressed[i*F:(i+1)*F, j*F:(j+1)*F] = idct_block
        
        return compressed.astype(np.uint8)
    
    def display_image(self, img_array, label, title):
        """Mostra un'immagine in un label"""
        # Converti array in immagine PIL
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Ridimensiona se troppo grande (mantieni aspect ratio)
        max_size = (500, 500)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Converti in PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Aggiorna il label
        label.config(image=photo, text="")
        label.image = photo  # Mantieni riferimento
    
    def show_compression_stats(self, F, d):
        """Mostra le statistiche di compressione - CORRETTA"""
        # Calcola il rapporto di compressione teorico
        total_coeffs = F * F
        kept_coeffs = 0
        for k in range(F):
            for l in range(F):
                if k + l < d:
                    kept_coeffs += 1
        
        compression_ratio = (1 - kept_coeffs / total_coeffs) * 100
        
        # FIX: Calcola MSE usando le dimensioni corrette
        h_comp, w_comp = self.compressed_image.shape
        img_cropped = self.original_image[:h_comp, :w_comp]
        
        mse = np.mean((img_cropped.astype(float) - self.compressed_image.astype(float)) ** 2)
        
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
        else:
            psnr = float('inf')
        
        # Aggiorna label
        self.compression_label.config(
            text=f"Compressione: {compression_ratio:.1f}% | "
                 f"Coefficienti: {kept_coeffs}/{total_coeffs} | "
                 f"PSNR: {psnr:.2f} dB"
        )
    
    def save_compressed(self):
        """Salva l'immagine compressa"""
        if self.compressed_image is None:
            messagebox.showwarning("Attenzione", "Prima comprimi un'immagine!")
            return
        
        # Crea cartella risultati se non esiste
        os.makedirs('risultati', exist_ok=True)
        
        file_path = filedialog.asksaveasfilename(
            title="Salva immagine compressa",
            initialdir="risultati/",
            defaultextension=".bmp",
            filetypes=[("BMP files", "*.bmp"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                img = Image.fromarray(self.compressed_image)
                img.save(file_path)
                messagebox.showinfo("Successo", f"Immagine salvata in:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Errore", f"Impossibile salvare l'immagine:\n{str(e)}")

def main():
    root = tk.Tk()
    app = DCTImageCompressor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
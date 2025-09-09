# setup.py - Installa le dipendenze del progetto
import subprocess
import sys

def install_packages():
    """Installa tutti i pacchetti necessari per il progetto"""
    packages = [
        'numpy',
        'scipy',
        'matplotlib', 
        'pillow',
    ]
    
    for package in packages:
        print(f"Installazione di {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n Tutte le librerie sono state installate!")
    print("\nLibrerie installate:")
    print("- numpy: per operazioni su array")
    print("- scipy: per DCT fast (fft)")
    print("- matplotlib: per i grafici")
    print("- pillow: per leggere/salvare immagini")
    print("\nNota: tkinter è incluso con Python. Su Linux potrebbe")
    print("      servire: sudo apt-get install python3-tk")

if __name__ == "__main__":
    install_packages()
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os

folder_path = "../data"
box_downloads = Path(folder_path) / "box_downloads"
original_folder = Path(folder_path) / "original"
metahuman_folder = Path(folder_path) / "metahuman"
furhat_folder = Path(folder_path) / "furhat"

filename = "A102_gra_p_3.mov"

y1, sr1 = librosa.load(os.path.join(box_downloads, filename))
y2, sr2 = librosa.load(os.path.join(original_folder, filename))
y3, sr3 = librosa.load(os.path.join(metahuman_folder, filename))
y4, sr4 = librosa.load(os.path.join(furhat_folder, filename))

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
librosa.display.waveshow(y1, sr=sr1)
plt.title('From Box')

plt.subplot(4, 1, 2)
librosa.display.waveshow(y2, sr=sr2)
plt.title('Original')


plt.subplot(4, 1, 3)
librosa.display.waveshow(y3, sr=sr3)
plt.title('Metahuman')


plt.subplot(4, 1, 4)
librosa.display.waveshow(y4, sr=sr4)
plt.title('Furhat')


plt.suptitle('Waveform Comparison filename = ' + filename)
plt.tight_layout()
plt.show()
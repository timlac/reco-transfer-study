import os.path
import pandas as pd
from tensorboard.plugins import projector
from sklearn.preprocessing import StandardScaler

from constants import AU_INTENSITY_COLS

file_path = "../data/out/openface_data.csv"

log_dir = "projector_logs"
metadata_filename = "metadata.tsv"
tensor_filename = "tensor.tsv"

df = pd.read_csv(file_path)

meta = df[["condition",
           "filename",
           "video_id",
           "emotion",
           "intensity_level",
           "mode"]]
vec = df[AU_INTENSITY_COLS]

# Normalize the vectors
scaler = StandardScaler()
vec_normalized = scaler.fit_transform(vec)


print(vec.shape)  # Number of rows and columns in tensor
print(vec.isna().sum())
print(meta.shape)  # Number of rows and columns in metadata

meta.to_csv(os.path.join(log_dir, metadata_filename),
            sep='\t', index=False)

pd.DataFrame(vec_normalized).to_csv(os.path.join(log_dir, tensor_filename),
           sep='\t', index=False, header=False)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.metadata_path = metadata_filename
embedding.tensor_path = tensor_filename
projector.visualize_embeddings(log_dir, config)

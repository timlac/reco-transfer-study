import re
from pathlib import Path

# def clean_filename(filename):
#     return re.sub(r'_V1.*', '', filename)

def clean_filename(filename):
    # Remove everything after _V1 and also remove _f and _m
    filename = re.sub(r'_V1.*', '', filename)
    filename = re.sub(r'_[fm]$', '', filename)
    return filename

def rename_files_in_folder(folder_path):
    for file_path in Path(folder_path).glob("*.mov"):
        new_name = clean_filename(file_path.stem) + file_path.suffix
        new_path = file_path.with_name(new_name)
        file_path.rename(new_path)
        print(f'Renamed: {file_path} -> {new_path}')

# Example usage
rename_files_in_folder("../data/videos/furhat")
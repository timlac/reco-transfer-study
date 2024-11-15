from pathlib import Path
import shutil
import re

def clean_filename(filename):
    # Remove everything after _V1 and also remove _f and _m
    filename = re.sub(r'_V1.*', '', filename)
    filename = re.sub(r'_[fm]$', '', filename)
    return filename

def move_files_based_on_last_char(folder_path):
    furhat_folder = Path(folder_path) / "furhat"
    metahuman_folder = Path(folder_path) / "metahuman"
    original_folder = Path(folder_path) / "original"

    # Create directories if they don't exist
    furhat_folder.mkdir(exist_ok=True)
    metahuman_folder.mkdir(exist_ok=True)
    original_folder.mkdir(exist_ok=True)

    for file_path in Path(folder_path).glob("*.mov"):
        stem = file_path.stem
        last_char = stem[-1]

        if last_char == 'f':
            target_folder = furhat_folder
        elif last_char == 'm':
            target_folder = metahuman_folder
        elif last_char.isdigit():
            target_folder = original_folder
        else:
            continue  # Skip files that don't match the criteria

        new_name = clean_filename(file_path.stem) + file_path.suffix
        target_path = target_folder / new_name
        shutil.move(str(file_path), str(target_path))
        print(f'Moved: {file_path} -> {target_path}')

# Example usage
move_files_based_on_last_char("data")
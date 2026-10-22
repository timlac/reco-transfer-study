"""Download selected single-emotion clips from Edited/for validation study.

Only the 18 candidate video IDs below are eligible, and IDs currently present
in either local dataset are always excluded. All intensities, both modes,
and all filename versions are included. Box subfolders are preserved locally.
The observed filename aliases A0506/A0526 are compared as A506/A526 in both
the Box listing and the CSVs; original filenames are preserved on download.

Run again with the same output directory to continue: completed files are
verified and skipped. Existing files that differ from Box are left untouched
and reported as errors. Interrupted files restart from the beginning.

Preview without downloading:
    .venv/bin/python box_scripts/download_selected_emotions.py --dry-run

Download using BOX_DEV_TOKEN from the project .env:
    .venv/bin/python box_scripts/download_selected_emotions.py
"""

import argparse
import csv
import hashlib
import os
from collections import Counter
from contextlib import closing
from pathlib import Path

from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth
from dotenv import load_dotenv
from nexa_sentimotion_filename_parser.metadata import Metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_EMOTIONS = {
    26: "boredom",
    27: "concentration",
    1: "confusion",
    2: "determination",
    37: "disappointment",
    3: "doubt",
    7: "gratitude",
    29: "interest_curiosity",
}
CANDIDATE_VIDEO_IDS = frozenset({
    "A2301", "A2302", "A2304", "A2305", "A2307", "A2308", "A2311", "A2313",
    "A501", "A503", "A504", "A505", "A506", "A507", "A520", "A521", "A525", "A526",
})
VIDEO_ID_ALIASES = {"A0506": "A506", "A0526": "A526"}


def canonical_video_id(video_id):
    return VIDEO_ID_ALIASES.get(video_id, video_id)


def read_video_ids(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing video_id column in {csv_path}")
        return {
            canonical_video_id(row["video_id"].strip())
            for row in reader if (row.get("video_id") or "").strip()
        }


def iter_folder_items(client, folder_id):
    marker = None
    while True:
        response = client.folders.get_folder_items(
            folder_id=folder_id,
            usemarker=True,
            marker=marker,
            limit=1000,
            fields=["id", "type", "name", "size", "sha1"],
        )
        yield from response.entries
        marker = response.next_marker
        if not marker:
            break


def find_folder(client, parent_id, name):
    matches = [
        item for item in iter_folder_items(client, parent_id)
        if item.type == "folder" and item.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one folder named {name!r} under Box folder {parent_id}.")
    return matches[0].id


def iter_files(client, folder_id, relative_path):
    for item in iter_folder_items(client, folder_id):
        if Path(item.name).name != item.name or item.name in {".", ".."}:
            raise ValueError(f"Invalid Box path component: {item.name!r}")
        path = relative_path / item.name
        if item.type == "folder":
            yield from iter_files(client, item.id, path)
        elif item.type == "file":
            yield item, path


def selected_metadata(filename, folder_video_id, eligible_ids):
    try:
        metadata = Metadata(Path(filename).stem)
    except (ValueError, AssertionError, IndexError):
        print(f"Skipping unparseable filename: {filename}")
        return None
    if (
        canonical_video_id(metadata.video_id) != folder_video_id
        or canonical_video_id(metadata.video_id) not in eligible_ids
        or metadata.emotion_1_id not in SELECTED_EMOTIONS
        or metadata.mix
    ):
        return None
    return metadata


def collect_downloads(client, eligible_ids):
    edited_id = find_folder(client, "0", "Edited")
    validation_id = find_folder(client, edited_id, "for validation study")
    folders = [
        item for item in iter_folder_items(client, validation_id)
        if item.type == "folder" and item.name in eligible_ids
    ]
    missing = eligible_ids - {folder.name for folder in folders}
    if missing:
        print(f"Candidate folders missing from validation: {', '.join(sorted(missing))}")

    selected = []
    for folder in sorted(folders, key=lambda item: item.name):
        count = 0
        for item, path in iter_files(client, folder.id, Path(folder.name)):
            metadata = selected_metadata(item.name, folder.name, eligible_ids)
            if metadata is not None:
                selected.append((item, path, metadata))
                count += 1
        print(f"{folder.name}: {count} matching files", flush=True)
    return sorted(selected, key=lambda entry: str(entry[1]))


def file_matches(path, item):
    if not path.is_file() or path.stat().st_size != item.size:
        return False
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item.sha_1


def download_file(client, item, destination):
    """Skip verified downloads; publish new files only after size/SHA-1 checks."""
    if item.size is None or not item.sha_1:
        raise RuntimeError(f"Box did not provide size/SHA-1 for {item.name}.")
    if destination.exists():
        if file_matches(destination, item):
            return False
        raise FileExistsError(f"Existing file differs from Box; move or remove it first: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{item.id}.part")
    try:
        stream = client.downloads.download_file(item.id)
        if stream is None:
            raise RuntimeError(f"Box has not made {item.name} ready for download; retry later.")
        with closing(stream), temporary.open("wb") as handle:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                handle.write(chunk)
        if not file_matches(temporary, item):
            raise RuntimeError(f"Downloaded size/SHA-1 does not match Box: {item.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="List matching files without downloading or creating directories.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/videos/box_downloads/selected_emotions")
    parser.add_argument("--internal-csv", type=Path, default=PROJECT_ROOT / "data/out/openface_data.csv")
    parser.add_argument("--external-csv", type=Path, default=PROJECT_ROOT / "data/out/openface_data_external.csv")
    args = parser.parse_args(argv)

    existing_ids = read_video_ids(args.internal_csv) | read_video_ids(args.external_csv)
    eligible_ids = CANDIDATE_VIDEO_IDS - existing_ids
    print(f"Eligible candidate IDs: {len(eligible_ids)} / {len(CANDIDATE_VIDEO_IDS)}")
    if CANDIDATE_VIDEO_IDS & existing_ids:
        print(f"Excluded IDs already in local datasets: {', '.join(sorted(CANDIDATE_VIDEO_IDS & existing_ids))}")
    if not eligible_ids:
        return

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("BOX_DEV_TOKEN")
    if not token:
        raise RuntimeError(f"Set BOX_DEV_TOKEN in {PROJECT_ROOT / '.env'} or the environment.")
    client = BoxClient(auth=BoxDeveloperTokenAuth(token=token))
    print("ID aliases for comparisons: A0506 = A506, A0526 = A526 (filenames preserved).")
    selected = collect_downloads(client, eligible_ids)
    for item, _, _ in selected:
        if item.size is None or not item.sha_1:
            raise RuntimeError(f"Box did not provide size/SHA-1 for {item.name}.")
    emotion_counts = Counter(metadata.emotion_1_id for _, _, metadata in selected)
    print(f"\nMatched {len(selected)} files across {len({metadata.video_id for _, _, metadata in selected})} video IDs.")
    for emotion_id, name in SELECTED_EMOTIONS.items():
        print(f"  {name} ({emotion_id}): {emotion_counts[emotion_id]}")
    total_bytes = sum(item.size or 0 for item, _, _ in selected)
    print(f"Total Box file size: {total_bytes / 1024 ** 3:.2f} GiB")
    print(f"Output directory: {args.output_dir.resolve()}")

    downloaded = 0
    for index, (item, relative_path, metadata) in enumerate(selected, start=1):
        if args.dry_run:
            print(f"Would download: {relative_path} [{SELECTED_EMOTIONS[metadata.emotion_1_id]}]")
        else:
            is_new = download_file(client, item, args.output_dir / relative_path)
            downloaded += is_new
            action = "Downloaded" if is_new else "Already downloaded (verified)"
            print(f"[{index}/{len(selected)}] {action}: {relative_path}", flush=True)
    if args.dry_run:
        print("Dry run complete. No files downloaded.")
    else:
        print(f"Finished: {downloaded} downloaded, {len(selected) - downloaded} already present.")


if __name__ == "__main__":
    main()

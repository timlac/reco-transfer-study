"""Offline checks for selection rules and download integrity."""

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from box_sdk_gen.schemas.file_full import FileFull

from box_scripts import download_selected_emotions as downloader


def box_file(name, contents=b"test clip"):
    return FileFull(
        id=name, name=name, size=len(contents),
        sha_1=hashlib.sha1(contents).hexdigest(),
    )


def box_folder(name):
    return SimpleNamespace(id=name, name=name, type="folder")


class DownloadSelectedEmotionsTests(unittest.TestCase):
    def test_zero_padded_aliases_match_and_are_excluded_by_csv(self):
        for alias, video_id in downloader.VIDEO_ID_ALIASES.items():
            with self.subTest(alias=alias):
                filename = f"{alias}_bor_p_2.mov"
                metadata = downloader.selected_metadata(filename, video_id, {video_id})
                self.assertEqual(metadata.video_id, alias)
                with tempfile.TemporaryDirectory() as directory:
                    csv_path = Path(directory) / "data.csv"
                    csv_path.write_text(f"video_id\n{alias}\n", encoding="utf-8")
                    excluded = downloader.read_video_ids(csv_path)
                self.assertEqual(excluded, {video_id})
                self.assertIsNone(downloader.selected_metadata(
                    filename, video_id, downloader.CANDIDATE_VIDEO_IDS - excluded,
                ))

    def test_all_selected_emotions_and_candidates(self):
        abbreviations = {26: "bor", 27: "conc", 1: "conf", 2: "det", 37: "disa", 3: "dou", 7: "gra", 29: "int"}
        self.assertEqual(len(downloader.CANDIDATE_VIDEO_IDS), 18)
        for video_id in downloader.CANDIDATE_VIDEO_IDS:
            for emotion_id, abbreviation in abbreviations.items():
                with self.subTest(video_id=video_id, emotion_id=emotion_id):
                    metadata = downloader.selected_metadata(
                        f"{video_id}_{abbreviation}_v_3_ver2.mov", video_id,
                        downloader.CANDIDATE_VIDEO_IDS,
                    )
                    self.assertEqual(metadata.emotion_1_id, emotion_id)

    def test_intensity_does_not_restrict_selection(self):
        for intensity in (1, 2, 3, 4, None):
            with self.subTest(intensity=intensity):
                suffix = f"_{intensity}" if intensity is not None else ""
                metadata = downloader.selected_metadata(
                    f"A501_bor_p{suffix}.mov", "A501", {"A501"},
                )
                self.assertIsNotNone(metadata)
                self.assertEqual(metadata.intensity_level, intensity)

    def test_rejects_other_emotions_mixes_wrong_ids_and_malformed_names(self):
        cases = [
            ("A501_ang_p_2.mov", "A501", {"A501"}),
            ("A501_mix_bor_dou_50_50_p_2.mov", "A501", {"A501"}),
            ("A503_bor_p_2.mov", "A501", {"A501", "A503"}),
            ("A501_bor_p_2.mov", "A501", {"A503"}),
            ("invalid.mov", "A501", {"A501"}),
            ("A501_not_an_emotion.mov", "A501", {"A501"}),
        ]
        with redirect_stdout(io.StringIO()):
            for args in cases:
                with self.subTest(filename=args[0]):
                    self.assertIsNone(downloader.selected_metadata(*args))

    def test_marker_pagination(self):
        client = Mock()
        client.folders.get_folder_items.side_effect = [
            SimpleNamespace(entries=["first"], next_marker="page-two"),
            SimpleNamespace(entries=["second"], next_marker=None),
        ]
        self.assertEqual(list(downloader.iter_folder_items(client, "folder")), ["first", "second"])
        calls = client.folders.get_folder_items.call_args_list
        self.assertIsNone(calls[0].kwargs["marker"])
        self.assertEqual(calls[1].kwargs["marker"], "page-two")

    def test_dry_run_excludes_both_datasets_and_never_downloads(self):
        client = Mock()
        tree = {
            "0": [box_folder("Edited")],
            "Edited": [box_folder("for validation study"), box_folder("A520")],
            "for validation study": [box_folder(name) for name in ["A2301", "A501", "A503", "A999"]],
            "A2301": [
                box_file("A2301_bor_p_1.mov"), box_file("A2301_bor_p_2.mov"),
                box_file("A2301_bor_p_4.mov"), box_folder("nested"),
            ],
            "nested": [box_file("A2301_int_p_3.mov"), box_file("A2301_ang_p_2.mov")],
        }
        client.folders.get_folder_items.side_effect = lambda **kwargs: SimpleNamespace(
            entries=tree[kwargs["folder_id"]], next_marker=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            internal, external = root / "internal.csv", root / "external.csv"
            internal.write_text("video_id\nA501\nA501\n\n", encoding="utf-8")
            external.write_text("video_id\nA503\n", encoding="utf-8")
            output = root / "downloads"
            stdout = io.StringIO()
            with (
                patch.object(downloader, "BoxClient", return_value=client),
                patch.object(downloader, "load_dotenv"),
                patch.dict("os.environ", {"BOX_DEV_TOKEN": "test-token"}),
                redirect_stdout(stdout),
            ):
                downloader.main([
                    "--dry-run", "--internal-csv", str(internal),
                    "--external-csv", str(external), "--output-dir", str(output),
                ])
            client.downloads.download_file.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn("Matched 4 files across 1 video IDs", stdout.getvalue())
            self.assertIn("A2301/nested/A2301_int_p_3.mov", stdout.getvalue())
            self.assertIn("A2301_bor_p_1.mov", stdout.getvalue())
            self.assertIn("A2301_bor_p_4.mov", stdout.getvalue())
            self.assertIn("Excluded IDs already in local datasets: A501, A503", stdout.getvalue())
            visited = {call.kwargs["folder_id"] for call in client.folders.get_folder_items.call_args_list}
            self.assertFalse(visited & {"A501", "A503", "A520", "A999"})

    def test_missing_video_id_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("filename\nA501_bor_p_1.mov\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                downloader.read_video_ids(path)

    def test_download_is_verified_and_skipped_on_rerun(self):
        client = Mock()
        item = box_file("A501_bor_p_1.mov")
        client.downloads.download_file.return_value = io.BytesIO(b"test clip")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "A501" / item.name
            self.assertTrue(downloader.download_file(client, item, destination))
            self.assertEqual(destination.read_bytes(), b"test clip")
            self.assertFalse(downloader.download_file(client, item, destination))
            client.downloads.download_file.assert_called_once_with(item.id)
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_corrupt_download_is_removed(self):
        client = Mock()
        item = box_file("A501_bor_p_1.mov")
        client.downloads.download_file.return_value = io.BytesIO(b"bad video")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / item.name
            with self.assertRaises(RuntimeError):
                downloader.download_file(client, item, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_different_file_is_not_overwritten(self):
        client = Mock()
        item = box_file("A501_bor_p_1.mov")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / item.name
            destination.write_bytes(b"local file")
            with self.assertRaises(FileExistsError):
                downloader.download_file(client, item, destination)
            self.assertEqual(destination.read_bytes(), b"local file")
            client.downloads.download_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()

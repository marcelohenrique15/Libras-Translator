import csv
import tempfile
import unittest
from pathlib import Path

from dataset.manifest_builder import ManifestBuilder
from dataset.manifest_splitter import ManifestSplitter


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)

    def _video(self, relative_path):
        path = self.root / "raw" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _read(self, path):
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return reader.fieldnames, list(reader)

    def test_minds_filters_repetitions_outside_one_to_five(self):
        for repetition in ("0", "1", "2", "5", "6", "7"):
            self._video(f"signer_01/01AcontecerSinalizador01-{repetition}.mp4")

        for dataset_format in ("minds_libras", "auto"):
            with self.subTest(dataset_format=dataset_format):
                manifest = ManifestBuilder(self.root, dataset_format).build()
                _, rows = self._read(manifest)
                self.assertEqual(len(rows), 3)
                self.assertEqual({row["repetition"] for row in rows}, {"1", "2", "5"})
                self.assertEqual({row["class_id"] for row in rows}, {"01"})
                self.assertEqual({row["signer_id"] for row in rows}, {"01"})
                self.assertEqual({row["label"] for row in rows}, {"Acontecer"})
                self.assertTrue(all((self.root / row["path"]).exists() for row in rows))
        self.assertEqual(len(list((self.root / "raw").rglob("*.mp4"))), 6)

    def test_class_signer_preserves_free_names_repetitions_and_signer_ids(self):
        for path in (
            "Acontecer/1/ensaio livre.mp4",
            "Acontecer/signer-A/6.mp4",
            "Aluno/signer-A/7.mp4",
            "Aluno/1/take-final.mp4",
        ):
            self._video(path)

        for dataset_format in ("class_signer", "auto"):
            with self.subTest(dataset_format=dataset_format):
                manifest = ManifestBuilder(self.root, dataset_format).build()
                _, rows = self._read(manifest)
                self.assertEqual(len(rows), 4)
                self.assertEqual({row["signer_id"] for row in rows}, {"1", "signer-A"})
                self.assertEqual({row["repetition"] for row in rows}, {"ensaio livre", "6", "7", "take-final"})
                self.assertEqual({row["class_id"] for row in rows}, {"Acontecer", "Aluno"})
                self.assertEqual(len({row["sample_id"] for row in rows}), 4)
                self.assertTrue(all(row["label"] == row["class_id"] for row in rows))
                self.assertTrue(all((self.root / row["path"]).exists() for row in rows))

    def test_splitter_preserves_extra_columns_and_all_rows_when_updating_set(self):
        original = [
            {"sample_id": "a", "signer_id": "1", "repetition": "6", "camera": "left", "note": "Vídeo A"},
            {"sample_id": "b", "signer_id": "signer-A", "repetition": "take-final", "camera": "right", "note": "Vídeo B"},
            {"sample_id": "c", "signer_id": "02", "repetition": "99", "camera": "front", "note": "Vídeo C"},
        ]
        for existing_set in (False, True):
            with self.subTest(existing_set=existing_set):
                columns = ["sample_id", "signer_id", "repetition", "camera", "note"]
                if existing_set:
                    columns.insert(1, "set")
                path = self.root / "manifest.csv"
                with path.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    for row in original:
                        writer.writerow({**row, **({"set": "old_value"} if existing_set else {})})

                splitter = ManifestSplitter()
                splitter.split(path, "signer-A")
                actual_columns, rows = self._read(path)
                self.assertEqual(actual_columns, columns if existing_set else [*columns, "set"])
                self.assertEqual([row["set"] for row in rows], ["train", "test", "train"])
                self.assertEqual([{key: value for key, value in row.items() if key != "set"} for row in rows], original)

                splitter.split(path, "1")
                updated_columns, rows = self._read(path)
                self.assertEqual(updated_columns, actual_columns)
                self.assertEqual(updated_columns.count("set"), 1)
                self.assertEqual([row["set"] for row in rows], ["test", "train", "train"])
                self.assertEqual([{key: value for key, value in row.items() if key != "set"} for row in rows], original)


if __name__ == "__main__":
    unittest.main()

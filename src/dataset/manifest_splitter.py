from pathlib import Path
import csv


class ManifestSplitter():

    #Público
    def split(self, manifest_path: Path, test_signer_id: str) -> None:
        rows = self._filter_rows(self._read_csv(manifest_path))

        for row in rows:
            if row["signer_id"] == test_signer_id:
                row["set"] = "test"

            else:
                row["set"] = "train"

        self._write_csv(manifest_path, rows)

        return


    #Privado
    def _read_csv(self, manifest_path: Path) -> list[dict[str, str]]:
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        return rows

    def _write_csv(self, manifest_path: Path, rows: list[dict[str, str]]) -> None:
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "sample_id",
                    "class_id",
                    "label",
                    "signer_id",
                    "repetition",
                    "path",
                    "set"
                ]
            )
            writer.writeheader()
            writer.writerows(rows)
    
    def _filter_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        valid_repetitions = {"1", "2", "3", "4", "5"}
        valid_rows = []

        for row in rows:
            if row["repetition"] in valid_repetitions:
                valid_rows.append(row)

        return valid_rows
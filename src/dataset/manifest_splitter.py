from pathlib import Path
import csv
import tempfile


class ManifestSplitter():

    #Público
    def split(self, manifest_path: Path, test_signer_id: str) -> None:
        rows, columns = self._read_csv(manifest_path)
        if "set" not in columns:
            columns.append("set")

        for row in rows:
            if row["signer_id"] == test_signer_id:
                row["set"] = "test"

            else:
                row["set"] = "train"

        self._write_csv(manifest_path, rows, columns)

    #Privado
    def _read_csv(self, manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            columns = list(reader.fieldnames)
            rows = list(reader)

        return rows, columns

    def _write_csv(self, manifest_path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=manifest_path.parent, suffix=".csv.tmp", delete=False, encoding="utf-8", newline="") as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        temporary_path.replace(manifest_path)

from pathlib import Path
import redivis
from kneeassist.config import SAMPLE_DATA_DIR
from kneeassist.config import RAW_DATA_DIR

class MRNetDownloader:
    """
    Handles downloading MRI files from the Stanford MRNet dataset.
    """

    DATASET_REFERENCE = "aimi.mrnet_knee_mri_s:4a2c:v1_0"

    TRAIN_TABLE = "train:5wjf"
    VALID_TABLE = "valid:zcmk"

    def __init__(self):

        self.dataset = redivis.dataset(self.DATASET_REFERENCE)

    def get_train_table(self):

        return self.dataset.table(self.TRAIN_TABLE)

    def get_validation_table(self):

        return self.dataset.table(self.VALID_TABLE)

    def list_train_files(self):

        train = self.get_train_table()

        return train.list_files()

    def download_sample(self, index: int = 0):
        """
        Download a single MRI scan from the training dataset.
        Useful for testing and exploration.
        """

        # Ensure the output directory exists
        SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Get all training files
        files = self.list_train_files()

        # Select one file
        file = files[index]

        print(f"Downloading {file.name}...")

        # Download it
        file.download(SAMPLE_DATA_DIR)

        print("✅ Download complete.")
    def download_table(self, table_name: str, output_dir: Path):
        """
        Download all files from a Redivis table.
        """

        table = self.dataset.table(table_name)

        files = table.list_files()

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {len(files)} files to {output_dir}")

        for i, file in enumerate(files, start=1):

            destination = output_dir / file.name

            if destination.exists():
                print(f"[{i}/{len(files)}] Skipping {file.name}")
                continue

            print(f"[{i}/{len(files)}] Downloading {file.name}")

            file.download(output_dir)

        print("Download complete!")
    def download_train(self):
        self.download_table(
            self.TRAIN_TABLE,
            RAW_DATA_DIR / "train"
        )


    def download_validation(self):
        self.download_table(
            self.VALID_TABLE,
            RAW_DATA_DIR / "validation"
        )
    def download_all(self):
        """
        Download the complete MRNet dataset.
        """

        self.download_train()
        self.download_validation()
        print("✅ All MRNet files downloaded.")
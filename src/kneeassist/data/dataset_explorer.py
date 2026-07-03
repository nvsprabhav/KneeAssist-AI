from __future__ import annotations
import redivis
import matplotlib.pyplot as plt
import pandas as pd

class DatasetExplorer:
    """
    Explore the Stanford MRNet dataset metadata.
    """

    DATASET_REFERENCE = "aimi.mrnet_knee_mri_s:4a2c:v1_0"

    TRAIN_TABLE = "train:5wjf"
    VALID_TABLE = "valid:zcmk"

    TRAIN_ACL = "train_acl:v9mj"
    VALID_ACL = "valid_acl:sqwh"

    TRAIN_MENISCUS = "train_meniscus:3x4c"
    VALID_MENISCUS = "valid_meniscus:nqng"

    TRAIN_ABNORMAL = "train_abnormal:4v5w"
    VALID_ABNORMAL = "valid_abnormal:851r"
    def _label_counts(self, df: pd.DataFrame):
        """
        Return positive and negative counts from a label table.
        """

        label_column = df.columns[-1]

        counts = df[label_column].value_counts()

        positive = int(counts.get(1, 0))
        negative = int(counts.get(0, 0))

        return positive, negative

    def __init__(self):
        self.dataset = redivis.dataset(self.DATASET_REFERENCE)

    def load_table(self, table_name: str) -> pd.DataFrame:
        table = self.dataset.table(table_name)
        return table.to_pandas_dataframe()

    def train_files(self):
        return self.load_table(self.TRAIN_TABLE)

    def validation_files(self):
        return self.load_table(self.VALID_TABLE)

    def train_acl_labels(self):
        return self.load_table(self.TRAIN_ACL)

    def validation_acl_labels(self):
        return self.load_table(self.VALID_ACL)

    def train_meniscus_labels(self):
        return self.load_table(self.TRAIN_MENISCUS)

    def validation_meniscus_labels(self):
        return self.load_table(self.VALID_MENISCUS)

    def train_abnormal_labels(self):
        return self.load_table(self.TRAIN_ABNORMAL)

    def validation_abnormal_labels(self):
        return self.load_table(self.VALID_ABNORMAL)
    def dataset_summary(self):
        """
        Print a summary of the MRNet dataset.
        """

        train = self.train_files()
        valid = self.validation_files()

        train_acl = self.train_acl_labels()
        valid_acl = self.validation_acl_labels()

        train_meniscus = self.train_meniscus_labels()
        valid_meniscus = self.validation_meniscus_labels()

        train_abnormal = self.train_abnormal_labels()
        valid_abnormal = self.validation_abnormal_labels()

        train_acl_pos, train_acl_neg = self._label_counts(train_acl)
        valid_acl_pos, valid_acl_neg = self._label_counts(valid_acl)

        train_men_pos, train_men_neg = self._label_counts(train_meniscus)
        valid_men_pos, valid_men_neg = self._label_counts(valid_meniscus)

        train_ab_pos, train_ab_neg = self._label_counts(train_abnormal)
        valid_ab_pos, valid_ab_neg = self._label_counts(valid_abnormal)

        print("=" * 60)
        print("MRNET DATASET SUMMARY")
        print("=" * 60)

        print(f"Training MRI Files   : {len(train)}")
        print(f"Validation MRI Files : {len(valid)}")

        print("\nACL")
        print(f"Train Positive : {train_acl_pos}")
        print(f"Train Negative : {train_acl_neg}")
        print(f"Valid Positive : {valid_acl_pos}")
        print(f"Valid Negative : {valid_acl_neg}")

        print("\nMeniscus")
        print(f"Train Positive : {train_men_pos}")
        print(f"Train Negative : {train_men_neg}")
        print(f"Valid Positive : {valid_men_pos}")
        print(f"Valid Negative : {valid_men_neg}")

        print("\nAbnormal")
        print(f"Train Positive : {train_ab_pos}")
        print(f"Train Negative : {train_ab_neg}")
        print(f"Valid Positive : {valid_ab_pos}")
        print(f"Valid Negative : {valid_ab_neg}")

        print("=" * 60)
    def plot_label_distribution(self, df, title):
        """
        Plot the distribution of binary labels.
        """

        label_column = df.columns[-1]

        counts = (
            df[label_column]
            .value_counts()
            .sort_index()
        )

        labels = ["Negative", "Positive"]

        plt.figure(figsize=(6, 4))

        bars = plt.bar(labels, counts)

        plt.title(title)
        plt.ylabel("Number of Exams")

        # Add count labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 5,
                f"{int(height)}",
                ha="center",
            )

        plt.show()
    def plot_acl_distribution(self):
        self.plot_label_distribution(
            self.train_acl_labels(),
            "ACL Tear Distribution (Training Set)"
        )


    def plot_meniscus_distribution(self):
        self.plot_label_distribution(
            self.train_meniscus_labels(),
            "Meniscus Tear Distribution (Training Set)"
        )


    def plot_abnormal_distribution(self):
        self.plot_label_distribution(
            self.train_abnormal_labels(),
            "Abnormal MRI Distribution (Training Set)"
        )
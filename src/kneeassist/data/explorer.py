from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class MRIExplorer:
    """
    Utility class for exploring MRI scans stored as .npy files.
    """

    def __init__(self):
        self.mri = None
        self.path = None

    def load(self, path: str | Path):
        """
        Load an MRI volume from disk.
        """
        self.path = Path(path)
        self.mri = np.load(self.path)

        return self

    def summary(self):
        """
        Print summary statistics for the MRI volume.
        """
        if self.mri is None:
            raise ValueError("No MRI loaded. Call load() first.")

        print("=" * 60)
        print("MRI SUMMARY")
        print("=" * 60)

        print(f"File Name           : {self.path.name}")
        print(f"Shape               : {self.mri.shape}")
        print(f"Data Type           : {self.mri.dtype}")
        print(f"Minimum Pixel Value : {self.mri.min()}")
        print(f"Maximum Pixel Value : {self.mri.max()}")
        print(f"Mean Pixel Value    : {self.mri.mean():.2f}")
        print(f"Std Pixel Value     : {self.mri.std():.2f}")
        print(f"Number of Slices    : {self.mri.shape[0]}")

        print("=" * 60)

    def show_slice(self, index: int):
        """
        Display a single MRI slice.
        """
        if self.mri is None:
            raise ValueError("No MRI loaded. Call load() first.")

        if index >= self.mri.shape[0]:
            raise IndexError("Slice index out of range.")

        plt.figure(figsize=(6, 6))
        plt.imshow(self.mri[index], cmap="gray")
        plt.title(f"Slice {index}")
        plt.axis("off")
        plt.show()

    def show_middle_slice(self):
        """
        Display the middle MRI slice.
        """
        middle = self.mri.shape[0] // 2
        self.show_slice(middle)

    def show_multiple_slices(self, num_slices: int = 10):
        """
        Display evenly spaced MRI slices.
        """
        if self.mri is None:
            raise ValueError("No MRI loaded. Call load() first.")

        indices = np.linspace(
            0,
            self.mri.shape[0] - 1,
            num_slices,
            dtype=int,
        )

        rows = 2
        cols = 5

        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))

        for ax, idx in zip(axes.flatten(), indices):
            ax.imshow(self.mri[idx], cmap="gray")
            ax.set_title(f"Slice {idx}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def pixel_histogram(self):
        """
        Plot histogram of pixel intensities.
        """
        if self.mri is None:
            raise ValueError("No MRI loaded. Call load() first.")

        plt.figure(figsize=(8, 5))

        plt.hist(
            self.mri.ravel(),
            bins=100,
        )

        plt.title("Pixel Intensity Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

        plt.show()

    def slice_dimensions(self):
        """
        Print slice dimensions.
        """
        if self.mri is None:
            raise ValueError("No MRI loaded. Call load() first.")

        print(f"Height : {self.mri.shape[1]}")
        print(f"Width  : {self.mri.shape[2]}")

    def get_shape(self):
        """
        Return MRI shape.
        """
        return self.mri.shape

    def get_num_slices(self):
        """
        Return number of slices.
        """
        return self.mri.shape[0]
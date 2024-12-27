import os
import cv2
from torch.utils.data import Dataset
from imutils import paths
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


class ImageDataset(Dataset):
    """
    A PyTorch Dataset for loading images and their corresponding labels
    from a specified folder path.
    """

    def __init__(self, folder_path, transform=None):
        """
        Initialize the dataset by loading image paths and labels.

        Args:
            folder_path (str): Path to the folder containing images organized by class.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.transform = transform
        self.image_paths, self.transformed_labels, self.classes = self._prepare_data(folder_path)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetch a single sample (image and its corresponding label) by index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the transformed image and its one-hot encoded label.
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        label = self.transformed_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def _prepare_data(folder_path):
        """
        Prepare and process image paths and labels.

        Args:
            folder_path (str): Path to the dataset folder.

        Returns:
            tuple: (image_paths, transformed_labels, classes)
        """
        print("[INFO] Preparing the dataset...")
        image_paths = sorted(list(paths.list_images(folder_path)))
        labels = []

        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Load the image to ensure it's valid
                cv2.imread(image_path)
            except Exception as e:
                print(f"[WARNING] Skipping invalid image: {image_path} | Error: {e}")
                continue
            # Extract class labels from the folder structure
            label = image_path.split(os.path.sep)[-2].split("_")
            labels.append(label)

        # One-hot encode the labels
        mlb = MultiLabelBinarizer()
        transformed_labels = mlb.fit_transform(labels)
        print(f"[INFO] # Classes: {len(mlb.classes_)}")
        print(f"[INFO] Classes: {mlb.classes_}")

        return image_paths, transformed_labels, mlb.classes_

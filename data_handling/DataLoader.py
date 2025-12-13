from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
import json

class DatasetLoader:
    def __init__(self):
        self.train_loader = None
        self.test_loader = None

        self.train_set = None
        self.test_set = None

        self.dataset_mean = 0.0
        self.dataset_std = 0.0

        self.dataset_name = None
        self.batch_size = 0

        self.dataset_metadata = None

        # self.data_dist = {
        #     "Cifar10": {
        #         "mean": [0.4914, 0.4822, 0.4465],
        #         "std": [0.2470, 0.2435, 0.2616],
        #     },
        #     "Food101": {
        #         "mean": [0.5493, 0.4450, 0.3435],
        #         "std": [0.2729, 0.2758, 0.2798],
        #     },
        # }

    def initialize(self, batch_size: int, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        assert self.dataset_name in ["Cifar10", "Food101"]

        with open('../data/datasets_metadata.json', 'r') as f:
            datasets_metadata = json.load(f)
            self.dataset_metadata = datasets_metadata[self.dataset_name]


        # # Compute dataset statistics
        # # Runs here learning purposes.
        # # In a real implementation this should be computed once, stored and loaded when needed
        # self.compute_dataset_stats()

        self.dataset_mean = self.dataset_metadata["mean"]
        self.dataset_std = self.dataset_metadata["std"]

        if self.dataset_name == "Cifar10":
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.dataset_mean, self.dataset_std)
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.dataset_mean, self.dataset_std)
            ])

            self.train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
            self.test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

        elif self.dataset_name == "Food101":
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(432),
                transforms.RandomCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(self.dataset_mean, self.dataset_std)
            ])

            transform_test = transforms.Compose([
                transforms.Resize(432),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(self.dataset_mean, self.dataset_std)
            ])

            self.train_set = datasets.Food101(root="./data", split="train", download=True, transform=transform_train)
            self.test_set = datasets.Food101(root="./data", split="test", download=True, transform=transform_test)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    @staticmethod
    def compute_dataset_stats(dataset_name: str):
        print("Computing dataset statistics")

        assert dataset_name in ["Cifar10", "Food101"]

        ds = None
        if dataset_name == "Cifar10":
            ds = datasets.CIFAR10(root="./data", train=True, download=True,
                                  transform=transforms.ToTensor())
        elif dataset_name == "Food101":
            ds = datasets.Food101(root="./data", split="train", download=True, transform=transforms.ToTensor())

        assert ds is not None

        loader = DataLoader(ds, batch_size=1, shuffle=False)

        # Pass 1: Compute mean
        mean = 0.
        total = 0
        widths = []
        heights = []
        labels = {}

        for img, label in loader:
            B, C, H, W = img.shape
            img = img.view(B, C, -1)
            mean += img.sum(dim=(0, 2))  # Sum across batch and pixels
            total += B * H * W

            widths.append(W)
            heights.append(H)
            label = int(label.cpu()[0])
            if label not in labels:
                labels[label] = 0
            labels[label] += 1

        shorter_edges = [min(w, h) for w, h in zip(widths, heights)]

        print("Image Dimensions =================================")
        print(f"Min shorter edge: {min(shorter_edges)}")
        print(f"Max shorter edge: {max(shorter_edges)}")
        print(f"Median shorter edge: {sorted(shorter_edges)[len(shorter_edges) // 2]}")
        print(f"Most common: {Counter(shorter_edges).most_common(5)}")

        mean /= total

        # Pass 2: Compute variance
        var = 0.
        for img, _ in loader:
            B, C, H, W = img.shape
            img = img.view(B, C, -1)
            var += ((img - mean.view(1, C, 1)) ** 2).sum(dim=(0, 2))

        std = (var / total).sqrt()

        dataset_mean = mean
        dataset_std = std

        with open('../data/datasets_metadata.json', 'r') as f:
            datasets_metadata = json.load(f)

        print("Data distribution ==========================")
        print(f"{dataset_name} dataset means = {dataset_mean}, std = {dataset_std}")
        print(f"Num labels = {len(labels.keys())}")
        for i, (key, count) in enumerate(labels.items()):
            print(f"{i}: {datasets_metadata[dataset_name]["classes"][key]} --> {count}")

def main():
    DatasetLoader.compute_dataset_stats("Food101")
    # DatasetLoader.compute_dataset_stats("Cifar10")

if __name__ == "__main__":
    main()
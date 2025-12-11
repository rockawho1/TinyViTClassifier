import time
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

from typing import Optional
from DataLoader import DatasetLoader

class Runner:
    def __init__(self):
        self.device = self.get_device()
        self.loss_fn = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.data_loader = None
        self.train_print_every_k = 0
        self.writer = None
        self.in_img_size = 0
        self.profiler = None
        self.return_attn = False
        self.log_per_class_acc = False

    def initialize(self, model: any, config: dict) -> None:
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model.to(self.device)
        self.data_loader = DatasetLoader()
        self.data_loader.initialize(config["training"]["batch_size"], config["data"]["dataset"])

        if "training" in config and "learning_rate" in config["training"]:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["training"]["epochs"])
            self.train_print_every_k = config["training"]["print_every_k"]

        dir_name = config["experiment"]["experiment_name"]
        self.writer = SummaryWriter(f"runs/{dir_name}")
        self.in_img_size = config["model"]["img_size"]

        self.return_attn = config["experiment"]["return_attn"]
        self.profiler = profile(activities=[ProfilerActivity.CPU], record_shapes=True) if config["experiment"]["profile"] is True else nullcontext()
        self.log_per_class_acc = config["experiment"]["log_per_class_acc"]

    @staticmethod
    def get_device():
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        return device

    def evaluate(self, epoch: int, class_names: Optional[list[str]] = None, num_batches_to_viz: int = 1):
        assert self.model is not None

        # Set model to evaluation mode
        self.model.eval()

        correct = 0
        total = 0
        loss_sum = 0.0

        dataloader = self.data_loader.test_loader
        assert dataloader is not None

        attn_maps = None
        images = None

        all_attn_maps = []
        all_preds = []
        all_labels = []
        all_images = []

        # Do not compute gradients
        with torch.no_grad():
            # Get images (x) and labels (y)
            for x, y in dataloader:
                # Move to device
                x, y = x.to(self.device), y.to(self.device)
                # Run inference --> Predict labels
                logits, attn_maps = self.model(x, self.return_attn)
                # Compute loss
                loss = self.loss_fn(logits, y)
                loss_sum += float(loss.item()) * x.size(0)
                # Compute how many correct labels were predicted
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

                if attn_maps is not None and len(all_images) < num_batches_to_viz:
                    all_images.append(x.cpu())
                    all_attn_maps.append(attn_maps[-1].cpu())

                if self.log_per_class_acc is True:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

        # Return the avg. loss and correctness
        val_loss = loss_sum / total
        val_acc = correct / total

        # Per-class accuracy
        if self.log_per_class_acc is True:
            correct_per_class = [0] * len(class_names)
            total_per_class = [0] * len(class_names)

            for pred, label in zip(all_preds, all_labels):
                total_per_class[label] += 1
                if pred == label:
                    correct_per_class[label] += 1

            # Print results
            print("\nPer-class Accuracy:")
            for i, name in enumerate(class_names):
                acc = 100 * correct_per_class[i] / total_per_class[i]
                print(f"{name}: {acc:.2f}% ({correct_per_class[i]}/{total_per_class[i]})")

        # Update tensorboard
        if epoch >= 0:
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)

        if len(all_attn_maps):
            self.visualize(all_attn_maps, all_images, epoch, num_batches_to_viz)
        return val_loss, val_acc

    def train_one_epoch(self, epoch):
        # Set model to training mode
        self.model.train()

        running_loss = 0.0
        correct_tensor = 0
        total = 0
        start = time.time()

        dataloader = self.data_loader.train_loader
        assert dataloader is not None

        # Initialize once before loop
        loss_tensor = torch.tensor(0.0, device=self.device)
        correct_tensor = torch.tensor(0, device=self.device)

        with self.profiler as prof:
            # Get images (x) and labels (y)
            for i, (x, y) in enumerate(dataloader):
                # Move to device
                if prof is not None:
                    prof.step()

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # Run forward pass --> Predict labels
                logits, attn_maps = self.model(x, False)
                # Compute loss
                loss = self.loss_fn(logits, y)
                # Compute gradients
                self.optimizer.zero_grad()
                loss.backward()
                # Update weights
                self.optimizer.step()
                # Update scheduler
                self.scheduler.step()

                # Update loss with total for batch
                # running_loss += float(loss.item()) * x.size(0)
                loss_tensor += (loss * x.size(0)).detach()

                # Compute how many correct labels were predicted
                preds = logits.argmax(dim=1)
                correct_tensor += (preds == y).sum()

                # outputs["total_size"] += x.size(0)
                total += x.size(0)

                if (i + 1) % self.train_print_every_k == 0:
                    print(
                        f"Epoch {epoch} | Step {i + 1}/{len(dataloader)}"
                    )

                if prof is not None and i > 20:
                    break

        dur = time.time() - start

        if prof is not None:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # train_loss = running_loss / total
        train_loss = loss_tensor.item() / total
        train_acc = correct_tensor.item() / total

        # Update tensorboard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)

        return train_loss, train_acc, dur

    def visualize(self, attn_maps: list[torch.Tensor], images: list[torch.Tensor], epoch: int, num_batches: int) -> None:
        if attn_maps is None or len(attn_maps) == 0:
            return

        num_to_show = min(num_batches, len(attn_maps))
        for bi in range(num_to_show):
            num_imgs = attn_maps[bi].shape[0]

            # Last layer (-1), first image (0), average heads (mean dim=0)
            # avg_attn = attn_maps[-1][:0].mean(dim=0)
            avg_attn = attn_maps[bi].mean(dim=1)
            # CLS token attention to patches
            # cls_attn = avg_attn[0, 1:]
            cls_attn = avg_attn[:, 0, 1:]
            # Reshape to patch size
            # dim = int(cls_attn.shape[0] ** 0.5)
            dim = int(cls_attn.shape[1] ** 0.5)
            # cls_attn = cls_attn.reshape(num_imgs, dim, dim)
            cls_attn = cls_attn.reshape(num_imgs, dim, dim)

            print(f"cls_attn shape = {cls_attn.shape}")

            # Normalize to 0-1 range
            for i in range(num_imgs):
                img_attn = cls_attn[i]
                cls_attn[i] = (img_attn - img_attn.min()) / (img_attn.max() - img_attn.min() + 1e-8)

            # # Tensorboard expect (C, H, W) format
            # cls_attn = cls_attn.unsqueeze(0)

            cls_attn_upsampled = F.interpolate(
                cls_attn.unsqueeze(1),
                size=(self.in_img_size, self.in_img_size),
                mode='bilinear',
                align_corners=False
            )

            cls_attn_composite = images[bi] * cls_attn_upsampled.cpu()

            if num_to_show == 1:
                self.writer.add_images('Attention/cls_token', cls_attn.unsqueeze(1), epoch)
                self.writer.add_images('Attention/upsampled', cls_attn_upsampled, epoch)
                self.writer.add_images('Attention/composite', cls_attn_composite, epoch)
                self.writer.add_images('images/sample', images[-bi], epoch)
            else:
                self.writer.add_images(f'Attention/cls_token_{bi}', cls_attn.unsqueeze(1), epoch)
                self.writer.add_images(f'Attention/upsampled_{bi}', cls_attn_upsampled, epoch)
                self.writer.add_images(f'Attention/composite_{bi}', cls_attn_composite, epoch)
                self.writer.add_images(f'images/sample_{bi}', images[bi], epoch)

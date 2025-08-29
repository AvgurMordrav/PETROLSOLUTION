import petroscope.segmentation as segm
from petroscope.segmentation.utils import load_image, load_mask
from petroscope.segmentation.classes import ClassSet, LumenStoneClasses
from petroscope.segmentation.eval import SegmDetailedTester

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import amp

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os
import cv2

from pathlib import Path

  


class Unetpp(segm.GeoSegmModel):
   
    class CombinedLoss(nn.Module):
        def __init__(self, weight_ce=0.5, weight_dice=0.5):
            super().__init__()
            self.ce = nn.CrossEntropyLoss()
            self.weight_ce = weight_ce
            self.weight_dice = weight_dice

        def forward(self, inputs, targets):
            ce_loss = self.ce(inputs, targets)
            dice_loss = self.dice_loss(inputs, targets)
            return self.weight_ce * ce_loss + self.weight_dice * dice_loss

        def dice_loss(self, inputs, targets, smooth=1e-5):
            
            probs = torch.softmax(inputs, dim=1)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=probs.shape[1]
            ).permute(0, 3, 1, 2).float()

            dims = (0, 2, 3)
            intersection = torch.sum(probs * targets_onehot, dims)
            union = torch.sum(probs, dims) + torch.sum(targets_onehot, dims)
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1 - dice.mean()

   
    class SegmentationDataset(Dataset):
        def __init__(self, img_dir, mask_dir, tile_size=256, overlap=64, transforms=None, classes: ClassSet = None):
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            self.tile_size = tile_size
            self.overlap = overlap
            self.transforms = transforms
            self.classes = classes

            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('png', 'jpg', 'jpeg', 'tif'))])
            self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg', 'tif'))])

            assert len(self.img_files) == len(self.mask_files), "Images and masks count mismatch!"

            self.tiles = []
            for img_file, mask_file in zip(self.img_files, self.mask_files):
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                img = load_image(img_path, normalize=True)
                h, w = img.shape[:2]
                step = self.tile_size - self.overlap
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        x1, y1 = x, y
                        x2, y2 = x + self.tile_size, y + self.tile_size
                        pad_x = max(0, x2 - w)
                        pad_y = max(0, y2 - h)
                        self.tiles.append({
                            "img_path": img_path, "mask_path": mask_path,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "pad_x": pad_x, "pad_y": pad_y
                        })

        def __len__(self):
            return len(self.tiles)

        def __getitem__(self, idx):
            tile = self.tiles[idx]
            img = load_image(tile["img_path"], normalize=True)
            mask = load_mask(tile["mask_path"], classes=self.classes, one_hot=False)
            patch_img = img[tile["y1"]:tile["y2"], tile["x1"]:tile["x2"]]
            patch_mask = mask[tile["y1"]:tile["y2"], tile["x1"]:tile["x2"]]

            if tile["pad_x"] > 0 or tile["pad_y"] > 0:
                patch_img = cv2.copyMakeBorder(patch_img, 0, tile["pad_y"], 0, tile["pad_x"], cv2.BORDER_REFLECT)
                patch_mask = cv2.copyMakeBorder(patch_mask, 0, tile["pad_y"], 0, tile["pad_x"], cv2.BORDER_REFLECT)

            if self.transforms:
                augmented = self.transforms(image=patch_img, mask=patch_mask)
                patch_img, patch_mask = augmented["image"], augmented["mask"]

            patch_img = np.transpose(patch_img, (2, 0, 1)).astype("float32")
            patch_mask = patch_mask.astype("int64")

            return patch_img, patch_mask

   
    def __init__(self, classes: ClassSet, weight_ce=0.75, weight_dice=0.25) -> None:
        super().__init__()
        self.classes = classes
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

       
        self.CrE = self.CombinedLoss(weight_ce=weight_ce, weight_dice=weight_dice)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.epochs = 120
        self.batch_size = 8
        self.num_workers = 4

        os.makedirs("epoch_show", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    
    def train(self, save_model=False) -> None:
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.07, 0.07), rotate=(-25, 25), p=0.7)
        ])

        train_dataset = self.SegmentationDataset(
            img_dir="imgs/train",
            mask_dir="masks/train",
            tile_size=256,
            overlap=64,
            transforms=train_transforms,
            classes=self.classes
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        scaler = amp.GradScaler()

        print(f"[INFO] Training dataset: {len(train_dataset)} tiles")

        for i in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_idx = 0
            for imgs, masks in train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device, dtype=torch.long)
                self.optimizer.zero_grad()
                with amp.autocast(device_type="cuda"):
                    outputs = self.model(imgs)
                    loss = self.CrE(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                epoch_loss += loss.item()
                batch_idx += 1
                print(f"\rEpoch {i+1}/{self.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {epoch_loss/batch_idx:.4f}", end='', flush=True)

            print(f"[INFO] Epoch {i+1}, Avg Loss: {epoch_loss/len(train_loader):.4f}")
            self.predict_image(np.array(load_image("imgs/test/test_01.jpg")), show=True, epoch=i, loss=epoch_loss/len(train_loader))
            if save_model:
                self.save(i+1, epoch_loss/len(train_loader))

   
    def predict_image(self, image: np.ndarray, tile_size=512, overlap=64, show=True, epoch=0, loss=0) -> np.ndarray:
        h, w = image.shape[:2]
        self.model.eval()
        stride = tile_size - overlap
        mask_pred = np.zeros((h, w), dtype=np.uint8)
        with torch.no_grad():
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    tile = image[y:y+tile_size, x:x+tile_size]
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        pad_h = tile_size - tile.shape[0]
                        pad_w = tile_size - tile.shape[1]
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    output = self.model(tile_tensor)
                    pred = torch.argmax(output, dim=1)
                    pred = pred.squeeze(0).cpu().numpy()
                    mask_pred[y:y+tile_size, x:x+tile_size] = pred[:min(tile_size, h-y), :min(tile_size, w-x)]

        if show:
            mask_color = cv2.applyColorMap((mask_pred * 40).astype(np.uint8), cv2.COLORMAP_JET)
            save_path = f"mask_epoch_{epoch}_{loss:.4f}.png"
            cv2.imwrite(save_path, mask_color)

        return mask_pred

   
    def save(self, iter, loss) -> None:
        save_path = os.path.join("moredice", f'unetpp_model_epoch:{iter}_loss:{loss:.4f}.pth')
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)

    
    def load(self, path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only = False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])




if __name__ == "__main__":
    classet = LumenStoneClasses.S1v1()
    model = Unetpp(classes=classet)
    model_dirs = [
    "CrosEntropy_WithRegularAugm",
    "epoch_show_CpD",
    "epoch_show_CpD_DopAug",
    "moredice"
    ]
    test_img_mask_p = [
        (img_p, Path("masks/test") / f"{img_p.stem}.png")
        for img_p in sorted((Path("imgs/test")).iterdir())
    ]
    
    path = "unetpp_model_epoch:42_loss:0.2771.pth"
    print(f"[INFO] Загружаю {path}")
    model.load(path)
    tester = SegmDetailedTester(
        Path("output"),
        classes=classet,
        void_pad=0,
        void_border_width=4,
        vis_plots=False,
        vis_segmentation=True,
    )
    res, res_void = tester.test_on_set(
        test_img_mask_p,
        lambda img: model.predict_image(img, show=False),
        description="test",
        return_void=True,
    )
    print(f"Metrics:\n{res}")
    print(f"Metrics with void borders:\n{res_void}")
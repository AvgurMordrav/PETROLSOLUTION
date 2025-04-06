import petroscope.segmentation as segm
from petroscope.segmentation.utils import load_image, load_mask

from typing import Iterable
import numpy as np
from tqdm import tqdm
from petroscope.segmentation.classes import ClassSet, LumenStoneClasses
import cv2
from sklearn.cluster import MiniBatchKMeans
from petroscope.segmentation.eval import SegmDetailedTester
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

class ColorClusterMockModel(segm.GeoSegmModel):
    @dataclass
    class PetrolDataset(Dataset):
        def __init__(self, Paths: Iterable[tuple[Path, Path]], classes: ClassSet, len, patch_size = 256):
            self.paths = Paths
            self.classes = classes
            self.len = len
            self.patch_size = patch_size

        def __len__(self):
            return self.len * 1000
        def __getitem__(self, idx):
            im_idx = idx // 1000
            im_p , mask_p = self.paths[im_idx]
            image = load_image(im_p, normalize = True)
            mask = load_mask(mask_p, classes = self.classes, one_hot = False)

            h, w = image.shape[:2]
            x = np.random.randint(0, h - self.patch_size)
            y = np.random.randint(0, w - self.patch_size)

            image_patch = image[x:x+self.patch_size, y:y+self.patch_size] / 255.0
            mask_patch = mask[x:x+self.patch_size, y:y+self.patch_size]

            image_tensor = torch.from_numpy(image_patch).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_patch).long()
            return image_tensor, mask_tensor
    
    class PetCNN(nn.Module):
        def __init__(self):
            super(ColorClusterMockModel.PetCNN, self).__init__()

            self.conv1 = self.my_conv(3, 64)
            self.conv2 = self.my_conv(64, 128)
            self.conv3 = self.my_conv(128, 256)
            self.conv4 = self.my_conv(256, 512)
            self.conv5 = self.my_conv(512, 1024)

            self.pool = nn.MaxPool2d(2,2)

            self.back4 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
            self.dec4 = self.my_conv(1024, 512)

            self.back3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
            self.dec3 = self.my_conv(512, 256)

            self.back2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
            self.dec2 = self.my_conv(256, 128)

            self.back1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
            self.dec1 = self.my_conv(128, 64)

            self.last = nn.Conv2d(64, 7, kernel_size = 1)

        def my_conv(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):

            enc1 = self.conv1(x)
            enc2 = self.pool(self.conv2(enc1))
            enc3 = self.pool(self.conv3(enc2))
            enc4 = self.pool(self.conv4(enc3))
            enc5 = self.pool(self.conv5(enc4))

            dec4 = self.back4(enc5)
            dec4 = torch.cat((dec4, enc4), dim = 1)
            dec4 = self.dec4(dec4)


            dec3 = self.back3(dec4)
            dec3 = torch.cat((dec3, enc3), dim = 1)
            dec3 = self.dec3(dec3)

            dec2 = self.back2(dec3)
            dec2 = torch.cat((dec2, enc2), dim = 1)
            dec2 = self.dec2(dec2)

            dec1 = self.back1(dec2)
            dec1 = torch.cat((dec1, enc1), dim = 1)
            dec1 = self.dec1(dec1)

            return self.last(dec1)



    def __init__(self, classes: ClassSet) -> None:
        super().__init__()
        self.classes = classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model = ColorClusterMockModel.PetCNN().to(self.device)
        self.crieterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

    def train_model(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        epochs=40,
    ):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            checkpoint = {f'stat_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            self.save_checkpoint(checkpoint, epoch + 27)
            for images, labels in tqdm(train_loader, f"training, epoch {epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")   
        




    def load(self, saved_path) -> None:
        print("=> Loading checkpoint")
        self.model.load_state_dict(saved_path['stat_dict'])
        self.optimizer.load_state_dict(saved_path['optimizer'])

    def train(
        self, img_mask_paths: Iterable[tuple[Path, Path]], **kwargs
    ) -> None:
        dataset_train = ColorClusterMockModel.PetrolDataset(img_mask_paths, self.classes, 59)
        dataloader_train = DataLoader(dataset_train, batch_size = 10, shuffle = True, num_workers = 4)
        self.train_model(
            self.model,
            dataloader_train,
            self.crieterion,
            self.optimizer,
        )

        

    def predict_image(self, image: np.ndarray, patch_size = 256) -> np.ndarray:
        self.model.eval()
        h, w = image.shape[:2]
        pad_h = (patch_size - h%patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode = 'reflect')
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.int64)
        with torch.no_grad():
            for i in range(0, image.shape[0], patch_size):
                for j in range(0, image.shape[1], patch_size):
                    patch = image[i:i + patch_size, j:j + patch_size]
                    patch_tensor = torch.FloatTensor(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)/ 255.0

                    output = self.model(patch_tensor)
                    pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                    full_mask[i:i + patch_size, j:j + patch_size] = pred_mask

        return full_mask[:h, :w]
        


        

        
        


classset = LumenStoneClasses.S1v1()
train_img_mask_p = [
    (img_p, Path("masks/train") / f"{img_p.stem}.png")
    for img_p in sorted((Path("imgs/train")).iterdir())
]

test_img_mask_p = [
    (img_p, Path("masks/test") / f"{img_p.stem}.png")
    for img_p in sorted((Path("imgs/test")).iterdir())
]

model = ColorClusterMockModel(classes=classset)


print("my_checkpoint27.pth.tar")
model.load(torch.load("my_checkpoint27.pth.tar"))

tester = SegmDetailedTester(
    Path("output"),
    classes=classset,
    void_pad=0,
    void_border_width=4,
    vis_plots=False,
    vis_segmentation=True,
)

res, res_void = tester.test_on_set(
    test_img_mask_p,
    lambda img: model.predict_image(img),
    description="test",
    return_void=True,
)

print(f"Metrics:\n{res}")
print(f"Metrics with void borders:\n{res_void}")
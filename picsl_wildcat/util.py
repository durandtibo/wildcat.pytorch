import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

class ImageFolderWithBBox(datasets.ImageFolder):
    """This dataset object loads images along with bounding boxes and converts
    those bounding boxes into masks in the alpha channel
    
    Args:
        data_path: path to data
        manifest: pandas DataFrame conaining columns 'id', 'w' and 'h'
        transform: transform chain capable of acting on 4D images
        min_box_size: smallest mask size allowed
    """

    # Pass the manifest dataframe as initializer
    def __init__(self, data_path, manifest = None, transform = None, min_box_size = 112):
        
        # Wrapped worker
        self._worker = datasets.ImageFolder(data_path)
        
        # Set the transform for the image 
        # t_img = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #])
        
        # Initialize parent with default image transform
        # super(ImageFolderWithBBox, self).__init__(data_path, transform=t_img)
        
        # Store the manifest and the custom transform
        self.manifest = manifest
        self.transform = transform
        self.classes = self._worker.classes
        self.class_to_idx = self._worker.class_to_idx
        self.targets = self._worker.targets
        self.samples = self._worker.samples
        self.min_box_size = min_box_size

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        
        # Get the image and label from the worker
        (img, label) = self._worker.__getitem__(index)
        
        # the image file path and patch id
        path = self._worker.imgs[index][0]
        patch_id = os.path.splitext(os.path.basename(path))[0]
        
        # Look up the patch
        if self.manifest is not None:
            match = self.manifest.loc[self.manifest['id']==int(patch_id), ('w','h')]
            b = np.array([match.iloc[0,0], match.iloc[0,1]])

            # Set the minimum box size to 56x56
            b = max(b[0],self.min_box_size), max(b[1],self.min_box_size)

            # Generate the mask
            half_range = lambda k : np.arange(-(k//2),k-(k//2))
            gx,gy = np.meshgrid(half_range(img.size[0]), half_range(img.size[1]))
            mask = ((gx > -0.5*b[0]) * (gx < 0.5*b[0]) * (gy > -0.5*b[1]) * (gy < 0.5*b[1])).astype(int)
        else:
            mask = np.zeros((img.size[0], img.size[1])).astype(int)
        
        # Compose the mask with the image
        img = torch.cat((transforms.functional.to_tensor(img), torch.tensor(mask).unsqueeze(0)))
        
        # Apply the transform
        if self.transform:
            img = self.transform(img)
        
        # make a new tuple that includes original and the path
        return img, label
    
    # Get the number of samples
    def __len__(self):
        return len(self._worker)


# Color jitter that works on the RGB channels only
class ColorJitterRGB(torch.nn.Module):
    """Color Jitter transform that only applies to the RGB channels of an RGBA image"""
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self._wrapped = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def forward(self, image):
        return torch.cat((self._wrapped(image[0:3,:,:]), image[3:,:,:]))

    
# Normalization that works on the RGB channels only
class NormalizeRGB(torch.nn.Module):
    """Normalize transform that only applies to the RGB channels of an RGBA image"""
    
    def __init__(self, mean, std):
        super().__init__()
        self._wrapped = transforms.Normalize(mean, std)
    
    def forward(self, image):
        return torch.cat((self._wrapped(image[0:3,:,:]), image[3:,:,:]))  


# Randomly samples crops a patch of specified size from inside of the larger input patch
class RandomCrop(torch.nn.Module):
    """Crop the patch randomly from inside of the larger parent patch, such that the center
    of the larger patch is somewhere inside of the sampled patch"""

    def __init__(self, patch_size):
        super().__init__()
        self.ps = patch_size

    def forward(self, image):
        sx,sy = image.shape[-2:]
        cx,cy = (sx+1)//2, (sy+1)//2 

        # Possible range of the box
        x0,y0 = max(0, cx-self.ps), max(0, cy-self.ps)
        x1,y1 = min(sx - self.ps, cx), min(sy - self.ps, cy)
        x,y = np.random.randint(x0, x1+1), np.random.randint(y0, y1+1)

        # Perform the crop
        return image[:,x:x+self.ps,y:y+self.ps]


# Plot a batch of images with masks in the alpha channel
# Plot a grid of images with bounding boxes
def show_patches_with_masks(img, labels=None, class_names=None, w=4, alpha=0.2, max_rows=0):
    """
    Plot a mini-batch of patches with masks overlaid. 
    Args:
        img: a [B,4,W,H] tensor of images, mask stored in the 4th channel
        labels: optional [B] tensor of labels
        class_names: optional list of class names corresponding to labels
        w: optional number of patches per row
        alpha: optional transparency of the overlaid masks
        max_rows: optional limit to only so many roes 
    """
    n = img.shape[0] if max_rows == 0 else min(max_rows * w, img.shape[0])
    for row in range(0, n, w):
        for i in range(row, min(row+w, n)):
            ax = plt.subplot(1+n//w, w, i+1)
            inp = np.transpose(img.numpy()[i,:],(1,2,0))
            ax.imshow((inp[:,:,0:3] + 2.2) / 5)
            ax.imshow(inp[:,:,3], alpha = alpha, vmin=0, vmax=1)
            # ax.contour(inp[:,:,3], levels=(.5,))
            if labels is not None and class_names is not None:
                plt.title(class_names[labels[i].item()])
            ax.set_axis_off()
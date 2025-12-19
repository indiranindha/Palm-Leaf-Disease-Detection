# import os
# from PIL import Image
# from torch.utils.data import Dataset

# class PalmLeafDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         """
#         Args:
#             root_dir (str): path to train/val/test directory
#             transform (callable, optional): image transformations
#         """

#         self.root_dir = root_dir
#         self.transform = transform

#         self.image_paths = []
#         self.labels = []

#         self.class_names = sorted(os.listdir(root_dir))
#         self.class_to_idx = {
#             class_name: idx for idx, class_name in enumerate(self.class_names)
#         }

#         for class_name in self.class_names:
#             class_dir = os.path.join(root_dir, class_name)

#             if not os.path.isdir(class_dir):
#                 continue

#             for img_name in os.listdir(class_dir):
#                 if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
#                     self.image_paths.append(
#                     os.path.join(class_dir, img_name)
#                     )
#                     self.labels.append(self.class_to_idx[class_name])

#     def __len__(self):
#         return len(self.image_paths)
        
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         label = self.labels[idx]

#         image = Image.open(image_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# SAMA CUMAN LEBIH SIMPLE

import os
from PIL import Image
from torch.utils.data import Dataset

class PalmLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        self.samples = []
        for cls_name in self.class_names:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                self.samples.append(
                    (os.path.join(cls_path, fname), self.class_to_idx[cls_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

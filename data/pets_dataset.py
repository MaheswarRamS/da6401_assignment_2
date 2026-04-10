"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    Class_Names = [ 'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
        'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
        'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
        'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
        'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
        'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
        'wheaten_terrier', 'yorkshire_terrier'
    ]

    def __init__(self, root_dir: str = "./data/pets", split: str = 'train', img_size: int=224) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.images: List[str] = []
        self.labels: List[int] = []

        list_file = os.path.join(root_dir, 'annotations','trainval.txt' if split =='train' else 'test.txt')

        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >=2:
                        self.images.append(parts[0])
                        self.labels.append(int(parts[1]) - 1)

        if split=='train':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5 if split == 'train' else 0),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=20, p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(transpose_mask=False)
            ],additional_targets={'mask': 'mask'},bbox_params=A.BboxParams(format='pascal_voc', label_fields =['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(transpose_mask=False)
            ],additional_targets={'mask': 'mask'},bbox_params=A.BboxParams(format='pascal_voc', label_fields =['class_labels']))


    def __len__(self) -> int:
        return len(self.images)
    
    
    @staticmethod
    def _load_bbox(xml_path:str) -> list:
        """ loading the xml files and returning (cx,cy,w,h) scaled to target size """
        try:  
            root = ET.parse(xml_path).getroot()
            bndbox = root.find('.//bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
    
            return torch.tensor([xmin,ymin,xmax,ymax], dtype = torch.float32)  
        except Exception:
            return None
    
    @staticmethod
    def bbox_from_mask(mask: np.ndarray, orig_w: int, orig_h:int) -> list:
        """ Derive the bbox from the mask for the test images since it is not available"""
        pet = (mask==0)
        if pet.sum() ==0:
            return [0.0,0.0, float(orig_w), float(orig_h)]        
        rows =np.where(np.any(pet,axis=1))[0]
        cols =np.where(np.any(pet,axis=0))[0]
        ymin, ymax = rows[0], rows[-1]
        xmin, xmax = cols[0], cols[-1]
        return [float(xmin),float(ymin),float(xmax),float(ymax)]
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.images[idx]
        img_path = os.path.join(
            self.root_dir, 'images', f'{img_name}.jpg'
        )
        img = np.array(Image.open(img_path).convert('RGB'))
        orig_h , orig_w = img.shape[:2]

        mask_path = os.path.join(self.root_dir, 'annotations/trimaps', f'{img_name}.png')

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            mask = mask-1
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype = np.uint8)
        
        xml_path = os.path.join(self.root_dir, 'annotations', 'xmls', f'{img_name}.xml')
        bbox = self._load_bbox(xml_path)
        if bbox is None:
            bbox =self.bbox_from_mask(mask,orig_w,orig_h)
        label = self.labels[idx] 
        transformed = self.transform(image=img, mask=mask,bboxes = [bbox],class_labels=[label]) 
        img = transformed['image']
        mask = transformed['mask']
        t_bbox =transformed['bboxes'][0]
                                      
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        cx = (t_bbox[0] + t_bbox[2]) / 2.0
        cy = (t_bbox[1] + t_bbox[3]) / 2.0    
        w = t_bbox[2] - t_bbox[0]
        h = t_bbox[3] - t_bbox[1]

        final_bbox = torch.tensor([cx,cy,w,h], dtype= torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': img,
            'label': label,
            'bbox': final_bbox,
            'mask': mask,
            'image_name': img_name
        }


import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, jaccard_score

import wandb
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from models.multitask import MultiTaskPerceptionModel

# Metrics calculation function
def cls_metrics(y_true, y_pred):
    """ Classification metrics"""
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'f1 Score': f1_score(y_true,y_pred, average='macro', zero_division=0) * 100,
        'Precision': precision_score(y_true,y_pred,average='macro', zero_division=0) * 100,
        'Recall' : recall_score(y_true,y_pred,average='macro', zero_division=0) * 100,
    
    }

def loc_metrics(pred_boxes, target_boxes):
    """Localization Metrics:
       Mean IOU for bounding boxes"""
   
    iou_fn = IoULoss(reduction='none')
    mean_iou = (1.0 - iou_fn(pred_boxes, target_boxes)).mean().item() * 100
    return {'Mean_IOU':mean_iou}   

def seg_metrics(y_true, y_pred, num_classes):
    """ Segementation Metrics:
        px_acc: Pixel accuracy
        miou: Mean IOU"""
    
    return {
        'px_acc': accuracy_score(y_true,y_pred) * 100,
        'miou' : jaccard_score(y_true,y_pred, average='macro', zero_division=0) * 100,
    }


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    wandb.init(project=args.wandb_project, entity= args.wandb_entity, name= 'inference', config=vars(args))

    model = MultiTaskPerceptionModel(num_breeds=args.num_breeds, seg_classes=args.seg_classes, classifier_path=args.cls_ckpt, localizer_path=args.loc_ckpt, unet_path=args.seg_ckpt,).to(device)

    model.eval ()
    test_ds = OxfordIIITPetDataset(args.data_dir, split='test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f'Test Sample: {len(test_ds)}')

    all_cls_preds, all_cls_labels = [],[]
    all_box_preds, all_box_targets = [],[]
    all_seg_preds, all_seg_labels = [],[]
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = imgs.to(device)
            labels = batch['label'].to(device)
            boxes = batch['bbox'].to(device)
            masks = batch['mask'].to(device)

            out = model(imgs)

            # Classification
            all_cls_preds.extend(out['classification'].argmax(1).cpu().tolist())
            all_cls_labels.extend(labels.cpu().tolist())
           
            # Localization
            all_box_preds.append(out['localization'].cpu())
            all_box_targets.append(boxes.cpu())
            
            # Segmentation
            all_seg_preds.extend(out['segmentation'].argmax(1).cpu().tolist())
            all_seg_labels.extend(masks.cpu().tolist())

    # Metrics Calculation
    cm = cls_metrics(all_cls_labels,all_cls_preds)
    
    box_preds = torch.cat(all_box_preds, dim=0)
    box_targets = torch.cat(all_box_targets, dim=0)
    lm = loc_metrics(box_preds, box_targets)

    sm = seg_metrics(all_seg_labels,all_seg_preds, args.seg_classes)

    results = {
            'test/cls_accuracy': cm['accuracy'],
            'test/cls_f1': cm['f1'],
            'test/cls_precision': cm['precision'],
            'test/cls_recall': cm['recall'],
            'test/loc_mean_iou': lm['Mean_IOU'],
            'test/seg_px_acc': sm['px_acc'],
            'test/seg_miou': sm['miou'],
    }


    print('\n------Test Results-------')
    for k,v in results.items():
        print(f' {k}: {v:.2f}%')

    
    wandb.log(results)
    wandb.finish()


def parse_args():
    p =argparse.ArgumentParser(description='DA6401_Assignment_2_train')
    p.add_argument('-d', '--data_dir', type=str, default= './data/pets')
    p.add_argument('-bs', '--batch_size', type=int, default=16)
    p.add_argument('-nm', '--num_workers', type=int, default=0)
    p.add_argument('-nb', '--num_breeds', type=int, default=37)
    p.add_argument('-sc', '--seg_classes', type=int, default=3)
    p.add_argument('-cck', '--cls_ckpt', type=str, default='checkpoints/classifier.pth')
    p.add_argument('-lck', '--loc_ckpt', type=str, default='checkpoints/localizer.pth')
    p.add_argument('-sck', '--seg_ckpt', type=str, default='checkpoints/segmentation.pth')
    p.add_argument('-wp', '--wandb_project', type=str, default='DA6401_Assignment_2')
    p.add_argument('-we', '--wandb_entity', type=str, default=None)
    return p.parse_args()

if __name__ == '__main__':
    evaluate(parse_args())
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, jaccard_score

import wandb
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
 

class DiceLoss(nn.Module):
    'Dice loss for segmentation'

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
        pred: [B,C,H,W] raw logits
        target: [B,H,W]
        """
        B,C,H,W = pred.shape
        target_onehot = torch.zeros(B,C,H,W, dtype= pred.dtype, device= pred.device)
        target_onehot.scatter_(1, target.unsqueeze(1),1)
        pred_soft = torch.softmax(pred, dim=1).view(B,C,-1)
        target_onehot = target_onehot.view(B,C,-1)
        intersection = (pred_soft * target_onehot).sum(dim=2)
        dice = (2.0* intersection) / (pred_soft.sum(dim=2) + target_onehot.sum(dim=2) + self.eps)
        return (1.0-dice).mean()
    

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_loaders(data_dir, batch_size, val_fraction=0.1, num_workers = 0):
    trainval_ds = OxfordIIITPetDataset(data_dir, split ='train')
    test_ds = OxfordIIITPetDataset(data_dir, split='test')

    n_val = max(1, int(len(trainval_ds) * val_fraction))
    n_train = len(trainval_ds) - n_val

    train_ds , val_ds = random_split(trainval_ds, [n_train,n_val], generator = torch.Generator().manual_seed(42),)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory =True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )

def save_chekpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Saved{path}')


def cls_metrics(y_true, y_pred):
    """ Classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'f1 Score': f1_score(y_true,y_pred, average='macro', zero_division=0) * 100,
        'precision': precision_score(y_true,y_pred,average='macro', zero_division=0) * 100,
        'recall' : recall_score(y_true,y_pred,average='macro', zero_division=0) * 100,
    
    }

def loc_metrics(preds, targets):
    """Localization metrics"""
    iou_per_sample = 1.0 - IoULoss(reduction='none')(preds.cpu(), targets.cpu())  
    return {
        'mean_iou': iou_per_sample.mean().item() * 100,
        'accuracy':   (iou_per_sample >= 0.5).float().mean().item() * 100,
        'mae':      (preds.cpu() - targets.cpu()).abs().mean().item(),
    }

def seg_metrics(pred_mask, true_mask, num_classes):
    """ Segementation Metrics"""
    pred = pred_mask.cpu().numpy().flatten()
    true = true_mask.cpu().numpy().flatten()
    return {
        'px_acc': accuracy_score(true,pred) * 100,
        'miou' : jaccard_score(true,pred, average='macro', zero_division=0) * 100,
    }

# Classification Train loop
def train_classifier(args, train_loader, val_loader, device):
    print('\n' + '-'*60)
    print('Task 1: Classification')
    print('\n' + '-'*60)

    wandb.init(project=args.wandb_project, entity= args.wandb_entity, name= 'classification', config=vars(args), reinit=True)

    model = VGG11Classifier(num_classes=args.num_breeds, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 3)

    best_val_loss = float('inf')

    for epoch in range(1 , args.epochs +1):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_preds.extend(logits.argmax(1).cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
           
            if batch_idx % 10 == 0:
                print(f"  [cls] epoch {epoch} batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}")
        
        train_loss /= len(train_loader.dataset)
        tm = cls_metrics(train_labels, train_preds)


        model.eval()
        val_loss =0.0 
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
        
         
        val_loss /= len(val_loader.dataset)
        vm = cls_metrics(val_labels, val_preds)
        scheduler.step(val_loss)

        print(f' Epoch {epoch:3d}/{args.epochs} |'
              f" Train_loss={train_loss:.4f} Train_accuracy={tm['accuracy']:.1f}% Train_f1={tm['f1 Score']:.1f}% |"
              f" Val_loss={val_loss:.4f} Val_accuracy={vm['accuracy']:.1f}% Val_f1={vm['f1 Score']:.1f}% |")
        
        wandb.log({
            'cls/train_loss': train_loss,
            'cls/train_accuracy': tm['accuracy'],
            'cls/train_f1': tm['f1 Score'],
            'cls/train_precision': tm['precision'],
            'cls/train_recall': tm['recall'],
            'cls/val_loss': val_loss,
            'cls/val_accuracy': vm['accuracy'],
            'cls/val_f1': vm['f1 Score'],
            'cls/val_precision': vm['precision'],
            'cls/val_recall': vm['recall'],
            'cls/lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_chekpoint(model, args.cls_ckpt)

    wandb.finish()

# Localization Train loop
def train_localizer(args, train_loader, val_loader, device):
    print('\n' + '-'*60)
    print('Task 2: Localization')
    print('\n' + '-'*60)

    wandb.init(project=args.wandb_project, entity= args.wandb_entity, name= 'localization', config=vars(args), reinit=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)
    criterion = nn.MSELoss()
    iou_fn = IoULoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 3)

    best_val_loss = float('inf')

    for epoch in range(1 , args.epochs +1):
        model.train()
        t_loss, t_mse, t_iou, n = 0.0,0.0,0.0,0.0
        t_miou, t_acc, t_mae = 0.0,0.0,0.0 
        for batch_idx, batch in enumerate (train_loader,1):
            imgs = batch['image'].to(device)
            boxes    = batch['bbox'].to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            l_mse = criterion(preds, boxes)
            l_iou = iou_fn(preds, boxes)
            loss = l_mse + l_iou
            loss.backward()
            optimizer.step()
            tm = loc_metrics(preds.detach(), boxes)
            t_loss += loss.item() 
            t_mse += l_mse.item()  
            t_iou += l_iou.item() 
            t_miou += tm['mean_iou']
            t_acc += tm['accuracy']
            t_mae += tm['mae']
            n += 1

            if batch_idx % 10 == 0:
                print(f"  [loc] epoch {epoch} batch {batch_idx}/{len(train_loader)} | "
                      f"loss={loss.item():.4f} mse={l_mse.item():.4f} iou={l_iou.item():.4f} Accuracy = {tm['accuracy']:.1f}%  mae={tm['mae']:.1f}%")
                

        
        t_loss /= n; t_mse /= n; t_iou /= n;
        t_miou /= n; t_acc /= n; t_mae /= n

        model.eval()
        v_loss, v_mse, v_iou, nv = 0.0,0.0,0.0,0.0
        v_miou, v_acc, v_mae = 0.0,0.0,0.0

        with torch.no_grad():
             for batch in val_loader:
                imgs = batch['image'].to(device)
                boxes    = batch['bbox'].to(device)
                preds = model(imgs)
                l_mse = criterion(preds, boxes)
                l_iou = iou_fn(preds, boxes)
                loss = l_mse + l_iou
                vm = loc_metrics(preds, boxes)
                v_loss += loss.item() 
                v_mse += l_mse.item()  
                v_iou += l_iou.item() 
                v_miou += tm['mean_iou']
                v_acc += tm['accuracy']
                v_mae += tm['mae'] 
                nv += 1
        
        v_loss /= n; v_mse /= n; v_iou /= n;
        v_miou /= nv; v_acc /= nv; v_mae /= nv       
        scheduler.step(v_loss)

        print(f' Epoch {epoch:3d}/{args.epochs} |'
              f" Train_loss={t_loss:.4f} mse={t_mse:.4f} iou={t_iou:.4f} Train_miou={t_miou:.1f} Train_acc={t_acc:.1f}%  Train_MAE={t_mae:.1f}% |"
              f" Val_loss={v_loss:.4f} mse={v_mse:.4f} iou={v_iou:.4f} Val_miou={v_miou:.1f} Val_acc={v_acc:.1f}%  Val_MAE={v_mae:.1f}% |")
        
        wandb.log({
            'loc/train_loss': t_loss,
            'loc/train_mse': t_mse,
            'loc/train_iou': t_iou,
            'loc/train_miou': t_miou,
            'loc/train_acc': t_acc,
            'loc/train_mae': t_mae,
            'loc/val_loss': v_loss,
            'loc/val_mse': v_mse,
            'loc/val_iou': v_iou,
            'loc/val_miou': v_miou,
            'loc/val_acc': v_acc,
            'loc/val_mae': v_mae,
            'loc/lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_chekpoint(model, args.loc_ckpt)

    wandb.finish

# Segmentation Train loop
def train_segmentation(args, train_loader, val_loader, device):
    print('\n' + '-'*60)
    print('Task 3: Segmentation')
    print('\n' + '-'*60)

    wandb.init(project=args.wandb_project, entity= args.wandb_entity, name= 'segmentation', config=vars(args), reinit=True)

    model = VGG11UNet(num_classes=args.seg_classes, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 3)

    best_val_loss = float('inf')

    for epoch in range(1 , args.epochs +1):
        model.train()
        t_loss, t_px_acc, t_miou, nt = 0.0,0.0,0.0,0.0
        # t_preds, t_masks = [], []

        for batch_idx, batch in enumerate(train_loader,1):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            t_loss += loss.item()
            tm = seg_metrics(preds,masks, args.seg_classes)
            t_px_acc += tm['px_acc']
            t_miou += tm ['miou']
            nt += 1
             
            if batch_idx % 10 == 0:
                print(f"  [seg] epoch {epoch} batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}")
 

        t_loss /= nt; t_px_acc /= nt; t_miou /= nt
        


        model.eval()
        v_loss, v_px_acc, v_miou,nv =0.0,0.0,0.0,0.0 
        # v_preds, v_masks = [], []

        with torch.no_grad(): 
            for batch in val_loader:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                logits = model(imgs)
                loss = criterion(logits, masks) + dice_loss(logits, masks)
                preds = logits.argmax(1)
                vm = seg_metrics(preds, masks, args.seg_classes)
                v_loss += loss.item()
                v_px_acc += vm['px_acc']
                v_miou += vm['miou']
                nv +=1

        v_loss /= nv; v_px_acc /= nv; v_miou /= nv
        
        scheduler.step(v_loss)

        print(f' Epoch {epoch:3d}/{args.epochs} |'
              f" Train_loss={t_loss:.4f} t_px_ac={tm['px_acc']:.1f}% t_miou={tm['miou']:.1f}% |"
              f" Val_loss={v_loss:.4f}  v_px_ac={vm['px_acc']:.1f}% v_miou={vm['miou']:.1f}% |")
        
        wandb.log({
            'seg/train_loss': t_loss,
            'seg/t_pc_acc': tm['px_acc'],
            'seg/t_miou': tm['miou'],
            'seg/val_loss': v_loss,
            'seg/v_pc_acc': vm['px_acc'],
            'seg/v_miou': vm['miou'],
            'seg/lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_chekpoint(model, args.seg_ckpt)

    wandb.finish

# Argument parsing

def parse_args():
    p =argparse.ArgumentParser(description='DA6401_Assignment_2_train')
    p.add_argument('-d', '--data_dir', type=str, default ='./data/pets')
    p.add_argument('-t', '--task', type=str, default='all', choices=['all', 'classification', 'localization', 'segmentation'])
    p.add_argument('-ep', '--epochs', type=int, default=50)
    p.add_argument('-bs', '--batch_size', type=int, default=16)
    p.add_argument('-lr', '--lr', type=float, default=1e-3)
    p.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    p.add_argument('-dp', '--dropout_p', type=float, default=0.5)
    p.add_argument('-vf', '--val_fraction', type=float, default=0.1)
    p.add_argument('-nm', '--num_workers', type=int, default=0)
    p.add_argument('-nb', '--num_breeds', type=int, default=37)
    p.add_argument('-sc', '--seg_classes', type=int, default=3)
    p.add_argument('-cck', '--cls_ckpt', type=str, default='checkpoints/classifier.pth')
    p.add_argument('-lck', '--loc_ckpt', type=str, default='checkpoints/localizer.pth')
    p.add_argument('-sck', '--seg_ckpt', type=str, default='checkpoints/segmentation.pth')
    p.add_argument('-wp', '--wandb_project', type=str, default='DA6401_Assignment_2')
    p.add_argument('-we', '--wandb_entity', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = get_device()
    print(f' Device: {device}')

    cls_train, cls_val, _ = make_loaders(args.data_dir, args.batch_size, args.val_fraction, args.num_workers)
    loc_train, loc_val, _ = make_loaders(args.data_dir, args.batch_size, args.val_fraction, args.num_workers)
    seg_train, seg_val, _ = make_loaders(args.data_dir, args.batch_size, args.val_fraction, args.num_workers)

    print(f'Train: {len(cls_train.dataset)} Val: {len(cls_val.dataset)}')

    if args.task in('all', 'classification'):
        train_classifier(args,cls_train, cls_val, device)
    
    if args.task in('all', 'localization'):
            train_localizer(args,loc_train, loc_val, device)
    
    if args.task in('all', 'segmentation'):
            train_segmentation(args,seg_train, seg_val, device)
    
    print('\nAll Training Complete.')


if __name__ == '__main__':
    main()
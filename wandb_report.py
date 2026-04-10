import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from train import (
    DiceLoss, get_device, make_loaders,
    cls_metrics, seg_metrics, loc_metrics,
)
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet, DecoderBlock
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
CMAP     = np.array([[0,0,0],[255,255,255],[128,128,128]], dtype=np.uint8)
CLASS_NAMES = OxfordIIITPetDataset.Class_Names

#---------------------------------------------------------------------------
# Helper functions for all the tasks
#---------------------------------------------------------------------------
def denorm(t):
    mean = torch.tensor(IMG_MEAN).view(3,1,1)
    std  = torch.tensor(IMG_STD).view(3,1,1)
    return (t.cpu() * std + mean).clamp(0,1)


def run_epoch_cls(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    loss_sum, preds, labels = 0.0, [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in loader:
            imgs = batch['image'].to(device); lbl = batch['label'].to(device)
            logits = model(imgs); loss = criterion(logits, lbl)
            if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(lbl.cpu().tolist())
    cm = cls_metrics(labels, preds)
    return loss_sum / len(loader), cm['accuracy'], cm['f1 Score']


def run_epoch_seg(model, loader, ce_fn, dice_fn, optimizer, device, train):
    model.train() if train else model.eval()
    loss_sum, px_sum, miou_sum, n = 0.0, 0.0, 0.0, 0
    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in loader:
            imgs = batch['image'].to(device); msk = batch['mask'].to(device)
            logits = model(imgs); loss = ce_fn(logits, msk) + dice_fn(logits, msk)
            if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
            sm = seg_metrics(logits.argmax(1), msk, 3)
            loss_sum += loss.item(); px_sum += sm['px_acc']; miou_sum += sm['miou']; n += 1
    return loss_sum/n, px_sum/n, miou_sum/n


# ---------------------------------------------------------------------------
# 2.1 BatchNorm effect on activations
# ---------------------------------------------------------------------------

def section_2_1(args):
    device = get_device()
    train_loader, val_loader, _ = make_loaders(args.data_dir, args.batch_size)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.1_batchnorm_effect', reinit=True)

    for use_bn in [True, False]:
        tag = 'with_BN' if use_bn else 'no_BN'
        model = VGG11Classifier(num_classes=37, dropout_p=0.5, use_bn=use_bn).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            t_loss, t_acc, _ = run_epoch_cls(model, train_loader, criterion, optimizer, device, True)
            v_loss, v_acc, _ = run_epoch_cls(model, val_loader,   criterion, None,      device, False)
            print(f'  2.1 [{tag}] epoch {epoch} | train={t_loss:.4f} val={v_loss:.4f} acc={v_acc:.1f}%')
            wandb.log({f'2.1/{tag}/train_loss': t_loss, f'2.1/{tag}/val_loss': v_loss,
                       f'2.1/{tag}/val_acc': v_acc, 'epoch': epoch})

        # Activation distribution at conv3_1
        model.eval(); acts = []
        hook = model.vgg11.conv3_1.register_forward_hook(lambda m,i,o: acts.append(o.detach().cpu()))
        with torch.no_grad(): model(next(iter(val_loader))['image'].to(device))
        hook.remove()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(acts[0].numpy().flatten(), bins=100, alpha=0.7)
        ax.set_title(f'conv3_1 activations — {tag}'); plt.tight_layout()
        wandb.log({f'2.1/activation_dist_{tag}': wandb.Image(fig)}); plt.close(fig)

    wandb.finish()


# ---------------------------------------------------------------------------
# 2.2 Dropout comparison
# ---------------------------------------------------------------------------

def section_2_2(args):
    device = get_device()
    train_loader, val_loader, _ = make_loaders(args.data_dir, args.batch_size)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.2_dropout_comparison', reinit=True)

    for p in [0.0, 0.2, 0.5]:
        tag = f'dropout_{p}'
        model = VGG11Classifier(num_classes=37, dropout_p=p).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            t_loss, t_acc, _ = run_epoch_cls(model, train_loader, criterion, optimizer, device, True)
            v_loss, v_acc, _ = run_epoch_cls(model, val_loader,   criterion, None,      device, False)
            print(f'  2.2 [{tag}] epoch {epoch} | train={t_loss:.4f} val={v_loss:.4f} acc={v_acc:.1f}%')
            wandb.log({f'2.2/{tag}/train_loss': t_loss, f'2.2/{tag}/val_loss': v_loss,
                       f'2.2/{tag}/val_acc': v_acc, 'epoch': epoch})

    wandb.finish()


# ---------------------------------------------------------------------------
# 2.3 Transfer learning strategies
# ---------------------------------------------------------------------------

def section_2_3(args):
    device = get_device()
    train_loader, val_loader, _ = make_loaders(args.data_dir, args.batch_size)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.3_transfer_learning', reinit=True)

    ce_fn, dice_fn = nn.CrossEntropyLoss(), DiceLoss()

    for strategy in ['frozen', 'partial', 'full']:
        print(f'\n--- 2.3: {strategy} ---')
        model = VGG11UNet(num_classes=3, dropout_p=0.5).to(device)

        if strategy == 'frozen':
            for p in model.encoder.parameters(): p.requires_grad = False
        elif strategy == 'partial':
            freeze = ['conv1_1', 'conv2_1', 'conv3_1', 'conv3_2']
            for name, param in model.encoder.named_parameters():
                if any(name.startswith(b) for b in freeze): param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=1e-4)

        for epoch in range(1, args.epochs + 1):
            t_loss, t_px, t_miou = run_epoch_seg(model, train_loader, ce_fn, dice_fn, optimizer, device, True)
            v_loss, v_px, v_miou = run_epoch_seg(model, val_loader,   ce_fn, dice_fn, None,      device, False)
            print(f'  epoch {epoch} | train={t_loss:.4f} miou={t_miou:.1f}% | val={v_loss:.4f} miou={v_miou:.1f}%')
            wandb.log({f'2.3/{strategy}/train_loss': t_loss, f'2.3/{strategy}/val_loss': v_loss,
                       f'2.3/{strategy}/train_miou': t_miou, f'2.3/{strategy}/val_miou': v_miou,
                       f'2.3/{strategy}/val_px_acc': v_px,   'epoch': epoch})

    wandb.finish()


# ---------------------------------------------------------------------------
# 2.4 Feature maps
# ---------------------------------------------------------------------------

def section_2_4(args):
    device = get_device()
    model = VGG11Classifier(num_classes=37).to(device)
    model.load_state_dict(torch.load(args.cls_ckpt, map_location=device)); model.eval()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.4_feature_maps', reinit=True)

    sample = OxfordIIITPetDataset(args.data_dir, split='test')[0]
    img_t  = sample['image'].unsqueeze(0).to(device)

    captured = {}
    def hook(name): return lambda m,i,o: captured.__setitem__(name, o.detach().cpu())
    h1 = model.vgg11.conv1_1.register_forward_hook(hook('conv1_1'))
    h5 = model.vgg11.conv5_1.register_forward_hook(hook('conv5_1'))
    with torch.no_grad(): model(img_t)
    h1.remove(); h5.remove()

    def plot_fmaps(fmap, title):
        fmap = fmap[0]; n = min(16, fmap.shape[0])
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i, ax in enumerate(axes.flatten()):
            if i < n: ax.imshow(fmap[i].numpy(), cmap='viridis')
            ax.axis('off')
        fig.suptitle(title); plt.tight_layout(); return fig

    wandb.log({
        '2.4/input':         wandb.Image((denorm(sample['image']).permute(1,2,0).numpy()*255).astype(np.uint8)),
        '2.4/conv1_1_fmaps': wandb.Image(plot_fmaps(captured['conv1_1'], 'conv1_1 — edges (224×224)')),
        '2.4/conv5_1_fmaps': wandb.Image(plot_fmaps(captured['conv5_1'], 'conv5_1 — semantics (14×14)')),
    })
    plt.close('all')
    print('2.4: Feature maps logged.'); wandb.finish()


# ---------------------------------------------------------------------------
# 2.5 Bounding box predictions table
# ---------------------------------------------------------------------------

def section_2_5(args):
    device = get_device()
    model = VGG11Localizer().to(device)
    model.load_state_dict(torch.load(args.loc_ckpt, map_location=device)); model.eval()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.5_bbox_table', reinit=True)

    ds    = OxfordIIITPetDataset(args.data_dir, split='test')
    table = wandb.Table(columns=['image', 'image_name', 'iou', 'confidence', 'failure'])

    with torch.no_grad():
        for idx in range(min(20, len(ds))):
            s        = ds[idx]
            gt       = s['bbox'].numpy()
            pred     = model(s['image'].unsqueeze(0).to(device))[0].cpu().numpy()
            iou      = (1.0 - IoULoss(reduction='none')(
                            torch.tensor(pred).unsqueeze(0),
                            torch.tensor(gt).unsqueeze(0))).item()
            conf     = float(np.exp(-np.linalg.norm(pred - gt) / 100))
            failure  = 'Yes' if conf > 0.5 and iou < 0.3 else 'No'

            orig = denorm(s['image']).permute(1,2,0).numpy()
            fig, ax = plt.subplots(figsize=(4,4)); ax.imshow(orig); ax.axis('off')
            for box, color, label in [(gt,'green','GT'), (pred,'red',f'IoU={iou:.2f}')]:
                cx,cy,w,h = box
                ax.add_patch(patches.Rectangle((cx-w/2,cy-h/2), w, h,
                             linewidth=2, edgecolor=color, facecolor='none'))
                ax.text(cx-w/2, cy-h/2-4, label, color=color, fontsize=7)
            plt.tight_layout()
            table.add_data(wandb.Image(fig), s['image_name'], round(iou,3), round(conf,3), failure)
            plt.close(fig)

    wandb.log({'2.5/bbox_table': table}); print('2.5: Table logged.'); wandb.finish()


# ---------------------------------------------------------------------------
# 2.6 Segmentation evaluation
# ---------------------------------------------------------------------------

def section_2_6(args):
    device = get_device()
    model = VGG11UNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.seg_ckpt, map_location=device)); model.eval()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.6_seg_eval', reinit=True)

    ds    = OxfordIIITPetDataset(args.data_dir, split='test')
    table = wandb.Table(columns=['original','gt_mask','pred_mask','px_acc','dice'])

    with torch.no_grad():
        for idx in range(min(5, len(ds))):
            s         = ds[idx]
            gt_mask   = s['mask'].numpy()
            logits    = model(s['image'].unsqueeze(0).to(device))
            pred_mask = logits.argmax(1)[0].cpu().numpy()
            m         = seg_metrics(torch.tensor(pred_mask), torch.tensor(gt_mask), 3)
            dice      = 2*m['miou'] / (100 + m['miou'])

            orig    = (denorm(s['image']).permute(1,2,0).numpy() * 255).astype(np.uint8)
            table.add_data(wandb.Image(orig),
                           wandb.Image(CMAP[gt_mask.astype(int)]),
                           wandb.Image(CMAP[pred_mask.astype(int)]),
                           round(m['px_acc'],2), round(dice,2))

    wandb.log({'2.6/seg_table': table}); print('2.6: Table logged.'); wandb.finish()


# ---------------------------------------------------------------------------
# 2.7 Novel image pipeline
# ---------------------------------------------------------------------------

def section_2_7(args):


    device = get_device()
    cls_model = VGG11Classifier(num_classes=37).to(device)
    cls_model.load_state_dict(torch.load(args.cls_ckpt, map_location=device)); cls_model.eval()
    loc_model = VGG11Localizer().to(device)
    loc_model.load_state_dict(torch.load(args.loc_ckpt, map_location=device)); loc_model.eval()
    seg_model = VGG11UNet(num_classes=3).to(device)
    seg_model.load_state_dict(torch.load(args.seg_ckpt, map_location=device)); seg_model.eval()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.7_novel_pipeline', reinit=True)

    transform = A.Compose([A.Resize(224,224),
                            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
                            ToTensorV2()])

    if not os.path.exists(args.novel_dir):
        print(f"novel_dir '{args.novel_dir}' not found."); wandb.finish(); return

    files = [f for f in os.listdir(args.novel_dir)
             if f.lower().endswith(('.jpg','.jpeg','.png'))][:3]
    table = wandb.Table(columns=['original','bbox_overlay','seg_mask','breed','confidence'])

    for fname in files:
        pil  = Image.open(os.path.join(args.novel_dir, fname)).convert('RGB')
        t    = transform(image=np.array(pil))['image'].unsqueeze(0).to(device)
        orig = np.array(pil.resize((224,224)))

        with torch.no_grad():
            cls_out  = cls_model(t)
            pred_cls = cls_out.argmax(1).item()
            conf     = torch.softmax(cls_out,1).max().item() * 100
            pred_box = loc_model(t)[0].cpu().numpy()
            pred_seg = seg_model(t).argmax(1)[0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(4,4)); ax.imshow(orig); ax.axis('off')
        cx,cy,w,h = pred_box
        ax.add_patch(patches.Rectangle((cx-w/2,cy-h/2), w, h,
                     linewidth=2, edgecolor='red', facecolor='none'))
        ax.set_title(f'{CLASS_NAMES[pred_cls]} ({conf:.1f}%)'); plt.tight_layout()

        table.add_data(wandb.Image(orig), wandb.Image(fig),
                       wandb.Image(CMAP[pred_seg.astype(int)]),
                       CLASS_NAMES[pred_cls], round(conf,2))
        plt.close(fig)

    wandb.log({'2.7/novel_pipeline': table}); print('2.7: Logged.'); wandb.finish()


# ---------------------------------------------------------------------------
# 2.8 Meta-analysis
# ---------------------------------------------------------------------------

def section_2_8(args):
    device = get_device()
    _, _, test_loader = make_loaders(args.data_dir, args.batch_size)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name='2.8_meta_analysis', reinit=True)

    ce_fn, dice_fn = nn.CrossEntropyLoss(), DiceLoss()

    # Classification
    cls_model = VGG11Classifier(num_classes=37).to(device)
    cls_model.load_state_dict(torch.load(args.cls_ckpt, map_location=device))
    _, cls_acc, cls_f1 = run_epoch_cls(cls_model, test_loader, ce_fn, None, device, False)
    print(f'  cls acc={cls_acc:.2f}% f1={cls_f1:.2f}%')

    # Localization
    loc_model = VGG11Localizer().to(device)
    loc_model.load_state_dict(torch.load(args.loc_ckpt, map_location=device)); loc_model.eval()
    iou_fn = IoULoss(reduction='none'); ious = []
    with torch.no_grad():
        for batch in test_loader:
            p = loc_model(batch['image'].to(device)).cpu()
            ious.extend((1 - iou_fn(p, batch['bbox'])).tolist())
    loc_miou = np.mean(ious) * 100
    print(f'  loc mean_iou={loc_miou:.2f}%')

    # Segmentation
    seg_model = VGG11UNet(num_classes=3).to(device)
    seg_model.load_state_dict(torch.load(args.seg_ckpt, map_location=device))
    _, seg_px, seg_miou = run_epoch_seg(seg_model, test_loader, ce_fn, dice_fn, None, device, False)
    seg_dice = 2 * seg_miou / (100 + seg_miou)
    print(f'  seg px_acc={seg_px:.2f}% miou={seg_miou:.2f}%')

    # Summary chart
    metrics = {'Cls Acc': cls_acc, 'Cls F1': cls_f1, 'Loc mIoU': loc_miou,
               'Seg Px Acc': seg_px, 'Seg mIoU': seg_miou, 'Seg Dice': seg_dice}
    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=['steelblue','cornflowerblue','coral','mediumseagreen','orchid','goldenrod'])
    ax.set_ylim(0,100); ax.set_ylabel('Score (%)'); ax.set_title('Final Test Metrics — All Tasks')
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15, ha='right'); plt.tight_layout()

    wandb.log({**{f'2.8/test_{k.lower().replace(" ","_")}': v for k,v in metrics.items()},
               '2.8/final_metrics': wandb.Image(fig)})
    plt.close(fig); print('2.8: Meta-analysis complete.'); wandb.finish()


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DA6401 A2 — Analysis (2.1–2.8)')
    p.add_argument('--section',       type=str, required=True,
                   choices=['2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8'])
    p.add_argument('--data_dir',      type=str, default='./data/pets')
    p.add_argument('--novel_dir',     type=str, default='./novel_images')
    p.add_argument('--cls_ckpt',      type=str, default='checkpoints/classifier.pth')
    p.add_argument('--loc_ckpt',      type=str, default='checkpoints/localizer.pth')
    p.add_argument('--seg_ckpt',      type=str, default='checkpoints/segmentation.pth')
    p.add_argument('--epochs',        type=int, default=10)
    p.add_argument('--batch_size',    type=int, default=16)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--wandb_project', type=str, default='DA6401_Assignment_2')
    p.add_argument('--wandb_entity',  type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    {
        '2.1': section_2_1, '2.2': section_2_2, '2.3': section_2_3,
        '2.4': section_2_4, '2.5': section_2_5, '2.6': section_2_6,
        '2.7': section_2_7, '2.8': section_2_8,
    }[args.section](args)
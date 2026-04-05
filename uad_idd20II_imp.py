import os
import gc
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from PIL import ImageDraw

# torchvision MobileNetV2 backbone
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# =============================================================================
# SECTION 1: CONFIG — change only this if your path ever moves
# =============================================================================

BASE_PATH  = r"S:\C++Coding\IACV\idd20kII"
SAVE_DIR   = r"S:\C++Coding\IACV"
IMG_SIZE   = 512         # paper uses 1280×964; 512 is a good GPU-friendly size
BATCH_SIZE = 5       # paper uses batch size 3
EPOCHS     = 20
LR         = 0.01
NUM_CLASSES = 3           # 0=non-drivable, 1=road, 2=drivable fallback


# =============================================================================
# SECTION 2: MODEL ARCHITECTURE
# =============================================================================

# -----------------------------------------------------------------------------
# 2A. ASPP MODULE  (5 branches: 1×1, dil-6, dil-12, dil-18, GAP)
#     Exactly as described in the paper / standard DeepLabv3+
# -----------------------------------------------------------------------------

class ASPPConv(nn.Sequential):
    """Single dilated conv branch inside ASPP."""
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """Global average-pooling branch inside ASPP."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(self.gap(x), size=size,
                             mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Branches: 1×1 conv | dil-6 | dil-12 | dil-18 | GAP
    All branches output 256 channels → concat → 1×1 project → 256 ch.
    """
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.b0 = nn.Sequential(                         # 1×1
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.b1 = ASPPConv(in_ch, out_ch, dilation=6)
        self.b2 = ASPPConv(in_ch, out_ch, dilation=12)
        self.b3 = ASPPConv(in_ch, out_ch, dilation=18)
        self.b4 = ASPPPooling(in_ch, out_ch)

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        branches = [self.b0(x), self.b1(x), self.b2(x),
                    self.b3(x), self.b4(x)]
        return self.project(torch.cat(branches, dim=1))


# -----------------------------------------------------------------------------
# 2B. UNIT ATTENTION MODULE (UAM)
#     = Dual Attention Module (PAM + CAM) + Spatial Attention (SA)
#     PAM  → paper Eq. (1-4)
#     CAM  → paper Eq. (5-8)   ← includes the max(S)-S trick
#     SA   → paper Eq. (9-10)
# -----------------------------------------------------------------------------

class PositionAttentionModule(nn.Module):
    """
    PAM — captures spatial (position) dependencies.
    Paper Eq. (1-4).
    """
    def __init__(self, in_ch):
        super().__init__()
        mid = max(in_ch // 8, 1)
        self.query_conv = nn.Conv2d(in_ch, mid, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_ch, mid, kernel_size=1)
        self.value_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.alpha      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        # Q: (B, N, mid)   K: (B, mid, N)
        q = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, N)

        # Spatial attention map P: (B, N, N)
        attn = self.softmax(torch.bmm(q, k))            # Eq. (2)

        # V: (B, C, N)
        v   = self.value_conv(x).view(B, C, N)
        # Weighted output: (B, C, N) → (B, C, H, W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)

        return self.alpha * out + x                      # Eq. (4)


class ChannelAttentionModule(nn.Module):
    """
    CAM — captures channel dependencies.
    Paper Eq. (5-8), including the max(S)-S normalisation trick.
    """
    def __init__(self):
        super().__init__()
        self.beta    = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        # F1: (B, C, N)   F2: (B, N, C)
        f1 = x.view(B, C, N)
        f2 = f1.permute(0, 2, 1)

        # Channel similarity matrix S: (B, C, C)    Eq. (5)
        S = torch.bmm(f1, f2)

        # max(S) - S   →   softmax   →   weight matrix G   Eq. (6)
        # max over last dim, kept for broadcasting
        S_max = S.max(dim=-1, keepdim=True).values
        G = self.softmax(S_max - S)

        # Weighted channel features: (B, C, N) → (B, C, H, W)   Eq. (7)
        fc = torch.bmm(G, f1).view(B, C, H, W)

        return self.beta * fc + x                        # Eq. (8)


class DualAttentionModule(nn.Module):
    """
    DAM = PAM + CAM with 3×3 conv on each branch, then element-wise sum.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.pam      = PositionAttentionModule(in_ch)
        self.cam      = ChannelAttentionModule()
        self.conv_pam = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_cam = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f_pam = self.conv_pam(self.pam(x))
        f_cam = self.conv_cam(self.cam(x))
        return f_pam + f_cam                             # F_DAM


class SpatialAttentionModule(nn.Module):
    """
    SA Module — 7×7 conv on channel-max + channel-avg concat.
    Paper Eq. (9-10).
    """
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # F3: channel-max   F4: channel-avg    Eq. (9)
        f3, _ = torch.max(x, dim=1, keepdim=True)
        f4    = torch.mean(x, dim=1, keepdim=True)
        # F5: (B, 2, H, W) → spatial attention map
        f_sa  = self.sigmoid(self.conv(torch.cat([f3, f4], dim=1)))
        return x * f_sa                                  # Eq. (10)


class UnitAttentionModule(nn.Module):
    """
    UAM = DAM (PAM + CAM) followed by SA.
    Placed after ASPP in the encoder.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.dam = DualAttentionModule(in_ch)
        self.sa  = SpatialAttentionModule()

    def forward(self, x):
        f_dam = self.dam(x)
        return self.sa(f_dam)


# -----------------------------------------------------------------------------
# 2C. ENCODER  — MobileNetV2 backbone + ASPP + UAM
#     Low-level features: output of MobileNetV2 layer index 4
#                         (stride 4, 24 channels — rich edge/detail info)
#     High-level features: output of MobileNetV2 last layer
#                          (stride 16 or 32, 320 channels)
# -----------------------------------------------------------------------------

class MobileNetV2Encoder(nn.Module):
    """
    Wraps torchvision MobileNetV2.
    Returns:
        low_level  — from features[4]   : (B, 24,  H/4,  W/4)
        high_level — from features[18]  : (B, 1280, H/32, W/32)
                     then passed through ASPP → (B, 256, H/32, W/32)
                     then through UAM   → (B, 256, H/32, W/32)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights  = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        feats    = backbone.features          # 19 blocks (0–18)

        # Split at layer 4 for low-level features (24 ch, stride 4)
        self.low_level_extractor  = nn.Sequential(*feats[:4])   # → 24 ch
        # MobileNetV2 ends with a 1280-ch Conv2d (features[18]), not 320
        self.high_level_extractor = nn.Sequential(*feats[4:])   # -> 1280 ch

        self.aspp = ASPP(in_ch=1280, out_ch=256)
        self.uam  = UnitAttentionModule(in_ch=256)

    def forward(self, x):
        low  = self.low_level_extractor(x)   # (B, 24,  H/4,  W/4)
        high = self.high_level_extractor(low) # (B, 1280, H/32, W/32)
        high = self.aspp(high)               # (B, 256, H/32, W/32)
        high = self.uam(high)               # (B, 256, H/32, W/32)
        return low, high


# -----------------------------------------------------------------------------
# 2D. DECODER  — matches Table 1 of the paper exactly
#     Step 1: Upsample high-level ×8  (NOT ×4 — paper Table 1 says rate=4
#             but the footnote clarifies the first upsample brings H/32→H/4)
#             H/32 × 8 = H/4  which matches low-level spatial resolution
#     Step 2: Conv 1×1 on low-level to reduce channels
#     Step 3: Element-wise add (paper) / concat then conv (our impl keeps
#             channel dims compatible via 1×1 on low first)
#     Step 4: Conv 3×3 refinement
#     Step 5: Upsample ×4 → original resolution
# -----------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, high_ch=256, low_ch=24, num_classes=NUM_CLASSES):
        super().__init__()
        # Reduce low-level channels to 48 (standard DeepLabv3+ value)
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_ch, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        # After concat: high(256) + low(48) = 304 channels
        self.refine = nn.Sequential(
            nn.Conv2d(high_ch + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, high, low, input_size):
        # Step 1: Upsample high ×8 so it matches low-level spatial size (H/4)
        high_up = F.interpolate(high, size=low.shape[2:],
                                mode='bilinear', align_corners=False)
        # Step 2: Reduce low-level channels
        low_proj = self.low_conv(low)
        # Step 3: Concat high + low
        x = torch.cat([high_up, low_proj], dim=1)
        # Step 4: 3×3 conv refinement
        x = self.refine(x)
        # Step 5: Upsample ×4 → original input resolution
        x = F.interpolate(x, size=input_size,
                          mode='bilinear', align_corners=False)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# 2E. FULL UAD NETWORK
# -----------------------------------------------------------------------------

class UAD_Network(nn.Module):
    """
    Unit Attention DeepLabv3+ (UAD).
    Architecture: MobileNetV2 → ASPP → UAM → Decoder
    Matches the paper Figure 2 exactly.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.encoder = MobileNetV2Encoder(pretrained=pretrained)
        self.decoder = Decoder(high_ch=256, low_ch=24, num_classes=num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        low, high  = self.encoder(x)
        return self.decoder(high, low, input_size)


# =============================================================================
# SECTION 3: DATASET
# =============================================================================

# Class mapping (paper Section 3)
# Class 0 = non-drivable  (all 38 remaining labels)
# Class 1 = road
# Class 2 = drivable fallback
ROAD_LABELS     = {"road"}
FALLBACK_LABELS = {"drivable fallback"}     # only this per paper Section 3


def json_to_mask(json_path, img_width, img_height):
    """Render gtFine_polygons.json into a 2-D uint8 class mask."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = Image.fromarray(np.zeros((img_height, img_width), dtype=np.uint8))
    draw = ImageDraw.Draw(mask)

    for obj in data.get('objects', []):
        if obj.get('deleted', 1):
            continue
        label   = obj.get('label', '').lower().strip()
        polygon = obj.get('polygon', [])
        if len(polygon) < 3:
            continue
        pts = [(float(p[0]), float(p[1])) for p in polygon]

        if label in ROAD_LABELS:
            draw.polygon(pts, fill=1)
        elif label in FALLBACK_LABELS:
            draw.polygon(pts, fill=2)

    return np.array(mask)


class IDD20kIIDataset(Dataset):
    def __init__(self, base_dir, split='train'):
        self.img_paths = sorted(glob.glob(
            os.path.join(base_dir, 'leftImg8bit', split, '**', '*.jpg'),
            recursive=True
        ))

        self.valid_pairs = []
        for img_path in self.img_paths:
            fname     = os.path.basename(img_path)
            base_name = fname.replace('_leftImg8bit.jpg', '')
            subfolder = os.path.basename(os.path.dirname(img_path))
            json_path = os.path.join(
                base_dir, 'gtFine', split, subfolder,
                f"{base_name}_gtFine_polygons.json"
            )
            if os.path.exists(json_path):
                self.valid_pairs.append((img_path, json_path))

        print(f"Found {len(self.valid_pairs)} valid image-label pairs in '{split}' split.")

        # ImageNet normalisation used with pretrained MobileNetV2
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, json_path = self.valid_pairs[idx]

        img  = Image.open(img_path).convert("RGB")
        W, H = img.size

        mask_np  = json_to_mask(json_path, W, H)
        mask_pil = Image.fromarray(mask_np)

        image      = self.transform(img)
        final_mask = np.array(self.mask_transform(mask_pil))

        return image, torch.from_numpy(final_mask).long()


# =============================================================================
# SECTION 4: EVALUATION METRICS
# =============================================================================

def calculate_iou(prediction_mask, ground_truth_mask, num_classes=NUM_CLASSES):
    """Per-class IoU and mIoU, ignoring classes absent from both pred & GT."""
    pred_flat  = prediction_mask.view(-1)
    truth_flat = ground_truth_mask.view(-1)
    iou_per_class = []

    for cls in range(num_classes):
        pred_cls  = (pred_flat  == cls)
        truth_cls = (truth_flat == cls)
        intersection = (pred_cls & truth_cls).sum().float()
        union        = (pred_cls | truth_cls).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).item())

    valid = [v for v in iou_per_class if not np.isnan(v)]
    miou  = sum(valid) / len(valid) if valid else 0.0
    return iou_per_class, miou


# =============================================================================
# SECTION 5: TRAINING
# =============================================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # pretrained=True to load ImageNet weights for MobileNetV2 backbone,
    # which dramatically speeds up convergence (transfer learning).
    model     = UAD_Network(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=0.9, weight_decay=0.0001)

    # Poly LR decay: lr = base_lr * (1 - iter/max_iter)^0.9
    # This is the standard scheduler used with DeepLab models.
    total_steps = EPOCHS * 1  # will be updated once loader length is known
    def poly_lr(step):
        return (1 - step / max(total_steps, 1)) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    checkpoint_path = os.path.join(SAVE_DIR, "uad_model_idd20kII.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)

        # Unwrap {'model':…,'epoch':…} wrapper if present
        if isinstance(state, dict) and 'model' in state:
            weights     = state['model']
            saved_epoch = state.get('epoch', 0)
        else:
            weights     = state
            saved_epoch = 0

        # Check for NaN weights
        has_nan = any(torch.isnan(v).any() for v in weights.values()
                      if isinstance(v, torch.Tensor))

        # Check for architecture mismatch — compare key sets
        model_keys   = set(model.state_dict().keys())
        ckpt_keys    = set(weights.keys())
        keys_match   = model_keys == ckpt_keys

        if has_nan:
            print("WARNING: checkpoint has NaN weights — ignoring, training from scratch.")
            print(f"  (old checkpoint kept at {checkpoint_path})")
        elif not keys_match:
            extra   = ckpt_keys - model_keys
            missing = model_keys - ckpt_keys
            print("WARNING: checkpoint is from a different architecture — ignoring.")
            print(f"  Unexpected keys in checkpoint : {len(extra)}")
            print(f"  Missing keys from checkpoint  : {len(missing)}")
            print("  This is normal if you previously ran the old simple model.")
            print("  Training from scratch with the new UAD architecture.")
            # Rename old checkpoint so it isn't picked up again
            old_path = checkpoint_path.replace(".pth", "_old_arch.pth")
            os.rename(checkpoint_path, old_path)
            print(f"  Old checkpoint moved to: {old_path}")
        else:
            model.load_state_dict(weights)
            start_epoch = saved_epoch
            print(f"Resumed from epoch {start_epoch}: {checkpoint_path}")
    else:
        print("No saved model found — training from scratch.")

    train_dataset = IDD20kIIDataset(BASE_PATH, 'train')
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
    )

    # Now we know the loader length — fix poly scheduler
    total_steps = EPOCHS * len(train_loader)
    global_step = start_epoch * len(train_loader)

    # Re-create scheduler with correct total steps
    def poly_lr_fixed(step):
        return max((1 - step / total_steps) ** 0.9, 1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=poly_lr_fixed, last_epoch=global_step - 1
    )

    print(f"Training for {EPOCHS} epochs, {len(train_loader)} batches/epoch...\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents weight explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if (i + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | "
                      f"Batch {i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} complete | Avg Loss: {avg_loss:.4f}\n")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Save with epoch info so resuming is accurate
        torch.save(
            {'model': model.state_dict(), 'epoch': epoch + 1},
            checkpoint_path
        )

    print(f"Training complete. Model saved to: {checkpoint_path}")
    return model


# =============================================================================
# SECTION 6: EVALUATION (High-Speed Vectorized Version)
# =============================================================================

def evaluate(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Set num_workers=0 for Windows stability during eval
    val_loader = DataLoader(
        IDD20kIIDataset(BASE_PATH, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=0, 
        pin_memory=(device.type == 'cuda'),
    )

    # Accumulate confusion matrix
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

    print(f"Evaluating on {len(val_loader)} images...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = torch.max(outputs, 1)

            # --- SPEED FIX: Vectorized Confusion Matrix Calculation ---
            # This replaces the pixel-by-pixel loop with a single GPU operation
            mask = (labels >= 0) & (labels < NUM_CLASSES)
            label_flat = labels[mask].view(-1)
            pred_flat  = predicted[mask].view(-1)
            
            # Combine labels and predictions into a single index
            indices = NUM_CLASSES * label_flat + pred_flat
            m = torch.bincount(indices, minlength=NUM_CLASSES**2)
            confusion += m.reshape(NUM_CLASSES, NUM_CLASSES)
            # ---------------------------------------------------------

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(val_loader)} images...")

            del images, labels, outputs, predicted
            gc.collect()

    # Move to CPU for final metric printing
    confusion = confusion.cpu().float()

    # MIoU per class = TP / (TP + FP + FN)
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    denom = tp + fp + fn
    
    miou_per_class = []
    for c in range(NUM_CLASSES):
        if denom[c] > 0:
            miou_per_class.append((tp[c] / denom[c]).item())
    
    miou = (sum(miou_per_class) / len(miou_per_class)) * 100
    
    # MPA = TP / total_correct_class_pixels
    mpa = (tp.sum() / confusion.sum()) * 100

    print(f"\nMean Pixel Accuracy (MPA): {mpa:.2f}%  (paper target: 92.01%)")
    print(f"Mean IoU (MIoU):           {miou:.2f}%  (paper target: 85.99%)")

    class_names = ['non-drivable', 'road', 'drivable fallback']
    for c, name in enumerate(class_names):
        if c < len(miou_per_class):
            print(f"  {name:20s}  IoU: {miou_per_class[c]*100:.2f}%")

    return mpa, miou
# =============================================================================
# SECTION 7: VISUALISATION
# =============================================================================

# Denormalise for display (reverses ImageNet normalisation)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm(t):
    return (t * _STD + _MEAN).clamp(0, 1)


def visualize(model, idx=25):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = IDD20kIIDataset(BASE_PATH, split='val')
    model.eval()

    image, label = dataset[idx]
    with torch.no_grad():
        output       = model(image.unsqueeze(0).to(device))
        _, predicted = torch.max(output, 1)
        predicted    = predicted.squeeze().cpu().numpy()

    # Colour map: 0=grey, 1=orange (road), 2=cyan (drivable fallback)
    cmap = np.array([[80, 80, 80], [255, 140, 0], [0, 200, 200]], dtype=np.uint8)

    def mask_to_rgb(m):
        return cmap[m]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(denorm(image).permute(1, 2, 0).numpy())
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(mask_to_rgb(label.numpy()))
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(mask_to_rgb(predicted))
    axes[2].set_title("UAD Prediction")
    axes[2].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap[0]/255, label='Non-drivable'),
        Patch(facecolor=cmap[1]/255, label='Road'),
        Patch(facecolor=cmap[2]/255, label='Drivable fallback'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=11, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(SAVE_DIR, "prediction_uad_idd20kII.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualisation saved as {save_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    model = train()
    evaluate(model)
    visualize(model, idx=25)
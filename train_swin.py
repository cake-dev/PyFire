import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import os
import config
from dataset_parallel import FireEmulationDataset 
# IMPORT THE NEW MODEL
from model_swin import SwinUNetFireEmulator

# --- Hyperparameters ---
# Swin is heavier than UNet, so we might need to reduce batch size or use accumulation
BATCH_SIZE = 8          # Reduced from 24 due to Swin memory usage
LEARNING_RATE = 5e-5    # Transformers usually prefer lower LR than CNNs
EPOCHS = 30             
DATA_DIR = "./training_data_v1"
CHECKPOINT_DIR = "./checkpoints_swin"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Enable TF32 for Ampere GPUs (A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class ContiguousGrowthLoss(nn.Module):
    """
    Same loss function as before, proven to work for fire growth.
    """
    def __init__(self, active_weight=10.0, growth_penalty=1.0, tv_weight=0.5):
        super().__init__()
        self.active_weight = active_weight
        self.growth_penalty = growth_penalty
        self.tv_weight = tv_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        
        active_mask = (torch.abs(target) > 0.05).float()
        underestimation_mask = (target > pred).float() * active_mask
        
        weight_map = 1.0 + (self.active_weight * active_mask)
        weight_map = weight_map + (self.active_weight * self.growth_penalty * underestimation_mask)
        
        mse_loss = (loss * weight_map).mean()
        
        # 3D Total Variation
        diff_d = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).mean()
        diff_h = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]).mean()
        diff_w = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]).mean()
        tv_loss = diff_d + diff_h + diff_w
        
        return mse_loss + (self.tv_weight * tv_loss)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model: SwinUNetFireEmulator (3D)")

    full_dataset = FireEmulationDataset(DATA_DIR, cache_in_ram=False)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Num_workers adjusted for A100 IO capabilities
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # Initialize 3D Swin Model
    # img_size matches (Z, X, Y) -> (NZ, NX, NY) from config
    model = SwinUNetFireEmulator(
        in_channels=7, 
        out_channels=1,
        img_size=(config.NZ, config.NX, config.NY) 
    ).to(device)
    
    criterion = ContiguousGrowthLoss(active_weight=10.0, growth_penalty=1.0, tv_weight=0.5)
    
    # AdamW is crucial for Transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine Scheduler often helps convergence with Swin
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Step Scheduler
        scheduler.step()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Results: Train Loss: {running_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_swin_model.pth"))
            print("--> Best model saved.")

if __name__ == "__main__":
    train()
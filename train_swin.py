import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import os
import config
from dataset_parallel import FireEmulationDataset 
from model_swin import SwinUNetFireEmulator
import torch.nn.functional as F

# --- Hyperparameters ---
BATCH_SIZE = 8          
LEARNING_RATE = 2e-5    
EPOCHS = 30             
DATA_DIR = "./training_data_test"
CHECKPOINT_DIR = "./checkpoints_swin"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class FrontWeightedLoss(nn.Module):
    def __init__(self, active_weight=20.0, growth_penalty=5.0, distance_penalty=10.0):
        super().__init__()
        self.active_weight = active_weight
        self.growth_penalty = growth_penalty
        self.distance_penalty = distance_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, input_rr):
        """
        pred: Model prediction (Next step RR)
        target: Ground truth (Next step RR)
        input_rr: The RR at the previous step (t). Used to define the 'Front'.
        """
        # Clamp prediction
        pred = torch.clamp(pred, -1.0, 10.0)
        
        # Base Loss
        loss = self.mse(pred, target)
        
        # --- 1. Identify the Fire Front ---
        # Dilate the current input fire to find the "valid growth zone"
        # Any fire predicted OUTSIDE this zone is physically impossible (teleportation).
        # Input is (B, 1, D, H, W)
        # MaxPool3d with kernel 3, stride 1, padding 1 acts as morphological dilation
        with torch.no_grad():
            # Threshold to binary for mask generation
            curr_fire_mask = (input_rr > 0.01).float()
            # Dilation: Valid zone is current fire + 1 voxel neighbor
            valid_growth_zone = F.max_pool3d(curr_fire_mask, kernel_size=3, stride=1, padding=1)
        
        # --- 2. Masks ---
        active_mask = (target > 0.01).float()
        
        # Underestimation (Missing Growth): Target > Pred
        under_mask = (target > pred).float() * active_mask
        
        # Overestimation (Hallucination): Pred > Target
        over_mask = (pred > (target + 0.1)).float()
        
        # Teleportation (Physical Violation): Predicting fire outside the valid growth zone
        # This is the "jump everywhere" fix.
        teleport_mask = over_mask * (1.0 - valid_growth_zone)
        
        # --- 3. Weight Map ---
        weights = 1.0
        
        # Boost valid fire regions
        weights += (self.active_weight * active_mask)
        
        # Boost missed growth (but only near the front!)
        weights += (self.active_weight * self.growth_penalty * under_mask)
        
        # MASSIVE penalty for teleportation (jumping across map)
        weights += (self.distance_penalty * teleport_mask)
        
        mse_loss = (loss * weights).mean()
        
        return mse_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model: SwinUNetFireEmulator (3D Custom - Front Constrained)")

    full_dataset = FireEmulationDataset(DATA_DIR, cache_in_ram=False)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = SwinUNetFireEmulator(
        in_channels=7, 
        out_channels=1,
        img_size=(config.NZ, config.NX, config.NY) 
    ).to(device)
    
    # New Loss with Distance Penalty
    criterion = FrontWeightedLoss(active_weight=20.0, growth_penalty=3.0, distance_penalty=20.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Log Transform
            inputs[:, 1] = torch.log1p(torch.clamp(inputs[:, 1], min=0.0))
            # Capture the current RR (index 1) for the proximity mask
            # We need a copy that is detached from the graph for mask generation
            current_rr = inputs[:, 1:2].detach() 
            
            targets = torch.log1p(torch.clamp(targets, min=0.0))

            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(inputs)
                # Pass current_rr to the loss to calculate distance violations
                loss = criterion(outputs, targets, current_rr)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # Validation Loop (Simplified)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                inputs[:, 1] = torch.log1p(torch.clamp(inputs[:, 1], min=0.0))
                current_rr = inputs[:, 1:2]
                targets = torch.log1p(torch.clamp(targets, min=0.0))
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, current_rr)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Results: Train Loss: {running_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_swin_model.pth"))
            print("--> Best model saved.")

if __name__ == "__main__":
    # supress this warning: <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cuda module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.driver module instead.
    
    train()
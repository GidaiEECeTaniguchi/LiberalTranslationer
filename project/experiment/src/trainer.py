import os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import matplotlib
# ヘッドレス環境（サーバー等）でのプロットエラー防止
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# ===============================
# 1. LRFinder (メモリ節約版)
# ===============================
class LRFinder:
    """最適な学習率を自動探索 (CPUメモリに配慮)"""
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def find(self, train_loader, min_lr=1e-7, max_lr=1, num_iter=100, smooth_f=0.05):
        logger.info(f"🔍 LR Finder: Searching ({min_lr} to {max_lr})...")
        
        # ⚠️ deepcopyを避け、重みだけをCPUにクローンして保存 (メモリ節約)
        original_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        original_opt_state = {k: v for k, v in self.optimizer.state_dict().items()}
        
        mult = (max_lr / min_lr) ** (1 / num_iter)
        lr = min_lr
        self.optimizer.param_groups[0]['lr'] = lr
        
        avg_loss = 0.0
        best_loss = float('inf')
        
        self.model.train()
        iterator = iter(train_loader)
        
        for i in range(num_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 損失の平滑化
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss if i > 0 else loss.item()
            
            if avg_loss < best_loss: best_loss = avg_loss
            if i > 1 and avg_loss > 4 * best_loss:
                logger.warning("⚠️ Loss diverged, stopping LR finder.")
                break
                
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)
            
            loss.backward()
            self.optimizer.step()
            
            lr *= mult
            for pg in self.optimizer.param_groups: pg['lr'] = lr
            
        # 状態の復元
        self.model.load_state_dict(original_weights)
        self.optimizer.load_state_dict(original_opt_state)
        
        suggested_lr = self.history['lr'][self.history['loss'].index(min(self.history['loss']))] // 10
        logger.info(f"✅ Suggested LR: {suggested_lr:.2e}")
        return suggested_lr

    def plot(self, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

# ===============================
# 2. 学習補助
# ===============================
class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

def get_total_steps(phase_epochs, loaders_map, config):
    """全フェーズの総更新ステップ数を正確に計算"""
    total_updates = 0
    
    for phase_idx, n_epochs in enumerate(phase_epochs):
        if n_epochs <= 0: continue
        
        # フェーズごとのローダー構成をシミュレート
        if phase_idx == 0:
            loaders = [loaders_map["span"], loaders_map["bywork"], loaders_map["chunk"]]
        elif phase_idx == 1:
            loaders = [loaders_map["chunk"], loaders_map["bywork"], loaders_map["span"]]
        else:
            # Phase 3: アンカー(span)を混ぜる
            loaders = [loaders_map["practical_chunk"], loaders_map["practical_line"], loaders_map["span"]]
            
        phase_steps = sum(len(l) for l in loaders if l is not None)
        total_updates += (phase_steps // config.accumulation_steps) * n_epochs
        
    return total_updates

# ===============================
# 3. 学習コア
# ===============================
def train_epoch(model, loaders, optimizer, scheduler, scaler, device, config, epoch, criterion=None, ema=None):
    model.train()
    total_loss = 0
    update_count = 0
    
    # 複数のローダーを順番に回す
    for loader in loaders:
        if loader is None: continue
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=config.use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = (criterion(outputs.logits, labels) if criterion else outputs.loss) / config.accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % config.accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                if ema: ema.update()
                if scheduler: scheduler.step()
                update_count += 1

            total_loss += loss.item() * config.accumulation_steps
            pbar.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.1e}", loss=f"{loss.item()*config.accumulation_steps:.4f}")

    return total_loss / update_count if update_count > 0 else total_loss


# trainer.py に追加

def evaluate_model(model, val_loader, device, config, criterion=None, ema=None):
    """検証データでの損失を計算"""
    model.eval()
    if ema: ema.apply_shadow() # EMAを適用して評価
    
    total_loss = 0
    # 評価時は勾配計算をオフにしてメモリを節約
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # CPU/GPU 警告を回避した autocast
            with torch.amp.autocast(device_type=device.type, enabled=config.use_amp and device.type == 'cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = (criterion(outputs.logits, labels) if criterion else outputs.loss)
            
            total_loss += loss.item()
            
    if ema: ema.restore() # 学習用に重みを戻す
    model.train()
    return total_loss / len(val_loader) if len(val_loader) > 0 else 0
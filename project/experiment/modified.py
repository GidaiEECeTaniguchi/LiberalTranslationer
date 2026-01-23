import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
from torch.utils.data import Dataset, Subset
import os
from pathlib import Path
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import copy
import matplotlib.pyplot as plt

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================
# 🆕 新機能1: Focal Loss
# ===============================
class FocalLoss(nn.Module):
    """
    難しいサンプルにより注目するFocal Loss
    
    Args:
        alpha: クラスバランス調整パラメータ
        gamma: 焦点パラメータ（大きいほど難しいサンプルに注目）
        reduction: 損失の集約方法
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets, ignore_index=-100):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        # logitsを2次元に変形
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # ignore_indexをマスク
        mask = targets_flat != ignore_index
        valid_logits = logits_flat[mask]
        valid_targets = targets_flat[mask]
        
        if valid_logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # クロスエントロピー損失
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        
        # p_t: 正解クラスの予測確率
        p_t = torch.exp(-ce_loss)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ===============================
# 🆕 新機能2: EMA (Exponential Moving Average)
# ===============================
class EMA:
    """
    モデルパラメータの指数移動平均を保持
    学習の安定化と汎化性能向上に寄与
    
    Args:
        model: 対象モデル
        decay: 減衰率（0.999-0.9999が一般的）
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        """EMAの初期化"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        logger.info(f"✅ EMA registered with decay={self.decay}")
    
    def update(self):
        """パラメータの移動平均を更新"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """EMAパラメータをモデルに適用（評価時用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """元のパラメータを復元（学習再開時用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ===============================
# 🆕 新機能3: 学習率ファインダー
# ===============================
class LRFinder:
    """
    最適な学習率を自動探索
    
    使い方:
        lr_finder = LRFinder(model, optimizer, criterion, device)
        suggested_lr = lr_finder.find(train_loader, min_lr=1e-7, max_lr=1)
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def find(self, train_loader, min_lr=1e-7, max_lr=1, num_iter=100, smooth_f=0.05):
        """
        学習率を徐々に上げながら損失を記録
        
        Args:
            train_loader: 学習データローダー
            min_lr: 最小学習率
            max_lr: 最大学習率
            num_iter: イテレーション数
            smooth_f: 損失の平滑化係数
        
        Returns:
            推奨学習率
        """
        logger.info(f"🔍 LR Finder: Searching optimal learning rate ({min_lr} to {max_lr})...")
        
        # モデルとオプティマイザの状態を保存
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        
        # 学習率スケジューラを設定（指数的に増加）
        mult = (max_lr / min_lr) ** (1 / num_iter)
        lr = min_lr
        self.optimizer.param_groups[0]['lr'] = lr
        
        avg_loss = 0.0
        best_loss = float('inf')
        batch_num = 0
        losses = []
        
        iterator = iter(train_loader)
        
        pbar = tqdm(range(num_iter), desc="LR Finder")
        for iteration in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            batch_num += 1
            
            # 順伝播
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 移動平均で損失を平滑化
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            
            # 最良損失を更新
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # 記録
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)
            
            # 損失が発散したら停止
            if batch_num > 1 and avg_loss > 4 * best_loss:
                logger.info(f"⚠️ Loss diverged, stopping LR finder")
                break
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            # 学習率を更新
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
            
            pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{avg_loss:.4f}")
        
        # モデルとオプティマイザを復元
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        # 最適な学習率を提案（最小損失の1/10の位置）
        min_loss_idx = self.history['loss'].index(min(self.history['loss']))
        suggested_lr = self.history['lr'][max(0, min_loss_idx - len(self.history['lr']) // 10)]
        
        logger.info(f"✅ Suggested learning rate: {suggested_lr:.2e}")
        
        return suggested_lr
    
    def plot(self, save_path="lr_finder_plot.png"):
        """学習率vs損失のグラフを保存"""
        # ディレクトリが存在しない場合は作成
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()  # メモリリーク防止
        logger.info(f"📊 LR Finder plot saved to {save_path}")


# ===============================
# 0. 設定クラス（拡張版）
# ===============================
@dataclass
class TrainingConfig:
    model_name: str
    file_paths: List[str]
    epochs: int = 3
    batch_size: int = 32
    use_amp: bool = True
    max_samples_per_span_file: Optional[int] = None
    val_split: float = 0.05
    save_dir: str = "./models"
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = 2
    max_len: int = 64
    random_seed: int = 42
    tags: Optional[List[str]] = None
    num_workers: int = 4
    accumulation_steps: int = 4
    use_bfloat16: bool = True
    use_compile: bool = False
    scheduler_type: str = 'onecycle'
    warmup_steps: int = 500
    
    # 🆕 新機能のパラメータ
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    use_lr_finder: bool = True
    lr_finder_min: float = 1e-7
    lr_finder_max: float = 1e-2
    lr_finder_num_iter: int = 100
    
    # モックモード設定
    mock_mode: bool = False
    mock_samples: int = 100
    mock_force_cpu: bool = True


# ===============================
# 1. モックデータ生成（元のコードから）
# ===============================
def generate_mock_data(num_samples=100, seed=42):
    """テスト用のモックデータを生成"""
    random.seed(seed)
    
    templates = [
        ("I like {}", "私は{}が好きです"),
        ("This is a {}", "これは{}です"),
        ("How is the {}?", "{}はどうですか?"),
        ("I want to eat {}", "{}を食べたいです"),
        ("The {} is beautiful", "その{}は美しい"),
        ("I can see a {}", "{}が見えます"),
        ("Where is the {}?", "{}はどこですか?"),
        ("I need a {}", "{}が必要です"),
    ]
    
    words = ["apple", "book", "car", "dog", "house", "computer", "phone", "music", 
             "movie", "game", "coffee", "tea", "flower", "bird", "cat", "tree"]
    
    en_list, ja_list = [], []
    
    for i in range(num_samples):
        template_en, template_ja = random.choice(templates)
        word = random.choice(words)
        
        en_list.append(template_en.format(word))
        ja_list.append(template_ja.format(word))
    
    logger.info(f"🎭 Generated {len(en_list)} mock samples")
    return en_list, ja_list


# ===============================
# 2. データセットクラス（元のコードから簡略版）
# ===============================
class TranslationDataset(Dataset):
    def __init__(self, en_list, ja_list, tokenizer, max_len=64):
        self.en_list = en_list
        self.ja_list = ja_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.en_list)
    
    def __getitem__(self, idx):
        en_text = self.en_list[idx]
        ja_text = self.ja_list[idx]
        
        if hasattr(self.tokenizer, 'supported_language_codes'):
            en_text = ">>jap<< " + en_text
        
        inputs = self.tokenizer(
            en_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            ja_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }


# ===============================
# 3. Early Stopping
# ===============================
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ===============================
# 4. セットアップ関数（拡張版）
# ===============================
def setup_training(config: TrainingConfig):
    """学習環境のセットアップ"""
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    
    # デバイス設定
    if config.mock_mode and config.mock_force_cpu:
        device = torch.device("cpu")
        logger.info("🖥️  Using CPU (Mock Mode)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Using device: {device}")
    
    # モデルとトークナイザの読み込み
    logger.info(f"📦 Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    
    # 🆕 Label Smoothingの設定
    if config.use_label_smoothing:
        if hasattr(model.config, 'label_smoothing'):
            model.config.label_smoothing = config.label_smoothing
            logger.info(f"✅ Label Smoothing enabled: {config.label_smoothing}")
    
    model = model.to(device)
    
    return device, tokenizer, model


def create_dataloaders(config: TrainingConfig, tokenizer):
    """データローダーの作成"""
    if config.mock_mode:
        en_list, ja_list = generate_mock_data(config.mock_samples, config.random_seed)
    else:
        # 実際のデータ読み込み（簡略化）
        en_list, ja_list = [], []
        for file_path in config.file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if 'en' in data and 'ja' in data:
                            en_list.append(data['en'])
                            ja_list.append(data['ja'])
            except FileNotFoundError:
                logger.warning(f"⚠️ File not found: {file_path}")
    
    # データセット作成
    full_dataset = TranslationDataset(en_list, ja_list, tokenizer, config.max_len)
    
    # Train/Val分割
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if not config.mock_mode else 0,
        pin_memory=True if not config.mock_mode else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if not config.mock_mode else 0,
        pin_memory=True if not config.mock_mode else False
    )
    
    logger.info(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, config: TrainingConfig, train_loader):
    """オプティマイザとスケジューラの作成"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * config.epochs // config.accumulation_steps
    
    if config.scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
    
    return optimizer, scheduler


# ===============================
# 5. 学習・評価関数（拡張版）
# ===============================
def evaluate_model(model, val_loader, device, criterion=None, use_ema=False, ema=None):
    """
    モデルの評価
    
    Args:
        model: 評価対象モデル
        val_loader: 検証データローダー
        device: デバイス
        criterion: 損失関数（Noneの場合はモデルデフォルト）
        use_ema: EMAモデルを使用するか
        ema: EMAオブジェクト
    """
    model.eval()
    
    # 🆕 EMAパラメータを適用
    if use_ema and ema is not None:
        ema.apply_shadow()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if criterion is not None:
                # Focal Lossを使用
                loss = criterion(outputs.logits, labels)
            else:
                loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    # 🆕 EMAパラメータを元に戻す
    if use_ema and ema is not None:
        ema.restore()
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config, epoch, criterion=None, ema=None):
    """
    1エポックの学習
    
    Args:
        criterion: 損失関数（Focal Loss等）
        ema: EMAオブジェクト
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度学習
        if config.use_amp and device.type == "cuda":
            with autocast(dtype=dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 🆕 Focal Lossまたは通常の損失
                if criterion is not None:
                    loss = criterion(outputs.logits, labels)
                else:
                    loss = outputs.loss
                
                loss = loss / config.accumulation_steps
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if criterion is not None:
                loss = criterion(outputs.logits, labels)
            else:
                loss = outputs.loss
            
            loss = loss / config.accumulation_steps
        
        # 逆伝播
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 勾配蓄積
        if (batch_idx + 1) % config.accumulation_steps == 0:
            if config.gradient_clip > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # 🆕 EMAの更新
            if ema is not None:
                ema.update()
            
            if scheduler:
                scheduler.step()
            
            optimizer.zero_grad()
        
        batch_loss = loss.item() * config.accumulation_steps
        total_loss += batch_loss
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )
    
    return total_loss


# ===============================
# 6. メイン学習関数（拡張版）
# ===============================
def train_model(config: TrainingConfig):
    """
    🆕 新機能搭載版の学習関数
    - Focal Loss
    - Label Smoothing
    - EMA
    - LR Finder
    """
    # 保存ディレクトリを事前に作成
    os.makedirs(config.save_dir, exist_ok=True)
    
    device, tokenizer, model = setup_training(config)
    train_loader, val_loader = create_dataloaders(config, tokenizer)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loader)
    
    # 🆕 LR Finderの実行
    if config.use_lr_finder:
        logger.info("\n" + "="*60)
        logger.info("🔍 Running LR Finder...")
        logger.info("="*60)
        
        # 仮の損失関数（LR Finder用）
        temp_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        lr_finder = LRFinder(model, optimizer, temp_criterion, device)
        
        suggested_lr = lr_finder.find(
            train_loader,
            min_lr=config.lr_finder_min,
            max_lr=config.lr_finder_max,
            num_iter=config.lr_finder_num_iter
        )
        
        # グラフを保存
        lr_finder.plot(os.path.join(config.save_dir, "lr_finder_plot.png"))
        
        # 学習率を更新
        config.learning_rate = suggested_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = suggested_lr
        
        logger.info(f"✅ Learning rate updated to: {suggested_lr:.2e}\n")
    
    # 🆕 Focal Lossの設定
    criterion = None
    if config.use_focal_loss:
        criterion = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )
        logger.info(f"✅ Focal Loss enabled (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
    
    # 🆕 EMAの設定
    ema = None
    if config.use_ema:
        ema = EMA(model, decay=config.ema_decay)
        ema.register()
    
    # 混合精度学習の設定
    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if config.use_bfloat16 and not use_bf16:
        logger.warning("⚠️ BFloat16 requested but not supported. Falling back to FP16.")
    
    scaler = GradScaler() if config.use_amp and device.type == "cuda" and not use_bf16 else None
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=config.patience)
    best_val_loss = float('inf')
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting Training...")
    logger.info("="*60 + "\n")
    
    # 学習ループ
    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            device, config, epoch, criterion, ema
        )
        
        # 検証
        val_loss = evaluate_model(
            model, val_loader, device, criterion,
            use_ema=config.use_ema, ema=ema
        )
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        logger.info(
            f"📊 Epoch {epoch+1}/{config.epochs} -> "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}"
        )
        
        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.save_dir, "best_model")
            
            # 🆕 EMAモデルを保存
            if config.use_ema and ema is not None:
                ema.apply_shadow()
                model.save_pretrained(save_path)
                ema.restore()
            else:
                model.save_pretrained(save_path)
            
            tokenizer.save_pretrained(save_path)
            logger.info(f"⭐ New best model saved to {save_path}!")
        
        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    return model, tokenizer


# ===============================
# 7. 翻訳関数
# ===============================
def translate(model, tokenizer, text, max_length=64, num_beams=4):
    if hasattr(tokenizer, 'supported_language_codes'):
        text = ">>jap<< " + text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def batch_translate(model, tokenizer, texts, batch_size=8, max_length=64, num_beams=4):
    device = next(model.parameters()).device
    if hasattr(tokenizer, 'supported_language_codes'):
        texts = [">>jap<< " + t for t in texts]
    translations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        translations.extend([tokenizer.decode(o, skip_special_tokens=True) for o in outputs])
    return translations


# ===============================
# 8. モックテスト
# ===============================
def quick_mock_test():
    logger.info("\n" + "="*60)
    logger.info("🧪 ENHANCED MOCK TEST - Testing New Features")
    logger.info("="*60 + "\n")
    
    config = TrainingConfig(
        model_name="Helsinki-NLP/opus-mt-en-jap",
        file_paths=[],
        epochs=2,
        batch_size=4,
        mock_mode=True,
        mock_samples=50,
        mock_force_cpu=True,
        num_workers=0,
        accumulation_steps=2,
        use_amp=False,
        use_bfloat16=False,
        save_dir="./mock_output",
        use_focal_loss=True,
        use_label_smoothing=True,
        use_ema=True,
        use_lr_finder=True,
        lr_finder_num_iter=20  # モック用に短縮
    )
    
    try:
        model, tokenizer = train_model(config)
        
        # 翻訳テスト
        logger.info("\n🧪 Testing translation...")
        test_sentences = ["I like apples.", "How are you?", "This is a test."]
        results = batch_translate(model, tokenizer, test_sentences)
        
        for en, ja in zip(test_sentences, results):
            logger.info(f"  EN: {en} -> JA: {ja}")
        
        logger.info("\n✅ ENHANCED MOCK TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ ENHANCED MOCK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===============================
# 実行例
# ===============================
if __name__ == "__main__":
    import sys
    
    if "--mock" in sys.argv or os.getenv("MOCK_MODE") == "1":
        # モックモードで実行
        quick_mock_test()
    
    else:
        # 通常モード（実際の学習）
        files = [
            "./../data/sepalated_dataset.jsonl",
            "./../data/OpenSubtitles_sample_40000.jsonl",
            "./../data/TED_sample_40000.jsonl",
            "./../data/Tatoeba_sample_40000.jsonl",
            "./../data/all_outenjp.jsonl"
        ]
        
        config = TrainingConfig(
            model_name="Helsinki-NLP/opus-mt-en-jap",
            file_paths=files,
            epochs=3,
            batch_size=16,
            max_samples_per_span_file=40000,
            save_dir="./models/translation_model_enhanced",
            random_seed=42,
            num_workers=4,
            accumulation_steps=4,
            use_bfloat16=True,
            scheduler_type='onecycle',
            # 🆕 新機能を有効化
            use_focal_loss=True,
            focal_alpha=0.25,
            focal_gamma=2.0,
            use_label_smoothing=True,
            label_smoothing=0.1,
            use_ema=True,
            ema_decay=0.9999,
            use_lr_finder=True,
            lr_finder_min=1e-7,
            lr_finder_max=1e-2,
            lr_finder_num_iter=100
        )
        
        model, tokenizer = train_model(config)
        
        # 翻訳テスト
        test_sentences = [
            "I like apples.",
            "How are you?",
            "Machine learning is fun.",
            "I couldn't speak English well."
        ]
        results = batch_translate(model, tokenizer, test_sentences)
        for en, ja in zip(test_sentences, results):
            print(f"EN: {en} -> JA: {ja}")
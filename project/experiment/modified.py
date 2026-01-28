import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
from torch.utils.data import Dataset
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
    """難しいサンプルにより注目するFocal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets, ignore_index=-100):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        mask = targets_flat != ignore_index
        valid_logits = logits_flat[mask]
        valid_targets = targets_flat[mask]
        
        if valid_logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ===============================
# 🆕 新機能2: EMA
# ===============================
class EMA:
    """モデルパラメータの指数移動平均"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        logger.info(f"✅ EMA registered with decay={self.decay}")
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ===============================
# 🆕 新機能3: 学習率ファインダー
# ===============================
class LRFinder:
    """最適な学習率を自動探索"""
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def find(self, train_loader, min_lr=1e-7, max_lr=1, num_iter=100, smooth_f=0.05):
        logger.info(f"🔍 LR Finder: Searching optimal learning rate ({min_lr} to {max_lr})...")
        
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        
        mult = (max_lr / min_lr) ** (1 / num_iter)
        lr = min_lr
        self.optimizer.param_groups[0]['lr'] = lr
        
        avg_loss = 0.0
        best_loss = float('inf')
        batch_num = 0
        
        iterator = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Finder")
        
        for iteration in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            batch_num += 1
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
            
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)
            
            if batch_num > 1 and avg_loss > 4 * best_loss:
                logger.info(f"⚠️ Loss diverged, stopping LR finder")
                break
            
            loss.backward()
            self.optimizer.step()
            
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
            pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{avg_loss:.4f}")
        
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        min_loss_idx = self.history['loss'].index(min(self.history['loss']))
        suggested_lr = self.history['lr'][max(0, min_loss_idx - len(self.history['lr']) // 10)]
        
        logger.info(f"✅ Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr
    
    def plot(self, save_path="lr_finder_plot.png"):
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
        plt.close()
        logger.info(f"📊 LR Finder plot saved to {save_path}")


# ===============================
# 0. 設定クラス
# ===============================

@dataclass
class TrainingConfig:
    model_name: str
    file_paths: List[str]
    file_types: List[int] = field(default_factory=list)  # 0=Span,1=ByWork,2=Practical(ByWork+Chunk)
    epochs: int = 3
    batch_size: int = 32
    use_amp: bool = True
    max_samples_per_span_file: Optional[int] = None
    val_split: float = 0.05
    save_dir: str = "./models"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
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
    
    # new
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

    # phase control: if None, defaults computed below
    phase_epochs: Optional[List[int]] = None  # [phase1_epochs, phase2_epochs, phase3_epochs]

    # mock mode...
    mock_mode: bool = False
    mock_samples: int = 100
    mock_force_cpu: bool = True



# ===============================
# 1. データ読み込み（元のコードから）
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


def add_tag_if_needed(text, tag):
    """タグを追加するヘルパー関数"""
    return f"{tag} {text}" if tag else text


def is_chunk_delimiter(text):
    """チャンク区切り行かどうかを判定"""
    return "%%%%%%%%THISWORKENDSHERE%%%%%%%%" in text or "%%%%%%%%この作品ここまで%%%%%%%%" in text


def load_single_dataset_streaming(file_path, max_samples=None, random_seed=42, tag=None):
    """メモリ効率的にJSONLファイルを読み込み（区切り行を除外）"""
    en_list, ja_list = [], []
    error_count = 0
    skipped_delimiters = 0
    
    logger.info(f"📖 Loading {file_path} ...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if max_samples:
                total_lines = sum(1 for _ in f)
                f.seek(0)
                
                if total_lines > max_samples:
                    logger.info(f"  ⚡ Sampling {max_samples} from {total_lines} lines")
                    random.seed(random_seed)
                    selected_indices = set(random.sample(range(total_lines), max_samples))
                    
                    for idx, line in enumerate(tqdm(f, total=total_lines,
                                                    desc=f"Reading {os.path.basename(file_path)}",
                                                    unit=" lines")):
                        if idx in selected_indices:
                            try:
                                data = json.loads(line)
                                en, ja = data.get("en"), data.get("ja")
                                
                                # 🆕 区切り行をスキップ
                                if en and is_chunk_delimiter(en):
                                    skipped_delimiters += 1
                                    continue
                                
                                if en and ja and len(en.strip()) > 0 and len(ja.strip()) > 0:
                                    en_list.append(add_tag_if_needed(en, tag))
                                    ja_list.append(ja)
                            except json.JSONDecodeError:
                                error_count += 1
                else:
                    f.seek(0)
                    for line in tqdm(f, total=total_lines,
                                   desc=f"Reading {os.path.basename(file_path)}",
                                   unit=" lines"):
                        try:
                            data = json.loads(line)
                            en, ja = data.get("en"), data.get("ja")
                            
                            # 🆕 区切り行をスキップ
                            if en and is_chunk_delimiter(en):
                                skipped_delimiters += 1
                                continue
                            
                            if en and ja and len(en.strip()) > 0 and len(ja.strip()) > 0:
                                en_list.append(add_tag_if_needed(en, tag))
                                ja_list.append(ja)
                        except json.JSONDecodeError:
                            error_count += 1
            else:
                for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}", unit=" lines"):
                    try:
                        data = json.loads(line)
                        en, ja = data.get("en"), data.get("ja")
                        
                        # 🆕 区切り行をスキップ
                        if en and is_chunk_delimiter(en):
                            skipped_delimiters += 1
                            continue
                        
                        if en and ja and len(en.strip()) > 0 and len(ja.strip()) > 0:
                            en_list.append(add_tag_if_needed(en, tag))
                            ja_list.append(ja)
                    except json.JSONDecodeError:
                        error_count += 1
    
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return [], []
    except Exception as e:
        logger.error(f"❌ Unexpected error loading {file_path}: {e}")
        return [], []
    
    if error_count > 0:
        logger.warning(f"  ⚠️  Skipped {error_count} invalid lines")
    if skipped_delimiters > 0:
        logger.info(f"  🔖 Skipped {skipped_delimiters} chunk delimiter lines")
    
    logger.info(f"  ✅ Loaded {len(en_list)} pairs from {os.path.basename(file_path)}")
    return en_list, ja_list


def load_chunks_from_file(file_path, tag=None):
    """
    🆕 ByWorkファイルからチャンク単位でデータを読み込み
    
    区切り行で分割し、各チャンク内の全文を結合した大きな翻訳ペアを作成
    
    Returns:
        chunk_en_list: [チャンク1の全文結合, チャンク2の全文結合, ...]
        chunk_ja_list: [チャンク1の全文結合, チャンク2の全文結合, ...]
    """
    chunk_en_list = []
    chunk_ja_list = []
    
    current_en_chunk = []
    current_ja_chunk = []
    
    logger.info(f"📚 Loading chunks from {file_path} ...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading chunks {os.path.basename(file_path)}", unit=" lines"):
                try:
                    data = json.loads(line)
                    en, ja = data.get("en"), data.get("ja")
                    
                    if not en or not ja:
                        continue
                    
                    # 区切り行を検出
                    if is_chunk_delimiter(en):
                        # 現在のチャンクを保存
                        if current_en_chunk and current_ja_chunk:
                            # チャンク内の全文を結合
                            combined_en = " ".join(current_en_chunk)
                            combined_ja = " ".join(current_ja_chunk)
                            
                            chunk_en_list.append(add_tag_if_needed(combined_en, tag))
                            chunk_ja_list.append(combined_ja)
                        
                        # 次のチャンク用にリセット
                        current_en_chunk = []
                        current_ja_chunk = []
                    else:
                        # 通常の行はチャンクに追加
                        if len(en.strip()) > 0 and len(ja.strip()) > 0:
                            current_en_chunk.append(en)
                            current_ja_chunk.append(ja)
                
                except json.JSONDecodeError:
                    continue
        
        # 最後のチャンクを保存
        if current_en_chunk and current_ja_chunk:
            combined_en = " ".join(current_en_chunk)
            combined_ja = " ".join(current_ja_chunk)
            chunk_en_list.append(add_tag_if_needed(combined_en, tag))
            chunk_ja_list.append(combined_ja)
    
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return [], []
    except Exception as e:
        logger.error(f"❌ Unexpected error loading chunks from {file_path}: {e}")
        return [], []
    
    logger.info(f"  ✅ Loaded {len(chunk_en_list)} chunks from {os.path.basename(file_path)}")
    return chunk_en_list, chunk_ja_list


def load_datasets_balanced(file_paths, file_types, max_samples_per_type=None, random_seed=42, tags=None):
    if file_types is None:
        file_types = [0] * len(file_paths)
    """
    ByWork系とRandomSpan系を分けてサンプリング
    
    🆕 ByWorkファイルは:
    - 行単位データ（区切り行を除外）
    - チャンク単位データ（区切り行で分割して結合）
    の両方を返す
    """
    bywork_files = []
    bywork_chunk_files = []  # 🆕 チャンク単位データ
    span_files = []
    
    if tags is None:
        tags = [None] * len(file_paths)
    
    for fp, tag, ftype in zip(file_paths, tags, file_types):
        # ftype: 0 => RandomSpan, 1 => ByWork (line+chunk), 2 => Practical (ByWork + chunk, treated like 1 but flagged)
        if ftype == 0:
            logger.info(f"\n🎲 [SPAN-LEVEL] {fp}")
            en_list, ja_list = load_single_dataset_streaming(fp, max_samples=max_samples_per_type, random_seed=random_seed, tag=tag)
            span_files.append((fp, en_list, ja_list))
        elif ftype in (1, 2):
            logger.info(f"\n🎯 [WORK-LEVEL] {fp} (type={ftype})")
            # ftype=1 (文学一般) の場合は max_samples を適用し、ftype=2 (本命) は全件ロードする
            sample_limit = max_samples_per_type if ftype == 1 else None
            en_list, ja_list = load_single_dataset_streaming(fp, max_samples=sample_limit, random_seed=random_seed, tag=tag)
            bywork_files.append((fp, en_list, ja_list))
            # チャンク読み込みは元のファイルから行う（ここはそのまま）
            chunk_en_list, chunk_ja_list = load_chunks_from_file(fp, tag=tag)
            bywork_chunk_files.append((fp, chunk_en_list, chunk_ja_list))
        else:
            raise ValueError(f"Unknown file_types value: {ftype} for {fp}")

    
    # サマリー表示
    logger.info("\n" + "="*60)
    logger.info("📊 LOADING SUMMARY")
    logger.info("="*60)
    
    total_bywork = sum(len(data[1]) for data in bywork_files)
    logger.info(f"ByWork datasets (line-by-line): {len(bywork_files)} files, {total_bywork:,} pairs total")
    for fp, en_list, _ in bywork_files:
        logger.info(f"  - {os.path.basename(fp)}: {len(en_list):,} pairs")
    
    total_bywork_chunks = sum(len(data[1]) for data in bywork_chunk_files)
    logger.info(f"\n🆕 ByWork datasets (chunk-level): {len(bywork_chunk_files)} files, {total_bywork_chunks:,} chunks total")
    for fp, en_list, _ in bywork_chunk_files:
        logger.info(f"  - {os.path.basename(fp)}: {len(en_list):,} chunks")
    
    total_span = sum(len(data[1]) for data in span_files)
    logger.info(f"\nRandomSpan datasets: {len(span_files)} files, {total_span:,} pairs total")
    for fp, en_list, _ in span_files:
        logger.info(f"  - {os.path.basename(fp)}: {len(en_list):,} pairs")
    
    logger.info(f"\n🎉 GRAND TOTAL: {total_bywork + total_bywork_chunks + total_span:,} samples")
    logger.info("="*60 + "\n")
    
    return bywork_files, bywork_chunk_files, span_files

# ===============================
# 2. Datasetクラス（元のコードから）
# ===============================
def build_randomspan_collator(tokenizer, label_smoothing=0.1):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        label_pad_token_id=-100,
        padding=True,
    )


class TranslationDatasetRandomSpan(Dataset):
    """RandomSpan系データ用Dataset"""
    def __init__(self, en_texts, ja_texts, tokenizer, max_len=64, multi_prob=0.5):
        self.en_texts = en_texts
        self.ja_texts = ja_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multi_prob = multi_prob

    def __len__(self):
        return len(self.en_texts)

    def set_multi_prob(self, prob):
        self.multi_prob = prob

    def __getitem__(self, idx):
        en_text = self.en_texts[idx]
        ja_text = self.ja_texts[idx]
        
        if hasattr(self.tokenizer, 'supported_language_codes'):
            en_text = ">>jap<< " + en_text

        # マルチセンテンス化
        if random.random() < self.multi_prob and idx + 1 < len(self.en_texts):
            en_text = en_text + " " + self.en_texts[idx + 1]
            ja_text = ja_text + " " + self.ja_texts[idx + 1]

        inputs = self.tokenizer(en_text, max_length=self.max_len, truncation=True, padding=False)
        labels = self.tokenizer(ja_text, max_length=self.max_len, truncation=True, padding=False)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }


class TranslationDatasetByWork(Dataset):
    def __init__(self, en_texts, ja_texts, tokenizer, max_len=64):
        self.en_texts = en_texts
        self.ja_texts = ja_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.en_texts)

    def __getitem__(self, idx):
        en_text, ja_text = self.en_texts[idx], self.ja_texts[idx]
        if hasattr(self.tokenizer, 'supported_language_codes'):
            en_text = ">>jap<< " + en_text
        
        # padding=False にして、生のリボンのまま返す
        inputs = self.tokenizer(en_text, max_length=self.max_len, truncation=True, padding=False)
        labels = self.tokenizer(ja_text, max_length=self.max_len, truncation=True, padding=False)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

class TranslationDatasetByWorkChunk(Dataset):
    def __init__(self, en_chunks, ja_chunks, tokenizer, max_len=512):
        self.en_chunks = en_chunks
        self.ja_chunks = ja_chunks
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.en_chunks)

    def __getitem__(self, idx):
        en_text, ja_text = self.en_chunks[idx], self.ja_chunks[idx]
        if hasattr(self.tokenizer, 'supported_language_codes'):
            en_text = ">>jap<< " + en_text

        inputs = self.tokenizer(en_text, max_length=self.max_len, truncation=True, padding=False)
        labels = self.tokenizer(ja_text, max_length=self.max_len, truncation=True, padding=False)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

# ===============================
# 3. Early Stopping
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
            if self.counter >= self.patience:
                self.early_stop = True


# ===============================
# 4. 学習補助関数
# ===============================
def freeze_encoder_layers(model, ratio=0.5):
    """Encoderの一部レイヤーを凍結"""
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
    elif hasattr(model, 'encoder'):
        encoder = model.encoder
    else:
        logger.warning("⚠️ Could not find encoder to freeze")
        return

    if hasattr(encoder, 'layers'):
        total_layers = len(encoder.layers)
        freeze_count = int(total_layers * ratio)
        for i, layer in enumerate(encoder.layers):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info(f"🔒 Frozen {freeze_count}/{total_layers} encoder layers")


def evaluate_model(model, val_loader, device, criterion=None, use_ema=False, ema=None):
    """検証ループ"""
    model.eval()
    
    if use_ema and ema is not None:
        ema.apply_shadow()
    
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            if criterion is not None:
                loss = criterion(outputs.logits, labels)
            else:
                loss = outputs.loss
            
            total_loss += loss.item()
    
    if use_ema and ema is not None:
        ema.restore()
    
    model.train()
    return total_loss / len(val_loader)


# ===============================
# 5. セットアップ関数
# ===============================
def setup_training(config: TrainingConfig):
    """デバイス、モデル、トークナイザーの初期化"""
    if config.mock_mode:
        logger.info("🎭 " + "="*60)
        logger.info("🎭 MOCK MODE ENABLED - Running with synthetic data")
        logger.info("🎭 " + "="*60)
        
        if config.mock_force_cpu:
            device = torch.device("cpu")
            logger.info("🎭 Forcing CPU for mock mode")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"🔧 Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    # 【修正】torch_dtype=torch.float16 を削除し、デフォルト(FP32)で読み込む
    # GradScalerを使う場合、重みはFP32である必要があります。
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        # torch_dtype=torch.float16,  <-- 削除
        use_safetensors=True
    ).to(device)
    
    # 🆕 Label Smoothingの設定
    if config.use_label_smoothing:
        if hasattr(model.config, 'label_smoothing'):
            model.config.label_smoothing = config.label_smoothing
            logger.info(f"✅ Label Smoothing enabled: {config.label_smoothing}")
    
    if config.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("⚡ Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}")
    
    return device, tokenizer, model


def create_dataloaders(config: TrainingConfig, tokenizer):
    """
    データローダーの作成（RandomSpan, ByWork, Practical を区別）
    戻り値: (train_loaders, val_loader, dataset, loaders_map)
    loaders_map keys: "span", "bywork", "chunk", "practical_line", "practical_chunk"
    """
    # データ読み込み
    if config.mock_mode:
        logger.info(f"🎭 Generating {config.mock_samples} mock samples...")
        en_list, ja_list = generate_mock_data(config.mock_samples, seed=config.random_seed)
        span_files = [("mock_data", en_list, ja_list)]
        bywork_files = []
        bywork_chunk_files = []
        practical_line_files = []
        practical_chunk_files = []
    else:
        # file_types を渡す
        if not hasattr(config, "file_types") or not config.file_types:
            raise ValueError("❌ config.file_types must be provided (list of ints matching file_paths)")
        bywork_files, bywork_chunk_files, span_files = load_datasets_balanced(
            config.file_paths,
            config.file_types,
            max_samples_per_type=config.max_samples_per_span_file,
            random_seed=config.random_seed,
            tags=config.tags
        )

    # ファイルタイプマップ（file_paths に基づく簡易参照）
    file_type_map = {}
    if hasattr(config, "file_types") and config.file_paths:
        for fp, ftype in zip(config.file_paths, config.file_types):
            file_type_map[fp] = ftype

    # データセット作成：non-practical は all_datasets に入れ、practical は別に貯める
    all_datasets = []
    practical_line_datasets = []
    practical_chunk_datasets = []
    bywork_chunk_datasets_nonpractical = []

    # RandomSpan 系（基本は all_datasets に入れる）
    for fp, en_list, ja_list in span_files:
        ds = TranslationDatasetRandomSpan(en_list, ja_list, tokenizer, max_len=config.max_len)
        all_datasets.append(ds)

    # ByWork 行単位: file_type_map で practical 判定
    for fp, en_list, ja_list in bywork_files:
        ds = TranslationDatasetByWork(en_list, ja_list, tokenizer, max_len=config.max_len)
        if file_type_map.get(fp) == 2:
            practical_line_datasets.append((fp, ds))
        else:
            all_datasets.append(ds)

    # ByWork チャンク単位
    for fp, en_list, ja_list in bywork_chunk_files:
        ds = TranslationDatasetByWorkChunk(en_list, ja_list, tokenizer, max_len=512)
        if file_type_map.get(fp) == 2:
            practical_chunk_datasets.append((fp, ds))
        else:
            bywork_chunk_datasets_nonpractical.append((fp, ds))

    # safety: all_datasets が空の場合でも ConcatDataset を作れるようにする
    if all_datasets:
        dataset = ConcatDataset(all_datasets)
    else:
        # 空のダミーデータセットを用意（random_split 等で問題が起きないように）
        empty_ds = TranslationDatasetRandomSpan([], [], tokenizer, max_len=config.max_len)
        dataset = ConcatDataset([empty_ds])

    # Train/Val 分割（non-practical の合成 dataset に対して）
    total_len = len(dataset)
    val_size = int(total_len * config.val_split)
    train_size = total_len - val_size

    # random_split は Dataset を受け取る
    if total_len > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = Subset(dataset, [])
        val_dataset = Subset(dataset, [])

    logger.info(f"📊 Train: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

    # 元のコードに倣って RandomSpan / ByWork のインデックスを計算
    # ==========================================
    # 🛠️ 修正・デバッグ版インデックス振り分け処理
    # ==========================================
    logger.info("⚙️ Starting index filtering... (This might take a moment)")

    span_indices = []
    bywork_indices = []
    offset = 0
    
    # データセットの構成をスキャン
    for i, ds in enumerate(dataset.datasets):
        ds_len = len(ds)
        # 型判定
        if isinstance(ds, TranslationDatasetRandomSpan):
            span_indices.extend(range(offset, offset + ds_len))
        elif isinstance(ds, TranslationDatasetByWork):
            bywork_indices.extend(range(offset, offset + ds_len))
        # Chunkなど他の型があればここに追加
        elif isinstance(ds, TranslationDatasetByWorkChunk):
            # ChunkはPracticalでなければByWork扱いにするか、除外するかなど
            # 今回のロジックでは train_bywork に含める運用であれば以下
            bywork_indices.extend(range(offset, offset + ds_len))
            
        offset += ds_len

    logger.info(f"  - Total Span indices: {len(span_indices)}")
    logger.info(f"  - Total ByWork indices: {len(bywork_indices)}")

    # ⚠️ 【重要】高速化のため必ず set に変換する (O(N) -> O(1))
    span_index_set = set(span_indices)
    bywork_index_set = set(bywork_indices)
    
    logger.info("  - Converted to sets for fast lookup.")

    # Trainデータセット内のインデックスを振り分け
    # train_dataset.indices は Subset が持つ元のインデックスリスト
    if hasattr(train_dataset, 'indices'):
        current_indices = train_dataset.indices
    else:
        # random_splitされなかった場合など
        current_indices = list(range(len(train_dataset)))

    train_span_indices = [i for i in current_indices if i in span_index_set]
    train_bywork_indices = [i for i in current_indices if i in bywork_index_set]

    logger.info(f"✅ Filtering completed. Train Span: {len(train_span_indices)}, Train ByWork: {len(train_bywork_indices)}")
    # ==========================================[]

    train_span = Subset(dataset, train_span_indices)
    train_bywork = Subset(dataset, train_bywork_indices)

    # non-practical chunk データセットをまとめて Train/Val 分割（もしあれば）
    if bywork_chunk_datasets_nonpractical:
        chunk_dataset = ConcatDataset([ds for _, ds in bywork_chunk_datasets_nonpractical])
        chunk_val_size = int(len(chunk_dataset) * config.val_split)
        chunk_train_size = len(chunk_dataset) - chunk_val_size
        train_chunk_dataset, val_chunk_dataset = random_split(chunk_dataset, [chunk_train_size, chunk_val_size])
        logger.info(f"📚 Chunk data: Train {len(train_chunk_dataset):,}, Val {len(val_chunk_dataset):,}")
    else:
        train_chunk_dataset = None
        val_chunk_dataset = None

    # ==========================================
    # 🛠️ Practicalデータの分割とアップサンプリング
    # ==========================================
    # Practical (Line)
    if practical_line_datasets:
        practical_line_dataset = ConcatDataset([ds for _, ds in practical_line_datasets])
        p_line_val_size = max(1, int(len(practical_line_dataset) * config.val_split)) # 最低1件は確保
        p_line_train_size = len(practical_line_dataset) - p_line_val_size
        
        train_practical_line_base, val_practical_line = random_split(
            practical_line_dataset, 
            [p_line_train_size, p_line_val_size],
            generator=torch.Generator().manual_seed(config.random_seed)
        )
        
        # 【修正】データが少なすぎるため、Trainデータだけ20倍に複製（Upsampling）する
        # これにより 300件 -> 6000件 相当になり、1epochでしっかり学習できる
        upsample_factor = 5
        logger.info(f"⚡ Upsampling Practical (Line) train data by {upsample_factor}x")
        train_practical_line = ConcatDataset([train_practical_line_base] * upsample_factor)
        
    else:
        train_practical_line = None
        val_practical_line = None

    # Practical (Chunk)
    if practical_chunk_datasets:
        practical_chunk_dataset = ConcatDataset([ds for _, ds in practical_chunk_datasets])
        p_chunk_val_size = max(1, int(len(practical_chunk_dataset) * config.val_split))
        p_chunk_train_size = len(practical_chunk_dataset) - p_chunk_val_size
        
        train_practical_chunk_base, val_practical_chunk = random_split(
            practical_chunk_dataset, 
            [p_chunk_train_size, p_chunk_val_size],
            generator=torch.Generator().manual_seed(config.random_seed)
        )
        
        # 【修正】Chunkデータも同様に20倍にする
        upsample_factor = 20
        logger.info(f"⚡ Upsampling Practical (Chunk) train data by {upsample_factor}x")
        train_practical_chunk = ConcatDataset([train_practical_chunk_base] * upsample_factor)
        
    else:
        train_practical_chunk = None
        val_practical_chunk = None

    # Collator（RandomSpan 用）
    span_collator = build_randomspan_collator(tokenizer, label_smoothing=config.label_smoothing)

    # DataLoader の共通引数
    loader_args = {
        'num_workers': 0 if config.mock_mode else config.num_workers,
        'pin_memory': False if config.mock_mode else True,
        'persistent_workers': False
    }

    actual_batch_size = config.batch_size
    # 文学データ(ByWork)のバッチサイズを 4分の1 から 2分の1 に引き上げ（4 -> 16相当）
    bywork_batch_size = max(1, config.batch_size // 2) 
    chunk_batch_size = max(1, config.batch_size // 8)

    train_loaders = []

    # RandomSpan ローダー（mixed training の中核）
    if len(train_span) > 0:
        train_loader_span = DataLoader(
            train_span,
            batch_size=actual_batch_size,
            shuffle=True,
            collate_fn=span_collator,
            **loader_args
        )
        train_loaders.append(train_loader_span)
        logger.info(f"✅ RandomSpan loader: {len(train_span)} samples, batch_size={actual_batch_size}")
    else:
        train_loader_span = None

    # ByWork 行単位ローダー（non-practical）
    if len(train_bywork) > 0:
        train_loader_bywork = DataLoader(
            train_bywork,
            batch_size=bywork_batch_size,
            shuffle=True,
            collate_fn=span_collator,
            **loader_args
        )
        train_loaders.append(train_loader_bywork)
        logger.info(f"✅ ByWork (line) loader: {len(train_bywork)} samples, batch_size={bywork_batch_size}")
    else:
        train_loader_bywork = None

    # Non-practical chunk ローダー
    if train_chunk_dataset is not None and len(train_chunk_dataset) > 0:
        train_loader_chunk = DataLoader(
            train_chunk_dataset,
            batch_size=chunk_batch_size,
            shuffle=True,
            collate_fn=span_collator,
            **loader_args
        )
        train_loaders.append(train_loader_chunk)
        logger.info(f"✅ ByWork (chunk) loader: {len(train_chunk_dataset)} chunks, batch_size={chunk_batch_size}")
    else:
        train_loader_chunk = None

    # Practical 専用ローダー（line / chunk）
    if train_practical_line is not None and len(train_practical_line) > 0:
        train_loader_practical_line = DataLoader(
            train_practical_line,
            batch_size=bywork_batch_size,
            shuffle=True,
            collate_fn=span_collator,
            **loader_args
        )
        logger.info(f"✅ Practical (line) loader: {len(train_practical_line)} samples, batch_size={bywork_batch_size}")
    else:
        train_loader_practical_line = None

    if train_practical_chunk is not None and len(train_practical_chunk) > 0:
        train_loader_practical_chunk = DataLoader(
            train_practical_chunk,
            batch_size=chunk_batch_size,
            shuffle=True,
            collate_fn=span_collator,
            **loader_args
        )
        logger.info(f"✅ Practical (chunk) loader: {len(train_practical_chunk)} chunks, batch_size={chunk_batch_size}")
    else:
        train_loader_practical_chunk = None

    if not train_loaders and train_loader_practical_line is None and train_loader_practical_chunk is None:
        raise ValueError("❌ No training loaders created!")

    # Validation ローダー（non-practical val と practical val をまとめる）
    combined_val_datasets = []
    if len(val_dataset) > 0:
        combined_val_datasets.append(val_dataset)
    if val_chunk_dataset is not None and len(val_chunk_dataset) > 0:
        combined_val_datasets.append(val_chunk_dataset)
    if val_practical_line is not None and len(val_practical_line) > 0:
        combined_val_datasets.append(val_practical_line)
    if val_practical_chunk is not None and len(val_practical_chunk) > 0:
        combined_val_datasets.append(val_practical_chunk)

    if combined_val_datasets:
        combined_val = ConcatDataset(combined_val_datasets) if len(combined_val_datasets) > 1 else combined_val_datasets[0]
        val_loader = DataLoader(
            combined_val,
            batch_size=actual_batch_size * 2,
            shuffle=False,
            collate_fn=span_collator,
            **loader_args
        )
    else:
        # 空の validation
        empty_val_ds = TranslationDatasetRandomSpan([], [], tokenizer, max_len=config.max_len)
        val_loader = DataLoader(
            empty_val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=span_collator,
            **loader_args
        )

    # loaders_map を作って返却（phase 制御で使いやすいように）
    loaders_map = {
        "span": train_loader_span,
        "bywork": train_loader_bywork,
        "chunk": train_loader_chunk,
        "practical_line": train_loader_practical_line,
        "practical_chunk": train_loader_practical_chunk
    }

    return train_loaders, val_loader, dataset, loaders_map


def create_optimizer_and_scheduler(model, config: TrainingConfig, train_loaders):
    """オプティマイザとスケジューラの作成（weight_decay を尊重し、bias/LayerNorm には適用しない）"""
    # パラメータグループ：bias と LayerNorm の weight decay を 0 にする
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 名前に 'bias' を含むか LayerNorm / layer_norm / LayerNorm を含む場合は no_decay
        if name.endswith(".bias") or "layernorm" in name.lower() or "layer_norm" in name.lower() or "ln_" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": float(getattr(config, "weight_decay", 0.0))},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate)

    # total steps 計算（train_loaders はリストの想定）
    total_steps_per_epoch = sum(len(loader) for loader in train_loaders if loader is not None)
    total_steps = 100*total_steps_per_epoch * config.epochs
    if config.scheduler_type == 'onecycle':
        # max_lr を config.learning_rate * 10 としている既存の方針を尊重
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate * 10,
            total_steps=max(1, total_steps),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        logger.info(f"📈 OneCycleLR scheduler initialized (total_steps={total_steps})")
    elif config.scheduler_type == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max(1, total_steps)
        )
        logger.info(f"📈 Linear warmup scheduler initialized (warmup_steps={config.warmup_steps}, total_steps={total_steps})")
    else:
        scheduler = None

    return optimizer, scheduler



# ===============================
# 6. 学習エポック関数
# ===============================
def train_epoch(model, loaders, optimizer, scheduler, scaler, device, config, epoch, criterion=None, ema=None):
    """1エポック分の学習処理"""
    model.train()
    total_loss = 0
    
    # BF16が使えるか確認
    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    # AMPを使うか確認（BF16が使えなくても、FP16でAMPをする場合がある）
    use_amp = config.use_amp and device.type == "cuda"

    for loader in loaders:
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx % config.accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss_divisor = config.accumulation_steps
            
            
            if use_bf16:
                with autocast( dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = (criterion(outputs.logits, labels) if criterion else outputs.loss) / loss_divisor
            elif use_amp:
                
                with autocast( dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = (criterion(outputs.logits, labels) if criterion else outputs.loss) / loss_divisor
            else:
                # AMPなし（FP32）
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = (criterion(outputs.logits, labels) if criterion else outputs.loss) / loss_divisor

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

            batch_loss = loss.item() * loss_divisor
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            
    return total_loss


# ===============================
# 7. メイン学習関数
# ===============================
def train_model(config: TrainingConfig):
    """3-Phase 対応版の学習関数。戻り値は (model, tokenizer)."""
    os.makedirs(config.save_dir, exist_ok=True)

    device, tokenizer, model = setup_training(config)
    train_loaders, val_loader, dataset, loaders_map = create_dataloaders(config, tokenizer)
    # optimizer/scheduler は train_loaders（非-practical の混合ローダ群）を基に作成
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loaders)

    # --- LR Finder: span ローダーがあればそれを優先して使う ---
    span_loader_for_lr = loaders_map.get("span") if isinstance(loaders_map, dict) else None
    if not span_loader_for_lr and train_loaders:
        span_loader_for_lr = train_loaders[0]

    if config.use_lr_finder and span_loader_for_lr is not None:
        logger.info("\n" + "="*60)
        logger.info("🔍 Running LR Finder...")
        logger.info("="*60)
        temp_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        lr_finder = LRFinder(model, optimizer, temp_criterion, device)
        suggested_lr = lr_finder.find(
            span_loader_for_lr,
            min_lr=config.lr_finder_min,
            max_lr=config.lr_finder_max,
            num_iter=config.lr_finder_num_iter
        )
        lr_finder.plot(os.path.join(config.save_dir, "lr_finder_plot.png"))
        config.learning_rate = suggested_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = suggested_lr
        logger.info(f"✅ Learning rate updated to: {suggested_lr:.2e}\n")

    # --- criterion, EMA, scaler の初期化 ---
    criterion = None
    if config.use_focal_loss:
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        logger.info(f"✅ Focal Loss enabled (alpha={config.focal_alpha}, gamma={config.focal_gamma})")

    ema = None
    if config.use_ema:
        ema = EMA(model, decay=config.ema_decay)
        ema.register()

    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if config.use_bfloat16 and not use_bf16:
        logger.warning("⚠️ BFloat16 requested but not supported. Falling back to FP16.")

    scaler = GradScaler() if (config.use_amp and device.type == "cuda" and not use_bf16) else None

    early_stopping = EarlyStopping(patience=config.patience)
    best_val_loss = float('inf')

    # freeze encoder 初期化（ratio はハードコード済みの 0.5）
    freeze_encoder_layers(model, ratio=0.5)
    logger.info("🔒 Encoder layers partially frozen (ratio=0.5)")

    logger.info("\n" + "="*60)
    logger.info("🚀 Starting Training (3-Phase enabled)...")
    logger.info("="*60 + "\n")

    # --- phase_epochs の決定 ---
    if config.phase_epochs is None:
        # practical がいるかどうかでデフォルト配分を決める
        has_practical = hasattr(config, "file_types") and any(ft == 2 for ft in config.file_types)
        if has_practical and config.epochs >= 3:
            phase_epochs = [max(1, config.epochs - 2), 1, 1]
        elif config.epochs == 2:
            phase_epochs = [1, 1, 0]
        else:
            phase_epochs = [config.epochs, 0, 0]
    else:
        if len(config.phase_epochs) != 3 or sum(config.phase_epochs) != config.epochs:
            raise ValueError("phase_epochs must be length-3 and sum to config.epochs")
        phase_epochs = config.phase_epochs

    logger.info(f"Phases epochs: {phase_epochs} (Phase1: mixed, Phase2: mixed+chunk, Phase3: practical)")

    global_epoch = 0

    # 便利な補助：loaders_map から利用可能なローダーを取り出す
    span_loader = loaders_map.get("span") if isinstance(loaders_map, dict) else None
    bywork_loader = loaders_map.get("bywork") if isinstance(loaders_map, dict) else None
    chunk_loader = loaders_map.get("chunk") if isinstance(loaders_map, dict) else None
    practical_line_loader = loaders_map.get("practical_line") if isinstance(loaders_map, dict) else None
    practical_chunk_loader = loaders_map.get("practical_chunk") if isinstance(loaders_map, dict) else None

    # Phase ごとのループ
    for phase_idx, n_epochs in enumerate(phase_epochs):
        if n_epochs <= 0:
            continue

        # Phase 切替時のログ
        if phase_idx == 0:
            # Phase1: 元の混合（train_loaders）を使う
            phase_loaders = train_loaders
            logger.info(f"--- PHASE 1 (mixed) : {n_epochs} epochs ---")
        elif phase_idx == 1:
            # Phase2: 優先的に chunk を含める混合。chunk がなければ train_loaders を使う
            candidate = []
            if chunk_loader is not None:
                candidate.append(chunk_loader)
            if bywork_loader is not None:
                candidate.append(bywork_loader)
            if span_loader is not None:
                candidate.append(span_loader)
            phase_loaders = candidate if candidate else train_loaders
            logger.info(f"--- PHASE 2 (mixed + chunk) : {n_epochs} epochs ---")
        else:
            # Phase3: practical 集中特訓
            practical_candidate = []
            if practical_chunk_loader is not None:
                practical_candidate.append(practical_chunk_loader)
            if practical_line_loader is not None:
                practical_candidate.append(practical_line_loader)
            if span_loader is not None:
                practical_candidate.append(span_loader)
            phase_loaders = practical_candidate
            logger.info(f"--- PHASE 3 (practical-focused) : {n_epochs} epochs ---")

        # 各 phase の epoch ループ
        for _ in range(n_epochs):
            # RandomSpan の multi_prob を dataset 側の RandomSpan にだけ適用（dataset は non-practical の ConcatDataset）
            # multi_prob の anneal は global_epoch に基づく（全体通しのスケジュール）
            start_prob, end_prob = 0.5, 0.1
            # total epochs may be 0 guard
            total_epochs = max(1, config.epochs - 1)
            current_prob = start_prob + (end_prob - start_prob) * (global_epoch / total_epochs)
            # dataset.datasets が存在すれば RandomSpan に set_multi_prob をかける
            if hasattr(dataset, "datasets"):
                for ds in dataset.datasets:
                    if isinstance(ds, TranslationDatasetRandomSpan):
                        ds.set_multi_prob(current_prob)
            logger.info(f"📉 RandomSpan multi_prob = {current_prob:.2f} (global_epoch={global_epoch})")

            # 1 epoch の学習
            train_loss = train_epoch(model, phase_loaders, optimizer, scheduler, scaler, device, config, global_epoch, criterion, ema)

            # エポック経過に応じて最初の頃に encoder を完全 unfreeze する（従来の挙動を残す）
            if global_epoch == 1:
                for p in model.parameters():
                    p.requires_grad = True
                logger.info("🔓 Encoder fully unfrozen")

            # 検証
            val_loss = evaluate_model(model, val_loader, device, criterion, use_ema=config.use_ema, ema=ema)

            # logging / 保存
            # total_train_samples: phase_loaders の dataset を合計（分母）
            try:
                total_train_samples = sum(len(l.dataset) for l in (phase_loaders if isinstance(phase_loaders, list) else [phase_loaders]) if l is not None and hasattr(l, "dataset"))
            except Exception:
                total_train_samples = 0
            avg_train_loss = train_loss / total_train_samples if total_train_samples > 0 else train_loss

            logger.info(f"📊 Epoch {global_epoch+1}/{config.epochs} -> Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config.save_dir, "best_model")
                # EMA を適用して保存する
                if config.use_ema and ema is not None:
                    ema.apply_shadow()
                    model.save_pretrained(save_path)
                    ema.restore()
                else:
                    model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"⭐ New best model saved to {save_path}!")

            # Early stopping 判定
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(f"🛑 Early stopping triggered at global epoch {global_epoch+1}")
                break

            global_epoch += 1

        if early_stopping.early_stop:
            break

    logger.info(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    return model, tokenizer


# ===============================
# 8. 翻訳関数
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
# 9. モックテスト
# ===============================
def quick_mock_test():
    logger.info("\n" + "="*60)
    logger.info("🧪 ENHANCED MOCK TEST - Testing New Features (3-Phase)")
    logger.info("="*60 + "\n")

    # mock 用のダミーファイルパス（実際には読まれない）
    file_paths = [
        "mock_randomspan.txt",      # RandomSpan
        "mock_bywork.txt",          # ByWork
        "mock_practical.txt"        # Practical corpus
    ]

    # 0 = RandomSpan
    # 1 = ByWork (line + chunk)
    # 2 = Practical (line + chunk, Phase3 集中)
    file_types = [0, 1, 2]

    config = TrainingConfig(
        # --- 基本 ---
        model_name="facebook/mbart-large-50-many-to-many-mmt",
        save_dir="./mock_output_complete",
        random_seed=42,

        # --- データ ---
        file_paths=file_paths,
        file_types=file_types,
        tags=None,

        # --- mock ---
        mock_mode=True,
        mock_samples=50,

        # --- 学習設定 ---
        epochs=3,
        phase_epochs=[1, 1, 1],  # Phase1 / Phase2 / Phase3
        batch_size=4,
        max_len=128,
        val_split=0.05,

        # --- 最適化 ---
        learning_rate=3e-4,
        weight_decay=0.01,
        num_workers=0,

        # --- LR Finder ---
        use_lr_finder=True,
        lr_finder_min=1e-7,
        lr_finder_max=1e-2,
        lr_finder_num_iter=20,

        # --- 損失・正則化 ---
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        label_smoothing=0.1,

        # --- EMA ---
        use_ema=True,
        ema_decay=0.9999,

        # --- AMP ---
        use_amp=False,
        use_bfloat16=False,

        # --- Early stopping ---
        patience=5,
    )

    model, tokenizer = train_model(config)

    logger.info("\n🧪 Testing translation...")
    test_sentences = [
        "I like apples.",
        "How are you?",
        "This is a test."
    ]

    model.eval()
    device = next(model.parameters()).device
    for s in test_sentences:
        inputs = tokenizer(s, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=4
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"  EN: {s} -> JA: {decoded}")

    logger.info("\n✅ ENHANCED MOCK TEST PASSED!")


# ===============================
# 実行例
# ===============================
if __name__ == "__main__":
    import sys
    
    if "--mock" in sys.argv or os.getenv("MOCK_MODE") == "1":
        quick_mock_test()
    
    else:
        files = [
            "./../data/lyrics_dataset.jsonl",
            "./../data/separated_literary_dataset.jsonl",
            "./../data/OpenSubtitles_sample_40000.jsonl",
            "./../data/TED_sample_40000.jsonl",
            "./../data/Tatoeba_sample_40000.jsonl",
            "./../data/all_outenjp.jsonl"
        ]
        file_values = [2,1,0,0,0,0]
        config = TrainingConfig(
            model_name="Helsinki-NLP/opus-mt-en-jap",
            file_paths=files,
            file_types = file_values,
            epochs=3,
            phase_epochs=[1,1,1],
            batch_size=16,
            max_samples_per_span_file=30000,
            save_dir="./models/translation_model_complete",
            random_seed=42,
            num_workers=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            accumulation_steps=2,
            use_bfloat16=True,
            scheduler_type='onecycle',
            # 🆕 新機能を有効化
            use_focal_loss=False,
            focal_alpha=0.25,
            focal_gamma=2.0,
            use_label_smoothing=True,
            label_smoothing=0.1,
            use_ema=True,
            ema_decay=0.9999,
            use_lr_finder=False,
            lr_finder_min=1e-7,
            lr_finder_max=1e-2,
            lr_finder_num_iter=100
        )
        
        model, tokenizer = train_model(config)
        
        test_sentences = [
            "I like apples.",
            "How are you?",
            "Machine learning is fun.",
            "I couldn't speak English well."
        ]
        results = batch_translate(model, tokenizer, test_sentences)
        for en, ja in zip(test_sentences, results):
            print(f"EN: {en} -> JA: {ja}")
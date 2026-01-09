import torch
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

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================
# 0. 設定クラス
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
    use_compile: bool = False  # torch.compileは互換性問題があるためデフォルトOFF
    scheduler_type: str = 'onecycle'
    warmup_steps: int = 500


# ===============================
# 1. メモリ効率的なデータ読み込み
# ===============================
def load_single_dataset_streaming(file_path, max_samples=None, random_seed=42, tag=None):
    """
    メモリ効率的にJSONLファイルを読み込み
    全行を一度にメモリに読み込まず、必要な分だけ処理
    
    Args:
        file_path: 読み込むファイルパス
        max_samples: 最大サンプル数
        random_seed: ランダムシード
        tag: 英文の先頭に追加するタグ (例: "[LYRICS]")
    """
    en_list, ja_list = [], []
    error_count = 0
    
    logger.info(f"📖 Loading {file_path} ...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        # max_samplesが指定されている場合
        if max_samples:
            # まず全行数をカウント(メモリ効率的に)
            total_lines = sum(1 for _ in f)
            f.seek(0)  # ファイルポインタを先頭に戻す
            
            if total_lines > max_samples:
                logger.info(f"  ⚡ Sampling {max_samples} from {total_lines} lines")
                # サンプリングする行番号を事前に決定
                random.seed(random_seed)
                selected_indices = set(random.sample(range(total_lines), max_samples))
                
                # 選択された行だけを処理
                for idx, line in enumerate(tqdm(f, total=total_lines,
                                                desc=f"Reading {os.path.basename(file_path)}",
                                                unit=" lines")):
                    if idx in selected_indices:
                        try:
                            data = json.loads(line)
                            en, ja = data.get("en"), data.get("ja")
                            if en and ja and len(en.strip()) > 0 and len(ja.strip()) > 0:
                                # 🆕 タグを追加
                                if tag:
                                    en = f"{tag} {en}"
                                en_list.append(en)
                                ja_list.append(ja)
                        except json.JSONDecodeError:
                            error_count += 1
            else:
                # 全行を処理
                f.seek(0)
                for line in tqdm(f, total=total_lines,
                               desc=f"Reading {os.path.basename(file_path)}",
                               unit=" lines"):
                    try:
                        data = json.loads(line)
                        en, ja = data.get("en"), data.get("ja")
                        if en and ja and len(en.strip()) > 0 and len(ja.strip()) > 0:
                            # 🆕 タグを追加
                            if tag:
                                en = f"{tag} {en}"
                            en_list.append(en)
                            ja_list.append(ja)
                    except json.JSONDecodeError:
                        error_count += 1
        else:
            # max_samplesなしの場合は全行を処理
            for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}", unit=" lines"):
                try:
                    data = json.loads(line)
                    en, ja = data.get("en"), data.get("ja")
                    if en and ja and len(en.strip()) > 0 and len(ja.strip()) > 0:
                        # 🆕 タグを追加
                        if tag:
                            en = f"{tag} {en}"
                        en_list.append(en)
                        ja_list.append(ja)
                except json.JSONDecodeError:
                    error_count += 1
    
    if error_count > 0:
        logger.warning(f"  ⚠️  Skipped {error_count} invalid lines")
    
    logger.info(f"  ✅ Loaded {len(en_list)} pairs from {os.path.basename(file_path)}")
    return en_list, ja_list


def load_datasets_balanced(file_paths, max_samples_per_type=None, random_seed=42, tags=None):
    """
    ByWork系とRandomSpan系を分けて、それぞれから適切にサンプリング
    
    Args:
        file_paths: ファイルパスのリスト
        max_samples_per_type: RandomSpan系の各ファイルから取得する最大サンプル数
        random_seed: ランダムシード
        tags: 各ファイルに対応するタグのリスト (Noneの場合はタグなし)
    
    Returns:
        bywork_files: [(file_path, en_list, ja_list), ...]
        span_files: [(file_path, en_list, ja_list), ...]
    """
    bywork_files = []
    span_files = []
    
    # tagsがNoneの場合は全てNoneのリストを作成
    if tags is None:
        tags = [None] * len(file_paths)
    
    for fp, tag in zip(file_paths, tags):
        is_bywork = "separated" in Path(fp).name or "sepalated" in Path(fp).name
        
        if is_bywork:
            # ByWork系は全て読み込む
            logger.info(f"\n🎯 [WORK-LEVEL] {fp} (loading ALL)")
            en_list, ja_list = load_single_dataset_streaming(fp, max_samples=None, random_seed=random_seed, tag=tag)
            bywork_files.append((fp, en_list, ja_list))
        else:
            # RandomSpan系はmax_samples_per_type分だけ
            logger.info(f"\n🎲 [SPAN-LEVEL] {fp}")
            en_list, ja_list = load_single_dataset_streaming(fp, max_samples=max_samples_per_type, random_seed=random_seed, tag=tag)
            span_files.append((fp, en_list, ja_list))
    
    # サマリー表示
    logger.info("\n" + "="*60)
    logger.info("📊 LOADING SUMMARY")
    logger.info("="*60)
    
    total_bywork = sum(len(data[1]) for data in bywork_files)
    logger.info(f"ByWork datasets: {len(bywork_files)} files, {total_bywork:,} pairs total")
    for fp, en_list, _ in bywork_files:
        logger.info(f"  - {os.path.basename(fp)}: {len(en_list):,} pairs")
    
    total_span = sum(len(data[1]) for data in span_files)
    logger.info(f"\nRandomSpan datasets: {len(span_files)} files, {total_span:,} pairs total")
    for fp, en_list, _ in span_files:
        logger.info(f"  - {os.path.basename(fp)}: {len(en_list):,} pairs")
    
    logger.info(f"\n🎉 GRAND TOTAL: {total_bywork + total_span:,} pairs")
    logger.info("="*60 + "\n")
    
    return bywork_files, span_files

# ===============================
# 2. メモリ効率的な Dataset クラス
# ===============================


# RandomSpan 用 collator（dynamic padding + label smoothing）
def build_randomspan_collator(tokenizer, label_smoothing=0.1):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,  # loss は model 側で
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8  # TensorCore 効率
    )


def split_concat_dataset(dataset):
    bywork_indices = []
    span_indices = []

    for i, ds in enumerate(dataset.datasets):
        if isinstance(ds, TranslationDatasetByWorkMemoryEfficient):
            bywork_indices.extend(range(dataset.cumulative_sizes[i - 1] if i > 0 else 0,
                                         dataset.cumulative_sizes[i]))
        else:
            span_indices.extend(range(dataset.cumulative_sizes[i - 1] if i > 0 else 0,
                                       dataset.cumulative_sizes[i]))

    return bywork_indices, span_indices


class TranslationDatasetRandomSpan(Dataset):
    """ランダムスパンデータセット"""

    def __init__(self, en_list, ja_list, tokenizer, max_len=128,
                 multi_prob=0.4,  # 複数文にする確率
                 max_k=4):  # 最大何文くっつけるか
        self.en = en_list
        self.ja = ja_list
        self.tok = tokenizer
        self.max_len = max_len
        self.multi_prob = multi_prob
        self.max_k = max_k
        self.add_prefix = hasattr(tokenizer, 'supported_language_codes')

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        L = len(self.en)

        # --- 文対文 or 複数文 ---
        if random.random() < self.multi_prob:
            k = random.randint(1, self.max_k)
            left = max(0, idx - random.randint(0, k))
            right = min(L, idx + random.randint(1, k + 1))
        else:
            left, right = idx, idx + 1

        src = " ".join(self.en[left:right])
        tgt = " ".join(self.ja[left:right])

        if self.add_prefix:
            src = ">>jap<< " + src

        src_tok = self.tok(src, max_length=self.max_len, truncation=True,
                           padding="max_length", return_tensors="pt")
        tgt_tok = self.tok(text_target=tgt, max_length=self.max_len,
                           truncation=True, padding="max_length",
                           return_tensors="pt")

        labels = tgt_tok["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids": src_tok["input_ids"].squeeze(),
            "attention_mask": src_tok["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

    def set_multi_prob(self, value: float):
        self.multi_prob = max(0.0, min(1.0, value))


class TranslationDatasetByWorkMemoryEfficient(torch.utils.data.Dataset):
    """
    メモリ効率的なByWorkデータセット
    
    改善点:
    1. __init__で全作品をメモリに展開せず、インデックスと位置情報のみ保持
    2. __getitem__で必要な時に必要な作品だけを構築
    """

    def __init__(self, en_list, ja_list, tokenizer, max_len=1024,
                 sep_en="%%%%%%%%THISWORKENDSHERE%%%%%%%%",
                 sep_ja="%%%%%%%%この作品ここまで%%%%%%%%"):
        self.en_list = en_list  # 元のリストへの参照を保持
        self.ja_list = ja_list
        self.tok = tokenizer
        self.max_len = max_len
        self.sep_en = sep_en
        self.sep_ja = sep_ja
        self.add_prefix = hasattr(tokenizer, 'supported_language_codes')

        # 作品の境界インデックスのみを保存(メモリ効率的)
        self.work_boundaries = []  # [(start_idx, end_idx), ...]
        
        start_idx = 0
        for i, (en, ja) in enumerate(zip(en_list, ja_list)):
            if en == self.sep_en and ja == self.sep_ja:
                if i > start_idx:  # 空の作品を避ける
                    self.work_boundaries.append((start_idx, i))
                start_idx = i + 1
        
        # 最後の作品
        if start_idx < len(en_list):
            self.work_boundaries.append((start_idx, len(en_list)))
        
        logger.info(f"  📚 Found {len(self.work_boundaries)} works in ByWork dataset")

    def __len__(self):
        return len(self.work_boundaries)

    def __getitem__(self, idx):
        # 必要な作品のみをその場で構築
        start_idx, end_idx = self.work_boundaries[idx]
        
        src = " ".join(self.en_list[start_idx:end_idx])
        tgt = " ".join(self.ja_list[start_idx:end_idx])

        # 翻訳先言語指定が必要なモデルの場合(OPUS系など)
        if self.add_prefix:
            src = ">>jap<< " + src

        # ---- トークナイズ ----
        src_tok = self.tok(src, max_length=self.max_len, truncation=True,
                           padding="max_length", return_tensors="pt")

        tgt_tok = self.tok(text_target=tgt, max_length=self.max_len,
                           truncation=True, padding="max_length",
                           return_tensors="pt")

        labels = tgt_tok["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids": src_tok["input_ids"].squeeze(),
            "attention_mask": src_tok["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


def build_combined_dataset(file_paths, tokenizer, max_len=256,
                          max_samples_per_span_file=None, random_seed=42, tags=None):
    """
    ByWork系とRandomSpan系を適切にサンプリングして結合
    
    Args:
        file_paths: ファイルパスのリスト
        tokenizer: トークナイザー
        max_len: 最大トークン長
        max_samples_per_span_file: RandomSpan系の各ファイルから取る最大サンプル数
        random_seed: ランダムシード
        tags: 各ファイルに対応するタグのリスト
    """
    # データ読み込み (バランス調整済み)
    bywork_files, span_files = load_datasets_balanced(
        file_paths,
        max_samples_per_type=max_samples_per_span_file,
        random_seed=random_seed,
        tags=tags
    )
    
    datasets = []
    
    # ByWork系のデータセット作成(メモリ効率的バージョン使用)
    for fp, en_list, ja_list in bywork_files:
        ds = TranslationDatasetByWorkMemoryEfficient(en_list, ja_list, tokenizer, max_len=max_len)
        datasets.append(ds)
        logger.info(f"✅ Created ByWork dataset from {os.path.basename(fp)}: {len(ds)} works")
    
    # RandomSpan系のデータセット作成
    for fp, en_list, ja_list in span_files:
        ds = TranslationDatasetRandomSpan(en_list, ja_list, tokenizer, max_len=max_len)
        datasets.append(ds)
        logger.info(f"✅ Created RandomSpan dataset from {os.path.basename(fp)}: {len(ds)} pairs")
    
    # 複数 dataset を連結
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)
    logger.info(f"\n🎯 Combined dataset total size: {len(combined)}")
    
    return combined


# ===============================
# 3. 検証関数
# ===============================
def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)


# ===============================
# 4. Early Stopping
# ===============================
class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.001):
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
            logger.info(f"⚠️ No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def freeze_encoder_layers(model, ratio=0.5):
    enc_layers = model.model.encoder.layers
    freeze_until = int(len(enc_layers) * ratio)
    for i, layer in enumerate(enc_layers):
        for p in layer.parameters():
            p.requires_grad = i >= freeze_until


# ===============================
# 5. リファクタリングされた学習関数群
# ===============================

def setup_training(config: TrainingConfig):
    """デバイス、ロギング、トークナイザ、モデルの初期化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    
    logger.info("\n" + "="*60 + "\n⚡ SPEED OPTIMIZATION SETTINGS\n" + "="*60)
    logger.info(f"✓ DataLoader workers: {config.num_workers}")
    logger.info(f"✓ Gradient accumulation steps: {config.accumulation_steps}")
    logger.info(f"✓ Effective batch size: {config.batch_size * config.accumulation_steps}")
    logger.info(f"✓ BFloat16: {config.use_bfloat16 and str(device) == 'cuda'}")
    logger.info(f"✓ torch.compile: {config.use_compile}")
    logger.info(f"✓ Scheduler type: {config.scheduler_type}")
    if config.scheduler_type == 'linear_warmup': logger.info(f"✓ Warmup steps: {config.warmup_steps}")
    logger.info("="*60 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_safetensors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, use_safetensors=True).to(device)
    model.config.dropout = 0.15
    model.config.attention_dropout = 0.15
    
    if config.use_compile and hasattr(torch, 'compile'):
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("✅ Model compiled successfully!")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}. Continuing without compilation.")
            
    return device, tokenizer, model

def create_dataloaders(config: TrainingConfig, tokenizer):
    """データセットとデータローダーの構築"""
    dataset = build_combined_dataset(
        config.file_paths,
        tokenizer,
        max_len=config.max_len,
        max_samples_per_span_file=config.max_samples_per_span_file,
        random_seed=config.random_seed,
        tags=config.tags
    )

    # Split dataset into training and validation sets
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )

    # Separate indices for bywork and span datasets within the training set
    bywork_idx, span_idx = split_concat_dataset(dataset)
    train_indices = set(train_dataset.indices)
    train_bywork_idx = [i for i in bywork_idx if i in train_indices]
    train_span_idx = [i for i in span_idx if i in train_indices]

    train_bywork = Subset(dataset, train_bywork_idx)
    train_span = Subset(dataset, train_span_idx)
    
    logger.info(f"\n📊 Dataset split:")
    logger.info(f"  Training: {len(train_dataset):,} samples ({len(train_bywork)} by-work, {len(train_span)} span)")
    logger.info(f"  Validation: {len(val_dataset):,} samples\n")

    span_collator = build_randomspan_collator(tokenizer)

    loader_args = {'num_workers': config.num_workers, 'pin_memory': True, 'persistent_workers': config.num_workers > 0}

    train_loader_span = DataLoader(train_span, batch_size=config.batch_size, shuffle=True, collate_fn=span_collator, **loader_args)
    train_loader_bywork = DataLoader(train_bywork, batch_size=max(1, config.batch_size // 4), shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, **loader_args)
    
    return [train_loader_span, train_loader_bywork], val_loader, dataset


def create_optimizer_and_scheduler(model, config: TrainingConfig, train_loaders):
    """オプティマイザとスケジューラの作成"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    total_steps_per_epoch = sum(len(loader) for loader in train_loaders)
    total_steps = total_steps_per_epoch * config.epochs // config.accumulation_steps

    if config.scheduler_type == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate * 10, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)
        logger.info(f"📈 OneCycleLR scheduler initialized (total_steps={total_steps})")
    elif config.scheduler_type == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)
        logger.info(f"📈 Linear warmup scheduler initialized (warmup_steps={config.warmup_steps}, total_steps={total_steps})")
    else:
        scheduler = None
        
    return optimizer, scheduler

def train_epoch(model, loaders, optimizer, scheduler, scaler, device, config: TrainingConfig, epoch: int):
    """1エポック分の学習処理"""
    model.train()
    total_loss = 0
    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    for loader in loaders:
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx % config.accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            
            loss_divisor = config.accumulation_steps
            autocast_args = {'dtype': torch.bfloat16} if use_bf16 else {}
            
            with autocast(**autocast_args):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / loss_divisor

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % config.accumulation_steps == 0:
                if config.gradient_clip > 0:
                    if scaler: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                if scheduler: scheduler.step()

            batch_loss = loss.item() * loss_divisor
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            
    return total_loss

def train_model(config: TrainingConfig):
    """最適化された学習関数（リファクタリング版）"""
    device, tokenizer, model = setup_training(config)
    train_loaders, val_loader, dataset = create_dataloaders(config, tokenizer)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loaders)
    
    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if config.use_bfloat16 and not use_bf16: logger.warning("⚠️ BFloat16 requested but not supported. Falling back to FP16.")
    scaler = GradScaler() if config.use_amp and device.type == "cuda" and not use_bf16 else None
    
    early_stopping = EarlyStopping(patience=config.patience)
    best_val_loss = float('inf')
    os.makedirs(config.save_dir, exist_ok=True)
    
    freeze_encoder_layers(model, ratio=0.5)
    logger.info("🔒 Encoder layers partially frozen (ratio=0.5)")

    for epoch in range(config.epochs):
        start_prob, end_prob = 0.5, 0.1
        current_prob = start_prob + (end_prob - start_prob) * (epoch / max(1, config.epochs - 1))
        for ds in dataset.datasets:
            if isinstance(ds, TranslationDatasetRandomSpan): ds.set_multi_prob(current_prob)
        logger.info(f"📉 RandomSpan multi_prob = {current_prob:.2f}")

        train_loss = train_epoch(model, train_loaders, optimizer, scheduler, scaler, device, config, epoch)
        
        if epoch == 1:
            for p in model.parameters(): p.requires_grad = True
            logger.info("🔓 Encoder fully unfrozen")

        val_loss = evaluate_model(model, val_loader, device)
        total_train_samples = sum(len(l.dataset) for l in train_loaders)
        avg_train_loss = train_loss / total_train_samples if total_train_samples > 0 else 0
        logger.info(f"📊 Epoch {epoch+1}/{config.epochs} -> Train loss: {avg_train_loss:.4f}, Validation loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.save_dir, "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"⭐ New best model saved to {save_path}!")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    return model, tokenizer

# ===============================
# 6. 翻訳関数
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
# 実行例
# ===============================
if __name__ == "__main__":
    files = [
        "./../data/sepalated_dataset.jsonl",  # ByWork系
        "./../data/OpenSubtitles_sample_40000.jsonl",  # RandomSpan系
        "./../data/TED_sample_40000.jsonl",  # RandomSpan系
        "./../data/Tatoeba_sample_40000.jsonl",  # RandomSpan系
        "./../data/all_outenjp.jsonl"  # RandomSpan系 (歌詞など)
    ]
    
    # 🆕 各ファイルに対応するタグ (必要に応じて設定)
    # tags = [None, None, None, None, "[LYRICS]"]  # 歌詞データにタグを付ける例
    tags = None  # タグなしの場合
   
    # --- 学習設定 ---
    config = TrainingConfig(
        model_name="Helsinki-NLP/opus-mt-en-jap",
        file_paths=files,
        epochs=2,
        batch_size=16,
        max_samples_per_span_file=40000,
        save_dir="./models/translation_model_final",
        random_seed=42,
        tags=tags,
        # --- 高速化設定 ---
        num_workers=4,
        accumulation_steps=4,
        use_bfloat16=True,
        scheduler_type='onecycle'
    )
    
    # --- 学習実行 ---
    model, tokenizer = train_model(config)
    
    # --- 翻訳テスト ---
    test_sentences = [
        "I like apples.",
        "How are you?",
        "Machine learning is fun.",
        "I couldn't speak English well."
    ]
    results = batch_translate(model, tokenizer, test_sentences)
    for en, ja in zip(test_sentences, results):
        print(f"EN: {en} -> JA: {ja}")

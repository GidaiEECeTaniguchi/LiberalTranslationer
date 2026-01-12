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
    use_compile: bool = False
    scheduler_type: str = 'onecycle'
    warmup_steps: int = 500
    
    # 🆕 モックモード設定
    mock_mode: bool = False
    mock_samples: int = 100  # モック時のサンプル数
    mock_force_cpu: bool = True  # モック時は強制的にCPU使用


# ===============================
# 🆕 モックデータ生成
# ===============================
def generate_mock_data(num_samples=100, seed=42):
    """
    テスト用のモックデータを生成
    
    Args:
        num_samples: 生成するサンプル数
        seed: ランダムシード
    
    Returns:
        en_list, ja_list
    """
    random.seed(seed)
    
    # シンプルな英日対訳のテンプレート
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


def create_mock_jsonl_files(output_dir="./mock_data", num_files=2, samples_per_file=50):
    """
    モック用のJSONLファイルを生成
    
    Args:
        output_dir: 出力ディレクトリ
        num_files: 生成するファイル数
        samples_per_file: ファイルごとのサンプル数
    
    Returns:
        生成されたファイルパスのリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    for i in range(num_files):
        file_path = os.path.join(output_dir, f"mock_data_{i+1}.jsonl")
        en_list, ja_list = generate_mock_data(samples_per_file, seed=42 + i)
        
        with open(file_path, "w", encoding="utf-8") as f:
            for en, ja in zip(en_list, ja_list):
                json.dump({"en": en, "ja": ja}, f, ensure_ascii=False)
                f.write("\n")
        
        file_paths.append(file_path)
        logger.info(f"✅ Created mock file: {file_path}")
    
    return file_paths


# ===============================
# 1. メモリ効率的なデータ読み込み
# ===============================
def add_tag_if_needed(text, tag):
    """タグを追加するヘルパー関数"""
    return f"{tag} {text}" if tag else text


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
    
    try:
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
                                    en_list.append(add_tag_if_needed(en, tag))
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
                                en_list.append(add_tag_if_needed(en, tag))
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

# RandomSpan 用 collator(dynamic padding + label smoothing)
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
    """ByWork系データ用Dataset (シンプル版)"""
    def __init__(self, en_texts, ja_texts, tokenizer, max_len=64):
        self.en_texts = en_texts
        self.ja_texts = ja_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, idx):
        en_text = self.en_texts[idx]
        ja_text = self.ja_texts[idx]
        
        if hasattr(self.tokenizer, 'supported_language_codes'):
            en_text = ">>jap<< " + en_text

        inputs = self.tokenizer(en_text, max_length=self.max_len, truncation=True, padding="max_length")
        labels = self.tokenizer(ja_text, max_length=self.max_len, truncation=True, padding="max_length")

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


def evaluate_model(model, val_loader, device):
    """検証ループ"""
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
# 5. 学習メイン処理
# ===============================
def setup_training(config: TrainingConfig):
    """デバイス、モデル、トークナイザーの初期化"""
    # 🆕 モックモード時の処理
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
    
    # モデルとトークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(device)
    
    # torch.compile (オプション)
    if config.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("⚡ Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}")
    
    return device, tokenizer, model


def create_dataloaders(config: TrainingConfig, tokenizer):
    """データローダーの作成"""
    # 🆕 モックモード時はモックデータを使用
    if config.mock_mode:
        logger.info(f"🎭 Generating {config.mock_samples} mock samples...")
        en_list, ja_list = generate_mock_data(config.mock_samples, seed=config.random_seed)
        
        # モックデータをRandomSpan形式として扱う
        span_files = [("mock_data", en_list, ja_list)]
        bywork_files = []
    else:
        # 通常のデータ読み込み
        bywork_files, span_files = load_datasets_balanced(
            config.file_paths,
            max_samples_per_type=config.max_samples_per_span_file,
            random_seed=config.random_seed,
            tags=config.tags
        )
    
    # データセット作成
    all_datasets = []
    
    # RandomSpan系
    for _, en_list, ja_list in span_files:
        ds = TranslationDatasetRandomSpan(en_list, ja_list, tokenizer, max_len=config.max_len)
        all_datasets.append(ds)
    
    # ByWork系
    for _, en_list, ja_list in bywork_files:
        ds = TranslationDatasetByWork(en_list, ja_list, tokenizer, max_len=config.max_len)
        all_datasets.append(ds)
    
    if not all_datasets:
        raise ValueError("❌ No data loaded! Check file paths.")
    
    # データセットの結合
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(all_datasets)
    
    # Train/Val分割
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"📊 Train: {train_size:,} samples, Validation: {val_size:,} samples")
    
    # RandomSpanとByWorkを分離
    span_indices = []
    bywork_indices = []
    offset = 0
    
    for i, ds in enumerate(all_datasets):
        ds_len = len(ds)
        if isinstance(ds, TranslationDatasetRandomSpan):
            span_indices.extend(range(offset, offset + ds_len))
        else:
            bywork_indices.extend(range(offset, offset + ds_len))
        offset += ds_len
    
    train_span_indices = [i for i in train_dataset.indices if i in span_indices]
    train_bywork_indices = [i for i in train_dataset.indices if i in bywork_indices]
    
    train_span = Subset(dataset, train_span_indices)
    train_bywork = Subset(dataset, train_bywork_indices)
    
    # Collator
    span_collator = build_randomspan_collator(tokenizer, label_smoothing=0.1)
    
    # DataLoader設定
    loader_args = {
        'num_workers': 0 if config.mock_mode else config.num_workers,  # 🆕 モック時はnum_workers=0
        'pin_memory': False if config.mock_mode else True,  # 🆕 モック時はpin_memory無効
        'persistent_workers': False
    }
    
    # 🆕 バッチサイズの安全性チェック
    actual_batch_size = config.batch_size
    bywork_batch_size = max(1, config.batch_size // 4)
    
    if config.accumulation_steps > len(train_span) // actual_batch_size:
        logger.warning(f"⚠️ accumulation_steps ({config.accumulation_steps}) is large relative to dataset size")
    
    train_loader_span = DataLoader(train_span, batch_size=actual_batch_size, shuffle=True, collate_fn=span_collator, **loader_args)
    
    # 🆕 ByWorkデータが存在する場合のみローダーを作成
    train_loaders = [train_loader_span]
    if len(train_bywork) > 0:
        train_loader_bywork = DataLoader(train_bywork, batch_size=bywork_batch_size, shuffle=True, **loader_args)
        train_loaders.append(train_loader_bywork)
    else:
        logger.info("ℹ️ No ByWork data - using RandomSpan only")
    
    # 🆕 val_loaderにもcollatorを適用
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size * 2, shuffle=False, collate_fn=span_collator, **loader_args)
    
    return train_loaders, val_loader, dataset


def create_optimizer_and_scheduler(model, config: TrainingConfig, train_loaders):
    """オプティマイザとスケジューラの作成"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    total_steps_per_epoch = sum(len(loader) for loader in train_loaders)
    total_steps = total_steps_per_epoch * config.epochs // config.accumulation_steps

    if config.scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=config.learning_rate * 10, 
            total_steps=total_steps, 
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
            num_training_steps=total_steps
        )
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

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss_divisor = config.accumulation_steps
            
            # 🆕 新しいautocast形式を使用
            if use_bf16:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / loss_divisor
            else:
                with autocast(enabled=False):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / loss_divisor

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
                
                if scheduler:
                    scheduler.step()

            batch_loss = loss.item() * loss_divisor
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            
    return total_loss


def train_model(config: TrainingConfig):
    """最適化された学習関数(リファクタリング版)"""
    device, tokenizer, model = setup_training(config)
    train_loaders, val_loader, dataset = create_dataloaders(config, tokenizer)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loaders)
    
    use_bf16 = config.use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if config.use_bfloat16 and not use_bf16:
        logger.warning("⚠️ BFloat16 requested but not supported. Falling back to FP16.")
    
    scaler = GradScaler() if config.use_amp and device.type == "cuda" and not use_bf16 else None
    
    early_stopping = EarlyStopping(patience=config.patience)
    best_val_loss = float('inf')
    os.makedirs(config.save_dir, exist_ok=True)
    
    freeze_encoder_layers(model, ratio=0.5)
    logger.info("🔒 Encoder layers partially frozen (ratio=0.5)")

    for epoch in range(config.epochs):
        start_prob, end_prob = 0.5, 0.1
        current_prob = start_prob + (end_prob - start_prob) * (epoch / max(1, config.epochs - 1))
        
        # RandomSpanデータセットにmulti_probを設定
        for ds in dataset.datasets:
            if isinstance(ds, TranslationDatasetRandomSpan):
                ds.set_multi_prob(current_prob)
        logger.info(f"📉 RandomSpan multi_prob = {current_prob:.2f}")

        train_loss = train_epoch(model, train_loaders, optimizer, scheduler, scaler, device, config, epoch)
        
        if epoch == 1:
            for p in model.parameters():
                p.requires_grad = True
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
# 🆕 7. モック実行用のヘルパー関数
# ===============================
def quick_mock_test():
    """構文チェック用のクイックテスト"""
    logger.info("\n" + "="*60)
    logger.info("🧪 QUICK MOCK TEST - Syntax and Basic Functionality Check")
    logger.info("="*60 + "\n")
    
    config = TrainingConfig(
        model_name="Helsinki-NLP/opus-mt-en-jap",
        file_paths=[],  # モックモードでは不要
        epochs=1,
        batch_size=4,
        mock_mode=True,
        mock_samples=20,
        mock_force_cpu=True,
        num_workers=0,
        accumulation_steps=1,
        use_amp=False,
        use_bfloat16=False,
        save_dir="./mock_output"
    )
    
    try:
        model, tokenizer = train_model(config)
        
        # 翻訳テスト
        logger.info("\n🧪 Testing translation...")
        test_sentences = ["I like apples.", "How are you?"]
        results = batch_translate(model, tokenizer, test_sentences)
        
        for en, ja in zip(test_sentences, results):
            logger.info(f"  EN: {en} -> JA: {ja}")
        
        logger.info("\n✅ MOCK TEST PASSED - All syntax checks successful!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ MOCK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===============================
# 実行例
# ===============================
if __name__ == "__main__":
    # 🆕 環境変数でモードを切り替え
    import sys
    
    if "--mock" in sys.argv or os.getenv("MOCK_MODE") == "1":
        # モックモードで実行
        quick_mock_test()
    
    elif "--mock-with-files" in sys.argv:
        # モックファイルを生成して実行
        logger.info("🎭 Creating mock JSONL files...")
        mock_files = create_mock_jsonl_files(output_dir="./mock_data", num_files=2, samples_per_file=50)
        
        config = TrainingConfig(
            model_name="Helsinki-NLP/opus-mt-en-jap",
            file_paths=mock_files,
            epochs=1,
            batch_size=4,
            mock_mode=True,
            mock_force_cpu=True,
            save_dir="./mock_output"
        )
        
        model, tokenizer = train_model(config)
    
    else:
        # 通常モード（実際の学習）
        files = [
            "./../data/sepalated_dataset.jsonl",
            "./../data/OpenSubtitles_sample_40000.jsonl",
            "./../data/TED_sample_40000.jsonl",
            "./../data/Tatoeba_sample_40000.jsonl",
            "./../data/all_outenjp.jsonl"
        ]
        
        # 各ファイルに対応するタグ (必要に応じて設定)
        # tags = [None, None, None, None, "[LYRICS]"]
        tags = None
       
        config = TrainingConfig(
            model_name="Helsinki-NLP/opus-mt-en-jap",
            file_paths=files,
            epochs=2,
            batch_size=16,
            max_samples_per_span_file=40000,
            save_dir="./models/translation_model_final",
            random_seed=42,
            tags=tags,
            num_workers=4,
            accumulation_steps=4,
            use_bfloat16=True,
            scheduler_type='onecycle'
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
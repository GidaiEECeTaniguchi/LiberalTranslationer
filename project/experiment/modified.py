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

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
# 5. 高速化された学習関数
# ===============================
def train_model(
    model_name,
    file_paths,
    epochs=3,
    batch_size=32,
    use_amp=True,
    max_samples_per_span_file=None,
    val_split=0.05,
    save_dir="./models",
    learning_rate=1e-4,
    gradient_clip=1.0,
    save_every=1,
    patience=2,
    max_len=64,
    random_seed=42,
    tags=None,
    # 🆕 高速化パラメータ
    num_workers=4,  # DataLoaderのワーカー数
    accumulation_steps=4,  # Gradient Accumulation
    use_bfloat16=True,  # BFloat16を使用するか
    use_compile=True,  # torch.compileを使用するか
    scheduler_type='onecycle',  # 'onecycle' or 'linear_warmup'
    warmup_steps=500  # linear_warmup用のウォームアップステップ数
):
    """
    最適化された学習関数
    
    Args:
        scheduler_type: 'onecycle' (OneCycleLR) または 'linear_warmup' (get_linear_schedule_with_warmup)
        warmup_steps: linear_warmup使用時のウォームアップステップ数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    
    # 🆕 高速化設定のログ出力
    logger.info("\n" + "="*60)
    logger.info("⚡ SPEED OPTIMIZATION SETTINGS")
    logger.info("="*60)
    logger.info(f"✓ DataLoader workers: {num_workers}")
    logger.info(f"✓ Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"✓ Effective batch size: {batch_size * accumulation_steps}")
    logger.info(f"✓ BFloat16: {use_bfloat16 and device.type == 'cuda'}")
    logger.info(f"✓ torch.compile: {use_compile}")
    logger.info(f"✓ Scheduler type: {scheduler_type}")
    if scheduler_type == 'linear_warmup':
        logger.info(f"✓ Warmup steps: {warmup_steps}")
    logger.info("="*60 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)
    model.config.dropout = 0.15
    model.config.attention_dropout = 0.15
    
    # 🆕 torch.compile (PyTorch 2.0+)
    # Transformersとの互換性問題を回避するため、エラー抑制を有効化
    """
    if use_compile and hasattr(torch, 'compile'):
        logger.info("🔥 Compiling model with torch.compile...")
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True  # Transformers互換性のため
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("✅ Model compiled successfully!")
        except Exception as e:
            logger.warning(f"⚠️  torch.compile failed: {e}. Continuing without compilation.")
            use_compile = False
    """
    # データセット構築
    dataset = build_combined_dataset(
        file_paths,
        tokenizer,
        max_len=max_len,
        max_samples_per_span_file=max_samples_per_span_file,
        random_seed=random_seed,
        tags=tags
    )
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # まず ConcatDataset 全体から index を作る
    bywork_idx, span_idx = split_concat_dataset(dataset)

    # 次に train / val 分割
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    train_indices = set(train_dataset.indices)

    train_bywork_idx = [i for i in bywork_idx if i in train_indices]
    train_span_idx = [i for i in span_idx   if i in train_indices]

    train_bywork = Subset(dataset, train_bywork_idx)
    train_span = Subset(dataset, train_span_idx)
    
    logger.info(f"\n📊 Dataset split:")
    logger.info(f"  Training: {train_size:,} samples")
    logger.info(f"  Validation: {val_size:,} samples\n")
    
    # 🆕 DataLoaderの最適化
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # マルチプロセス読み込み
        pin_memory=True,  # GPU転送の高速化
        prefetch_factor=2,  # 先読みバッファ
        persistent_workers=True if num_workers > 0 else False  # ワーカープロセスを維持
    )

    span_collator = build_randomspan_collator(tokenizer)

    train_loader_span = DataLoader(
        train_span,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=span_collator,
        pin_memory=True,
        persistent_workers=True
    )

    train_loader_bywork = DataLoader(
        train_bywork,
        batch_size=max(1, batch_size // 4),  # 長文なので小さめ
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 🆕 スケジューラの選択
    scheduler = None
    if scheduler_type == 'onecycle':
        total_steps = len(train_loader) * epochs // accumulation_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # 最大学習率
            total_steps=total_steps,
            pct_start=0.3,  # ウォームアップの割合
            anneal_strategy='cos',
            div_factor=25.0,  # 初期学習率 = max_lr / div_factor
            final_div_factor=1e4  # 最終学習率 = max_lr / final_div_factor
        )
        logger.info(f"📈 OneCycleLR scheduler initialized (total_steps={total_steps})")
    elif scheduler_type == 'linear_warmup':
        num_training_steps = (len(train_loader) // accumulation_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        logger.info(f"📈 Linear warmup scheduler initialized (warmup_steps={warmup_steps}, total_steps={num_training_steps})")
    
    # 🆕 BFloat16サポートチェック
    use_bf16 = use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_bfloat16 and not use_bf16:
        logger.warning("⚠️  BFloat16 requested but not supported. Falling back to FP16.")
    
    scaler = GradScaler() if use_amp and device.type == "cuda" and not use_bf16 else None
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    freeze_encoder_layers(model, ratio=0.5)

    for epoch in range(epochs):
        start_prob = 0.5
        end_prob = 0.1
        current_prob = start_prob + (end_prob - start_prob) * (epoch / max(1, epochs - 1))
        for ds in dataset.datasets:
            if isinstance(ds, TranslationDatasetRandomSpan):
                ds.set_multi_prob(current_prob)

        logger.info(f"📉 RandomSpan multi_prob = {current_prob:.2f}")

        model.train()
        total_loss = 0
        loaders = [train_loader_span, train_loader_bywork]
        for loader in loaders:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(pbar):

                # accumulation の先頭で zero_grad
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if use_bf16:
                    with autocast(dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / accumulation_steps
                    loss.backward()

                elif scaler:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / accumulation_steps
                    scaler.scale(loss).backward()

                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps
                    loss.backward()

                # ★ ここから「更新フェーズ」
                if (batch_idx + 1) % accumulation_steps == 0:

                    if gradient_clip > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    if scheduler:
                        scheduler.step()

                # ★ ログ用損失は毎バッチ
                total_loss += loss.item() * accumulation_steps

                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss.item() * accumulation_steps:.4f}",
                    lr=f"{current_lr:.2e}"
                )

        if epoch == 1:
            for p in model.parameters():
                p.requires_grad = True
            logger.info("🔓 Encoder fully unfrozen")

        # エポック終了時の検証
        val_loss = evaluate_model(model, val_loader, device)
        logger.info(f"📊 Epoch {epoch+1}/{epochs} - Validation loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(save_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
            logger.info("⭐ New best model saved!")
        
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
   
    MODEL_NAME = "Helsinki-NLP/opus-mt-en-jap"
    SAVE_DIR = "./models/translation_model_final"
    
    # === OneCycleLR スケジューラを使う場合 ===
    model, tokenizer = train_model(
        MODEL_NAME,
        files,
        epochs=2,
        batch_size=16,
        max_samples_per_span_file=40000,
        save_dir=SAVE_DIR,
        random_seed=42,
        tags=tags,
        # 高速化パラメータ
        num_workers=4,
        accumulation_steps=4,
        use_bfloat16=True,
        use_compile=True,
        scheduler_type='onecycle'  # OneCycleLR使用
    )
    
    # === Linear Warmup スケジューラを使う場合 ===
    # model, tokenizer = train_model(
    #     MODEL_NAME,
    #     files,
    #     epochs=2,
    #     batch_size=16,
    #     max_samples_per_span_file=40000,
    #     save_dir=SAVE_DIR,
    #     random_seed=42,
    #     tags=tags,
    #     # 高速化パラメータ
    #     num_workers=4,
    #     accumulation_steps=4,
    #     use_bfloat16=True,
    #     use_compile=True,
    #     scheduler_type='linear_warmup',  # Linear Warmup使用
    #     warmup_steps=500
    # )
    
    test_sentences = [
        "I like apples.",
        "How are you?",
        "Machine learning is fun.",
        "I couldn't speak English well."
    ]
    results = batch_translate(model, tokenizer, test_sentences)
    for en, ja in zip(test_sentences, results):
        print(f"EN: {en} -> JA: {ja}")

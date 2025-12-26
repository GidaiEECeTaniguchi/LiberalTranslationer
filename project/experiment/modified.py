import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random

import os
from pathlib import Path
import json

# ===============================
# 1. データ読み込み (JSONL対応)
# ===============================
def load_single_dataset(file_path, max_samples=None):
    """単一JSONLファイルを読み込み"""
    en_list, ja_list = [], []
    
    print(f"📖 Loading {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        # max_samplesが指定されている場合はランダムサンプリング
        if max_samples and len(lines) > max_samples:
            print(f"  ⚡ Sampling {max_samples} from {len(lines)} lines")
            lines = random.sample(lines, max_samples)
        
        for line in tqdm(lines, desc=f"Reading {os.path.basename(file_path)}", unit=" lines"):
            try:
                data = json.loads(line)
                en, ja = data.get("en"), data.get("ja")
                if en and ja:
                    en_list.append(en)
                    ja_list.append(ja)
            except json.JSONDecodeError:
                continue
    
    print(f"  ✅ Loaded {len(en_list)} pairs from {os.path.basename(file_path)}")
    return en_list, ja_list


def load_datasets_balanced(file_paths, max_samples_per_type=None):
    """
    ByWork系とRandomSpan系を分けて、それぞれから適切にサンプリング
    
    Args:
        file_paths: ファイルパスのリスト
        max_samples_per_type: RandomSpan系の各ファイルから取得する最大サンプル数
                             (ByWork系は全て使用)
    
    Returns:
        bywork_files: [(file_path, en_list, ja_list), ...]
        span_files: [(file_path, en_list, ja_list), ...]
    """
    bywork_files = []
    span_files = []
    
    for fp in file_paths:
        is_bywork = "separated" in Path(fp).name or "sepalated" in Path(fp).name
        
        if is_bywork:
            # ByWork系は全て読み込む
            print(f"\n🎯 [WORK-LEVEL] {fp} (loading ALL)")
            en_list, ja_list = load_single_dataset(fp, max_samples=None)
            bywork_files.append((fp, en_list, ja_list))
        else:
            # RandomSpan系はmax_samples_per_type分だけ
            print(f"\n🎲 [SPAN-LEVEL] {fp}")
            en_list, ja_list = load_single_dataset(fp, max_samples=max_samples_per_type)
            span_files.append((fp, en_list, ja_list))
    
    # サマリー表示
    print("\n" + "="*60)
    print("📊 LOADING SUMMARY")
    print("="*60)
    
    total_bywork = sum(len(data[1]) for data in bywork_files)
    print(f"ByWork datasets: {len(bywork_files)} files, {total_bywork:,} pairs total")
    for fp, en_list, _ in bywork_files:
        print(f"  - {os.path.basename(fp)}: {len(en_list):,} pairs")
    
    total_span = sum(len(data[1]) for data in span_files)
    print(f"\nRandomSpan datasets: {len(span_files)} files, {total_span:,} pairs total")
    for fp, en_list, _ in span_files:
        print(f"  - {os.path.basename(fp)}: {len(en_list):,} pairs")
    
    print(f"\n🎉 GRAND TOTAL: {total_bywork + total_span:,} pairs")
    print("="*60 + "\n")
    
    return bywork_files, span_files

# ===============================
# 2. Dataset クラス
# ===============================
from torch.utils.data import Dataset

class TranslationDatasetRandomSpan(Dataset):
    def __init__(self, en_list, ja_list, tokenizer, max_len=128,
                 multi_prob=0.4,   # 複数文にする確率
                 max_k=4):         # 最大何文くっつけるか
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


class TranslationDatasetByWork(torch.utils.data.Dataset):
    def __init__(self, en_list, ja_list, tokenizer, max_len=1024,
                 sep_en="%%%%%%%%THISWORKENDSHERE%%%%%%%%",
                 sep_ja="%%%%%%%%この作品ここまで%%%%%%%%"):
        self.tok = tokenizer
        self.max_len = max_len
        self.sep_en = sep_en
        self.sep_ja = sep_ja
        self.add_prefix = hasattr(tokenizer, 'supported_language_codes')

        # ---- ここで作品単位にまとめる ----
        self.en_works = []
        self.ja_works = []

        cur_en = []
        cur_ja = []

        for en, ja in zip(en_list, ja_list):
            if en == self.sep_en and ja == self.sep_ja:
                # 作品終了 → バッファを固めて保存
                if cur_en and cur_ja:
                    self.en_works.append(" ".join(cur_en))
                    self.ja_works.append(" ".join(cur_ja))
                cur_en = []
                cur_ja = []
            else:
                cur_en.append(en)
                cur_ja.append(ja)

        # 最後に作品が終わらず残った場合
        if cur_en and cur_ja:
            self.en_works.append(" ".join(cur_en))
            self.ja_works.append(" ".join(cur_ja))

    def __len__(self):
        return len(self.en_works)

    def __getitem__(self, idx):
        src = self.en_works[idx]
        tgt = self.ja_works[idx]

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
                          max_samples_per_span_file=None):
    """
    ByWork系とRandomSpan系を適切にサンプリングして結合
    
    Args:
        file_paths: ファイルパスのリスト
        tokenizer: トークナイザー
        max_len: 最大トークン長
        max_samples_per_span_file: RandomSpan系の各ファイルから取る最大サンプル数
    """
    # データ読み込み (バランス調整済み)
    bywork_files, span_files = load_datasets_balanced(
        file_paths, 
        max_samples_per_type=max_samples_per_span_file
    )
    
    datasets = []
    
    # ByWork系のデータセット作成
    for fp, en_list, ja_list in bywork_files:
        ds = TranslationDatasetByWork(en_list, ja_list, tokenizer, max_len=max_len)
        datasets.append(ds)
        print(f"✅ Created ByWork dataset from {os.path.basename(fp)}: {len(ds)} works")
    
    # RandomSpan系のデータセット作成
    for fp, en_list, ja_list in span_files:
        ds = TranslationDatasetRandomSpan(en_list, ja_list, tokenizer, max_len=max_len)
        datasets.append(ds)
        print(f"✅ Created RandomSpan dataset from {os.path.basename(fp)}: {len(ds)} pairs")
    
    # 複数 dataset を連結
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)
    print(f"\n🎯 Combined dataset total size: {len(combined)}")
    
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
            print(f"⚠️ No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ===============================
# 5. 学習関数
# ===============================
def train_model(
    model_name,
    file_paths,
    epochs=3,
    batch_size=32,
    use_amp=True,
    max_samples_per_span_file=None,  # RandomSpan系の各ファイルからの最大サンプル数
    val_split=0.05,
    save_dir="./models",
    learning_rate=1e-4,
    gradient_clip=1.0,
    save_every=1,
    patience=2,
    max_len=64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)
    
    # 改善されたデータセット構築
    dataset = build_combined_dataset(
        file_paths, 
        tokenizer, 
        max_len=max_len,
        max_samples_per_span_file=max_samples_per_span_file
    )
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Dataset split:")
    print(f"  Training: {train_size:,} samples")
    print(f"  Validation: {val_size:,} samples\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if scaler:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss = evaluate_model(model, val_loader, device)
        print(f"📊 Validation loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(save_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
            print("⭐ New best model saved!")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
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
        batch_texts = texts[i:i+batch_size]
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
        "./../data/sepalated_dataset.jsonl",           # ByWork系
        "./../data/OpenSubtitles_sample_40000.jsonl",  # RandomSpan系
        "./../data/TED_sample_40000.jsonl",            # RandomSpan系
        "./../data/Tatoeba_sample_40000.jsonl",        # RandomSpan系
        "./../data/all_outenjp.jsonl"                  # RandomSpan系 
    ]
   
    MODEL_NAME = "Helsinki-NLP/opus-mt-en-jap"
    SAVE_DIR = "./models/translation_model_balanced"
    
    model, tokenizer = train_model(
        MODEL_NAME,
        files,
        epochs=2,
        batch_size=16,
        max_samples_per_span_file=40000,  # RandomSpan系は各ファイル40000件まで
        save_dir=SAVE_DIR
    )
    
    test_sentences = ["I like apples.", "How are you?", "Machine learning is fun."]
    results = batch_translate(model, tokenizer, test_sentences)
    for en, ja in zip(test_sentences, results):
        print(f"EN: {en} -> JA: {ja}")
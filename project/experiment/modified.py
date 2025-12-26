import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import os
from pathlib import Path
import json

# ===============================
# 1. データ読み込み (JSONL対応)
# ===============================
def load_datasets(file_paths, max_samples=None):
    """複数JSONLデータセットを読み込み、英語と日本語のペアを返す"""
    en_list, ja_list = [], []
    total_loaded = 0

    for path in file_paths:
        print(f"📖 Loading {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(path)}", unit=" lines"):
                try:
                    data = json.loads(line)
                    en, ja = data.get("en"), data.get("ja")
                    if en and ja:
                        en_list.append(en)
                        ja_list.append(ja)
                        total_loaded += 1
                        if max_samples and total_loaded >= max_samples:
                            print(f"⚡ Reached max_samples={max_samples}")
                            return en_list, ja_list
                except json.JSONDecodeError:
                    continue

    if len(en_list) == 0:
        raise ValueError("No valid data loaded. Check your JSONL files.")
    
    return en_list, ja_list

# ===============================
# 2. Dataset クラス
# ===============================
# ===============================
# TranslationDataset (変更箇所のみ)
# ===============================

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, en_list, ja_list, tokenizer, max_len=64, max_k=20, hist_size=10):
        self.en = en_list
        self.ja = ja_list
        self.tok = tokenizer
        self.max_len = max_len
        self.max_k = max_k
        self.add_prefix = hasattr(tokenizer, 'supported_language_codes')

        # 重複抑制用の履歴（idx ごとに最近使った区間を記録）
        self.recent_intervals = {i: [] for i in range(len(en_list))}
        self.hist_size = hist_size

    def __getitem__(self, idx):
        L = len(self.en)
        tried = self.recent_intervals[idx]

        # 最近と被らない区間を選ぶ（最大10回試す）
        for _ in range(10):
            k = random.randint(1, self.max_k)

            left = max(0, idx - random.randint(0, k))
            right = min(L, idx + random.randint(1, k + 1))

            cand = (left, right)
            if cand not in tried:
                break

        # 履歴更新
        tried.append(cand)
        if len(tried) > self.hist_size:
            tried.pop(0)

        # 結合（区切り情報無しでそのまま）
        src = "".join(self.en[left:right])
        tgt = "".join(self.ja[left:right])

        # 以下は元コードと同じ
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
    max_samples=None,
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
    
    en_list, ja_list = load_datasets(file_paths, max_samples=max_samples)
    print(f"✅ Loaded {len(en_list)} translation pairs")
    
    dataset = TranslationDataset(en_list, ja_list, tokenizer, max_len=max_len)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
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
        "./data/OpenSubtitles_sample_40000.jsonl",
        "./data/TED_sample_40000.jsonl",
        "./data/Tatoeba_sample_40000.jsonl"
    ]
    MODEL_NAME = "Helsinki-NLP/opus-mt-en-jap"
    SAVE_DIR = "./models/translation_model_jsonl"
    
    model, tokenizer = train_model(
        MODEL_NAME,
        files,
        epochs=2,
        batch_size=16,
        max_samples=120000,  # 3ファイル x 40000
        save_dir=SAVE_DIR
    )
    
    test_sentences = ["I like apples.", "How are you?", "Machine learning is fun."]
    results = batch_translate(model, tokenizer, test_sentences)
    for en, ja in zip(test_sentences, results):
        print(f"EN: {en} -> JA: {ja}")

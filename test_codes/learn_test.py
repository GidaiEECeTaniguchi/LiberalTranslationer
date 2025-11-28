
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

# ===============================
# 1. データ読み込み
# ===============================
def load_dataset(path):
    import os
    
    # ファイルサイズを取得して行数を推定
    file_size = os.path.getsize(path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"📂 File size: {file_size_mb:.2f} MB")
    print("📖 Reading dataset...")
    
    en, ja = [], []
    
    # まず行数をカウント(プログレスバー用)
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    # データを読み込み
    with open(path, "r", encoding="utf-8") as f:
        pbar = tqdm(f, total=total_lines, desc="Loading data", 
                    ncols=100, unit=" lines")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            try:
                e, j = line.split("\t")
                en.append(e)
                ja.append(j)
            except ValueError:
                # タブ区切りでない行はスキップ
                continue
            
            # プログレスバーに現在の件数を表示
            if len(en) % 1000 == 0:
                pbar.set_postfix({"pairs": len(en)})
    
    return en, ja

# ===============================
# 2. Dataset クラス
# ===============================
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, en_list, ja_list, tokenizer, max_len=64):
        self.en = en_list
        self.ja = ja_list
        self.tok = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.en)
    
    def __getitem__(self, idx):
        src = self.en[idx]
        tgt = self.ja[idx]
        src_tok = self.tok(src, max_length=self.max_len, truncation=True,
                           padding="max_length", return_tensors="pt")
        tgt_tok = self.tok(tgt, max_length=self.max_len, truncation=True,
                           padding="max_length", return_tensors="pt")
        labels = tgt_tok["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": src_tok["input_ids"].squeeze(),
            "attention_mask": src_tok["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

# ===============================
# 3. 学習コード(可視化+Mixed Precision)
# ===============================
def train_model(model_name, data_path, epochs=3, batch_size=8, use_amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"🚀 Using device: {device}")
    if use_amp and device.type == "cuda":
        print("⚡ Mixed Precision Training: ENABLED")
    print("=" * 60)
    
    # モデルの読込
    print("\n📦 Loading model and tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    # データ読み込み
    print(f"\n📚 Loading dataset from {data_path}...")
    en_list, ja_list = load_dataset(data_path)
    print(f"✅ Loaded {len(en_list)} translation pairs")
    
    dataset = TranslationDataset(en_list, ja_list, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Mixed Precision用のスケーラー
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    
    # 学習ループ
    print(f"\n🎯 Starting training: {epochs} epochs, {len(loader)} batches per epoch")
    print("=" * 60)
    
    total_start = time.time()
    history = {"epoch": [], "loss": [], "time": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        # tqdmでプログレスバー表示
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", 
                    ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Mixed Precision Training
            if scaler:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # プログレスバーに現在のlossを表示
            pbar.set_postfix({"loss": f"{loss.item():.4f}", 
                              "avg_loss": f"{total_loss/(batch_idx+1):.4f}"})
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        
        # 履歴を保存
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["time"].append(epoch_time)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Time: {epoch_time:.2f}s ({epoch_time/60:.2f}m)")
        print(f"   Avg time per batch: {epoch_time/len(loader):.3f}s")
        print("-" * 60)
    
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"   Average time per epoch: {total_time/epochs:.2f}s")
    print("=" * 60)
    
    # 学習履歴の表示
    print("\n📈 Training History:")
    for i, (ep, loss, t) in enumerate(zip(history["epoch"], history["loss"], history["time"])):
        print(f"   Epoch {ep}: Loss={loss:.4f}, Time={t:.2f}s")
    
    return model, tokenizer

# ===============================
# 4. 翻訳関数
# ===============================
def translate(model, tokenizer, text):
    device = next(model.parameters()).device
    tok = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(**tok, max_length=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ===============================
# 実行
# ===============================
if __name__ == "__main__":
    MODEL = "Helsinki-NLP/opus-mt-en-jap"
    DATA = "train.txt"  # ← ここをあなたのデータファイル名にする
    
    model, tokenizer = train_model(
        MODEL, 
        DATA, 
        epochs=3, 
        batch_size=8,
        use_amp=True  # Mixed Precision Training を使用
    )
    
    # テスト
    print("\n" + "=" * 60)
    print("🧪 Translation Test")
    print("=" * 60)
    test_sentences = [
        "I like apples.",
        "How are you?",
        "Good morning."
    ]
    
    for sent in test_sentences:
        result = translate(model, tokenizer, sent)
        print(f"EN: {sent}")
        print(f"JA: {result}")
        print("-" * 60)

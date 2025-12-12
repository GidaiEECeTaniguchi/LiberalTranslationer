import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import os
import re
from pathlib import Path
import json

# ===============================
# 1. データ読み込み
# ===============================
def load_dataset(path, max_samples=None):
    """データセットを読み込み、英語と日本語のペアを返す"""
    file_size = os.path.getsize(path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"📂 File size: {file_size_mb:.2f} MB")
    print("📖 Reading dataset...")
    
    en, ja = [], []
    
    # 行数をカウント
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
                # タブまたは複数のスペースで分割
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = re.split(r'\s{2,}', line)
                
                if len(parts) >= 2:
                    e = parts[0].strip()
                    j = parts[1].strip()
                    if e and j:
                        en.append(e)
                        ja.append(j)
                        
                        # max_samplesに達したら終了
                        if max_samples and len(en) >= max_samples:
                            break
            except Exception:
                continue
            
            if len(en) % 1000 == 0:
                pbar.set_postfix({"pairs": len(en)})
    
    return en, ja

# ===============================
# 2. Dataset クラス (修正版)
# ===============================
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, en_list, ja_list, tokenizer, max_len=128):
        self.en = en_list
        self.ja = ja_list
        self.tok = tokenizer
        self.max_len = max_len
        
        # モデルがMarianMTの場合、プレフィックスを追加
        self.add_prefix = hasattr(tokenizer, 'supported_language_codes')
    
    def __len__(self):
        return len(self.en)
    
    def __getitem__(self, idx):
        src = self.en[idx]
        tgt = self.ja[idx]
        
        # MarianMTモデルの場合、ソース言語にプレフィックスを追加
        if self.add_prefix:
            src = ">>jap<< " + src
        
        # ソースのトークン化
        src_tok = self.tok(
            src, 
            max_length=self.max_len, 
            truncation=True,
            padding="max_length", 
            return_tensors="pt"
        )
        
        # ターゲットのトークン化 (text_targetを使用)
        tgt_tok = self.tok(
            text_target=tgt,  # text_targetパラメータを使用
            max_length=self.max_len, 
            truncation=True,
            padding="max_length", 
            return_tensors="pt"
        )
        
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
    """検証データでモデルを評価"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
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
            print(f"   ⚠️ No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ===============================
# 5. 学習コード
# ===============================
def train_model(
    model_name, 
    data_path, 
    epochs=10, 
    batch_size=16,  # バッチサイズを小さく
    use_amp=True, 
    max_samples=50000,  # サンプル数を減らす
    val_split=0.1,
    save_dir="./models",
    learning_rate=5e-5,  # 学習率を下げる
    gradient_clip=1.0,
    save_every=1,
    patience=3,  # Early stopping
    max_len=128  # シーケンス長を長く
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
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
    print(f"   Model type: {model.config.model_type}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # トークナイザー情報
    print(f"\n📝 Tokenizer info:")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, 'supported_language_codes'):
        print(f"   Supported languages: {tokenizer.supported_language_codes}")
        print(f"   ⚠️ Remember to add '>>jap<<' prefix to source text")
    
    # データ読み込み
    print(f"\n📚 Loading dataset from {data_path}...")
    en_list, ja_list = load_dataset(data_path, max_samples=max_samples)
    print(f"✅ Loaded {len(en_list):,} translation pairs")
    
    # データの方向を確認
    print(f"\n🔍 Checking data direction...")
    print(f"First 3 samples:")
    for i in range(min(3, len(en_list))):
        print(f"  EN: {en_list[i][:60]}...")
        print(f"  JA: {ja_list[i][:60]}...")
        print()
    
    # データセット作成
    full_dataset = TranslationDataset(en_list, ja_list, tokenizer, max_len=max_len)
    
    # 訓練/検証分割
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_dataset):,} pairs")
    print(f"   Validation: {len(val_dataset):,} pairs")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # 2→4に増やす
        pin_memory=True,
        prefetch_factor=2,  # 事前読み込み
        persistent_workers=True  # ワーカーを維持
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # 検証時はバッチサイズを2倍に
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Gradient Accumulation用の変数
    accumulation_steps = 2  # 2ステップ分の勾配を蓄積してから更新
    
    # 学習率スケジューラ (実際のステップ数を調整)
    total_steps = (len(train_loader) // accumulation_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Mixed Precision用のスケーラー
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # 保存ディレクトリ作成
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 学習ループ
    print(f"\n🎯 Starting training")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max sequence length: {max_len}")
    print(f"   Batches per epoch: {len(train_loader)}")
    print(f"   Early stopping patience: {patience}")
    print("=" * 60)
    
    total_start = time.time()
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "time": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                    ncols=120, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Mixed Precision Training
            if scaler:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps  # 勾配蓄積のためにlossを割る
                
                scaler.scale(loss).backward()
                
                # 勾配蓄積: accumulation_stepsごとに更新
                if (batch_idx + 1) % accumulation_steps == 0:
                    if gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()  # optimizerの後に呼ぶ
                    optimizer.zero_grad()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
                # 勾配蓄積: accumulation_stepsごとに更新
                if (batch_idx + 1) % accumulation_steps == 0:
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    
                    optimizer.step()
                    scheduler.step()  # optimizerの後に呼ぶ
                    optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # 表示用に戻す
            
            # プログレスバーに現在のlossを表示
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # エポック終了後の処理
        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / len(train_loader)
        
        # 検証
        print(f"\n📊 Evaluating on validation set...")
        val_loss = evaluate_model(model, val_loader, device)
        
        # 履歴を保存
        current_lr = scheduler.get_last_lr()[0]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        history["time"].append(epoch_time)
        
        # 損失の差を計算
        loss_gap = abs(avg_train_loss - val_loss)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Gap: {loss_gap:.4f} {'⚠️ OVERFITTING!' if loss_gap > 0.15 else '✅'}")
        print(f"   Learning Rate: {current_lr:.2e}")
        print(f"   Time: {epoch_time:.2f}s ({epoch_time/60:.2f}m)")
        print(f"   Avg time per batch: {epoch_time/len(train_loader):.3f}s")
        
        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement = (best_val_loss - val_loss) / best_val_loss * 100 if epoch > 0 else 0
            print(f"   ⭐ New best validation loss! (improved by {improvement:.2f}%)")
            model.save_pretrained(os.path.join(save_dir, "best_model"), safe_serialization=True)
            tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
        
        # 定期保存
        if (epoch + 1) % save_every == 0:
            checkpoint_dir = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"   💾 Checkpoint saved to {checkpoint_dir}")
        
        # Early Stopping チェック
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
    
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"   Average time per epoch: {total_time/len(history['epoch']):.2f}s")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    # 学習履歴の保存
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Training history saved to {history_path}")
    
    # 学習履歴の表示
    print("\n📈 Training History:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Gap':<10} {'LR':<12} {'Time':<10}")
    print("-" * 70)
    for ep, train_l, val_l, lr, t in zip(
        history["epoch"], 
        history["train_loss"], 
        history["val_loss"],
        history["learning_rate"],
        history["time"]
    ):
        gap = abs(train_l - val_l)
        print(f"{ep:<8} {train_l:<12.4f} {val_l:<12.4f} {gap:<10.4f} {lr:<12.2e} {t:<10.2f}s")
    
    # ベストモデルを読み込んで返す
    print(f"\n📦 Loading best model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        os.path.join(save_dir, "best_model"), 
        use_safetensors=True
    ).to(device)
    
    return model, tokenizer, history

# ===============================
# 6. 翻訳関数 (修正版)
# ===============================
def translate(model, tokenizer, text, max_length=128, num_beams=5, device=None):
    """テキストを翻訳 (MarianMT対応)"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # MarianMTモデルの場合、プレフィックスを追加
    if hasattr(tokenizer, 'supported_language_codes'):
        text = ">>jap<< " + text
    
    # 入力をトークン化
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            repetition_penalty=1.2,
        )
    
    # デコード
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# ===============================
# 7. バッチ翻訳
# ===============================
def batch_translate(model, tokenizer, texts, batch_size=8, max_length=128, num_beams=5):
    """複数のテキストをバッチで翻訳"""
    device = next(model.parameters()).device
    model.eval()
    
    # MarianMTモデルの場合、プレフィックスを追加
    if hasattr(tokenizer, 'supported_language_codes'):
        texts = [">>jap<< " + text for text in texts]
    
    translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.2,
            )
        
        batch_translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        translations.extend(batch_translations)
    
    return translations

# ===============================
# 実行例
# ===============================
if __name__ == "__main__":
    # 設定
    MODEL = "Helsinki-NLP/opus-mt-en-jap"
    DATA = "./transformer/data/jesc/raw"
    SAVE_DIR = "./transformer/models/translation_model_v3"
    
    # 学習
    model, tokenizer, history = train_model(
        MODEL, 
        DATA, 
        epochs=3,  # エポック数を3に減らす
        batch_size=32,  # バッチサイズを32に戻す(64は遅い)
        use_amp=True,
        max_samples=50000,  # 5万ペアに減らす
        val_split=0.05,  # 検証データを5%に減らす(高速化)
        save_dir=SAVE_DIR,
        learning_rate=1e-4,  # 学習率を少し上げる
        gradient_clip=1.0,
        save_every=1,
        patience=2,  # Early stoppingを2エポックに
        max_len=64  # シーケンス長を短く(128→64)で高速化
    )
    
    # テスト
    print("\n" + "=" * 60)
    print("🧪 Translation Test")
    print("=" * 60)
    
    test_sentences = [
        "I like apples.",
        "How are you?",
        "Good morning.",
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today.",
        "I am studying Japanese.",
        "Thank you very much."
    ]
    
    print("\n🔤 Single translations:")
    for sent in test_sentences:
        result = translate(model, tokenizer, sent)
        print(f"EN: {sent}")
        print(f"JA: {result}")
        print("-" * 60)
    
    print("\n🔤 Batch translations:")
    batch_results = batch_translate(model, tokenizer, test_sentences)
    for sent, result in zip(test_sentences, batch_results):
        print(f"EN: {sent}")
        print(f"JA: {result}")
        print("-" * 60)
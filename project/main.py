import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import json
import time

# ===============================
# 1. データ読み込み
# ===============================
def load_datasets(file_paths, max_samples=None):
    en_list, ja_list = [], []

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
                        if max_samples and len(en_list) >= max_samples:
                            print(f"⚡ Reached max_samples={max_samples}")
                            return en_list, ja_list
                except json.JSONDecodeError:
                    continue
    return en_list, ja_list

# ===============================
# 2. Dataset
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
        src = self.en_list[idx]
        tgt = self.ja_list[idx]
        src_tok = self.tokenizer(src, truncation=True, padding="max_length",
                                 max_length=self.max_len, return_tensors="pt")
        tgt_tok = self.tokenizer(tgt, truncation=True, padding="max_length",
                                 max_length=self.max_len, return_tensors="pt")
        labels = tgt_tok["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_tok["input_ids"].squeeze(),
            "attention_mask": src_tok["attention_mask"].squeeze(),
            "labels": labels.squeeze()
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
# 4. 学習関数
# ===============================
def train_model(
    model_name,
    file_paths,
    epochs=3,
    batch_size=32,
    use_amp=True,
    max_samples=50000,
    val_split=0.1,
    save_dir="models/translation_model",
    learning_rate=1e-4,
    gradient_clip=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)

    # データ読み込み
    en_list, ja_list = load_datasets(file_paths, max_samples=max_samples)
    print(f"✅ Total pairs loaded: {len(en_list)}")

    dataset = TranslationDataset(en_list, ja_list, tokenizer)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/(pbar.n+1):.4f}"})

        val_loss = evaluate_model(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} validation loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(save_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
            print(f"⭐ New best model saved at epoch {epoch+1}")

    return model, tokenizer

# ===============================
# 5. 翻訳関数
# ===============================
def translate_text(model, tokenizer, texts, max_length=64, num_beams=4):
    device = next(model.parameters()).device
    model.eval()
    translations = []
    for i in range(0, len(texts), 8):
        batch_texts = texts[i:i+8]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        translations.extend([tokenizer.decode(o, skip_special_tokens=True) for o in outputs])
    return translations

# ===============================
# 6. メイン
# ===============================
if __name__ == "__main__":
    files = [
        "data/OpenSubtitles_40000.jsonl",
        "data/TED_sample_40000.jsonl",
        "data/Tatoeba_sample_40000.jsonl"
    ]
    MODEL_NAME = "Helsinki-NLP/opus-mt-en-jap"

    # 学習
    model, tokenizer = train_model(
        MODEL_NAME,
        files,
        epochs=2,          # 軽量テスト用に2エポック
        batch_size=16,
        max_samples=10000,  # 軽量化
        learning_rate=1e-4
    )

    # テスト翻訳
    test_sentences = [
        "I like apples.",
        "How are you?",
        "Good morning.",
        "This is a test sentence.",
        "Machine learning is fascinating."
    ]
    results = translate_text(model, tokenizer, test_sentences)
    print("\n=== Translation Test ===")
    for en, ja in zip(test_sentences, results):
        print(f"EN: {en}")
        print(f"JA: {ja}")
        print("-"*40)

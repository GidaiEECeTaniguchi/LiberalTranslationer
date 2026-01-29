import json
import random
import logging
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
import os
logger = logging.getLogger(__name__)

# ===============================
# 1. Dataset クラス群
# ===============================

class TranslationDatasetBase(Dataset):
    def __init__(self, en_texts, ja_texts, tokenizer, max_len=64):
        self.en_texts = en_texts
        self.ja_texts = ja_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.en_texts)

    def _get_tokenized_pair(self, en, ja):
        # 特定のモデル用プレフィックス（Helsinki-NLP / mBART等）
        if hasattr(self.tokenizer, 'supported_language_codes'):
            en =  en
        
        inputs = self.tokenizer(en, max_length=self.max_len, truncation=True, padding=False)
        labels = self.tokenizer(ja, max_length=self.max_len, truncation=True, padding=False)
        # === 🚑 緊急デバッグ用コード (ここに追加！) ===
        # 1%の確率で中身を暴露する
        #import random
        #if random.random() < 0.01:
        #    print(f"\n[DEBUG] Raw EN: '{en}'")
        #    print(f"[DEBUG] Raw JA: '{ja}'")
        #    print(f"[DEBUG] Tokenized Labels: {labels['input_ids']}")
            # もしラベルが [EOS] (例: [1] や [2] だけ) なら、それが犯人
        # ==========================================
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

class TranslationDatasetRandomSpan(TranslationDatasetBase):
    def __init__(self, en_texts, ja_texts, tokenizer, max_len=64, multi_prob=0.5):
        super().__init__(en_texts, ja_texts, tokenizer, max_len)
        self.multi_prob = multi_prob

    def __getitem__(self, idx):
        en, ja = self.en_texts[idx], self.ja_texts[idx]
        # マルチセンテンス化（文脈を持たせる）
        if random.random() < self.multi_prob and idx + 1 < len(self.en_texts):
            en = f"{en} {self.en_texts[idx+1]}"
            ja = f"{ja} {self.ja_texts[idx+1]}"
        return self._get_tokenized_pair(en, ja)

class TranslationDatasetByWork(TranslationDatasetBase):
    def __getitem__(self, idx):
        return self._get_tokenized_pair(self.en_texts[idx], self.ja_texts[idx])

# ===============================
# 2. データロード・ユーティリティ
# ===============================

def is_chunk_delimiter(text):
    return any(d in text for d in ["%%%%%%%%THISWORKENDSHERE%%%%%%%%", "%%%%%%%%この作品ここまで%%%%%%%%"])

# data_utils.py (修正版)

def generate_mock_data(num_samples=50):
    """完全にランダムな翻訳ペアを生成"""
    en_samples = [f"This is mock english sentence {i}." for i in range(num_samples)]
    ja_samples = [f"これはモックの日本語文章 {i} です。" for i in range(num_samples)]
    return en_samples, ja_samples




def load_jsonl(file_path, tag=None, max_samples=None):
    en_list, ja_list = [], []
    logger.info(f"📖 Reading: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # max_samples で絞り込み
    if max_samples and len(lines) > max_samples:
        lines = random.sample(lines, max_samples)
            
    # ★ tqdm でプログレスバーを表示 (長い読み込みでも安心)
    for line in tqdm(lines, desc=f"Loading {os.path.basename(file_path)}"):
        try:
            data = json.loads(line)
            en, ja = data.get("en"), data.get("ja")
            if en and ja and not is_chunk_delimiter(en):
                en_list.append(f"{tag} {en}" if tag else en)
                ja_list.append(ja)
        except: continue
            
    return en_list, ja_list

def load_chunks(file_path, tag=None):
    chunks_en, chunks_ja = [], []
    curr_en, curr_ja = [], []
    
    logger.info(f"chunks loading: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        # chunkは行数不明なことが多いので単純ループ、あるいは tqdm(f) も可
        for line in f:
            try:
                data = json.loads(line)
                en, ja = data.get("en"), data.get("ja")
                if is_chunk_delimiter(en):
                    if curr_en:
                        chunks_en.append(f"{tag} {' '.join(curr_en)}" if tag else ' '.join(curr_en))
                        chunks_ja.append(' '.join(curr_ja))
                        curr_en, curr_ja = [], []
                elif en and ja:
                    curr_en.append(en); curr_ja.append(ja)
            except: continue

    if curr_en:
        chunks_en.append(f"{tag} {' '.join(curr_en)}" if tag else ' '.join(curr_en))
        chunks_ja.append(' '.join(curr_ja))

    return chunks_en, chunks_ja

# ===============================
# 3. DataLoader 生成 (3-Phase対応)
# ===============================

def create_dataloaders(config, tokenizer):
    """ファイルを読み込み、Train/Valに分割してLoaderを返す"""
    
    # --- Mock Mode (既存のまま) ---
    if config.mock_mode:
        first_file = config.file_paths[0] if config.file_paths else None
        if first_file and os.path.exists(first_file):
            logger.info(f"🎭 LIGHT MOCK: Sampling 20 real lines from {first_file}...")
            en, ja = load_jsonl(first_file, max_samples=20)
        else:
            logger.info("🎭 DUMMY MOCK: No files found, using synthetic data...")
            en = ["I love you.", "Who are you?", "This is a pen."] * 7
            ja = ["私はあなたを愛しています。", "あなたは誰ですか？", "これはペンです。"] * 7
        
        mock_ds = TranslationDatasetRandomSpan(en, ja, tokenizer, max_len=config.max_len)
        loader = DataLoader(mock_ds, batch_size=4, shuffle=True, 
                          collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True))
        # Mockの場合は全部同じLoaderを使い回す
        return {k: loader for k in ["span", "bywork", "chunk", "practical_line", "practical_chunk", "val"]}
    
    # --- 本番ロード ---
    loaders_map = {"span": None, "bywork": None, "chunk": None, "practical_line": None, "practical_chunk": None}
    span_datasets = [] 
    
    # Dataset作成ループ
    for path, ftype in zip(config.file_paths, config.file_types):
        if ftype == 0:
            en, ja = load_jsonl(path, max_samples=config.max_samples_per_span_file)
            ds = TranslationDatasetRandomSpan(en, ja, tokenizer, max_len=config.max_len)
            # ★ 修正2: 上書きせずに追加する
            span_datasets.append(ds)
        
        elif ftype == 1:
            # ここは書き換え済み（20000）のはず
            en, ja = load_jsonl(path, max_samples=20000) 
            loaders_map["bywork"] = TranslationDatasetByWork(en, ja, tokenizer, config.max_len)
            cen, cja = load_chunks(path)
            loaders_map["chunk"] = TranslationDatasetByWork(cen, cja, tokenizer, config.max_len * 4)

        elif ftype == 2:
            en, ja = load_jsonl(path, max_samples=350)
            factor = getattr(config, 'practical_upsample', 2)
            loaders_map["practical_line"] = ConcatDataset([TranslationDatasetByWork(en, ja, tokenizer, config.max_len)] * factor)
            
            cen, cja = load_chunks(path)
            loaders_map["practical_chunk"] = ConcatDataset([TranslationDatasetByWork(cen, cja, tokenizer, config.max_len * 4)] * factor)

    # ★ 修正3: ループを抜けたら、貯めたspanデータを合体させる
    if span_datasets:
        loaders_map["span"] = ConcatDataset(span_datasets)
        logger.info(f"📚 Combined {len(span_datasets)} span datasets into one.")
    # --- ★ Train/Val 分割と DataLoader 化 ---
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    final_loaders = {}
    val_datasets = [] # バリデーション用データセットをここに集める

    for key, ds in loaders_map.items():
        if ds:
            # データセット全体の長さ
            full_len = len(ds)
            # 5% をバリデーションにする（ただし最低1個は確保、データが少なすぎる場合は分割しない）
            val_len = int(full_len * 0.05)
            if val_len < 1 and full_len > 1: val_len = 1
            train_len = full_len - val_len

            if val_len > 0:
                # ★ ここで random_split を使用！
                train_ds, val_ds = random_split(ds, [train_len, val_len])
                final_loaders[key] = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
                val_datasets.append(val_ds)
            else:
                # データが少なすぎる場合は全部Train
                final_loaders[key] = DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=collator)

    # 集めたバリデーションデータを結合して1つのLoaderにする
    if val_datasets:
        combined_val_ds = ConcatDataset(val_datasets)
        final_loaders["val"] = DataLoader(combined_val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
        logger.info(f"📊 Validation Dataset Created: {len(combined_val_ds)} samples")
    else:
        logger.warning("⚠️ No validation dataset created (data might be too small).")

    return final_loaders
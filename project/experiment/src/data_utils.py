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
            en = ">>jap<< " + en
        
        inputs = self.tokenizer(en, max_length=self.max_len, truncation=True, padding=False)
        labels = self.tokenizer(ja, max_length=self.max_len, truncation=True, padding=False)
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
        if max_samples and len(lines) > max_samples:
            lines = random.sample(lines, max_samples)
            
        for line in lines:
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
    
    with open(file_path, "r", encoding="utf-8") as f:
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

    # ✨ ループ終了後、残っているデータがあれば追加する
    if curr_en:
        chunks_en.append(f"{tag} {' '.join(curr_en)}" if tag else ' '.join(curr_en))
        chunks_ja.append(' '.join(curr_ja))

    return chunks_en, chunks_ja

# ===============================
# 3. DataLoader 生成 (3-Phase対応)
# ===============================

def create_dataloaders(config, tokenizer):
    """ファイルを読み込み、フェーズごとのLoaderをマッピングして返す"""
    if config.mock_mode:
        # ファイルが存在するかチェック
        first_file = config.file_paths[0] if config.file_paths else None
        
        if first_file and os.path.exists(first_file):
            logger.info(f"🎭 LIGHT MOCK: Sampling 20 real lines from {first_file}...")
            # 最初のファイルから20行だけ本物を借りてくる
            en, ja = load_jsonl(first_file, max_samples=20)
        else:
            logger.info("🎭 DUMMY MOCK: No files found, using synthetic data...")
            en = ["I love you.", "Who are you?", "This is a pen."] * 7
            ja = ["私はあなたを愛しています。", "あなたは誰ですか？", "これはペンです。"] * 7
        
        mock_ds = TranslationDatasetRandomSpan(en, ja, tokenizer, max_len=config.max_len)
        loader = DataLoader(mock_ds, batch_size=4, shuffle=True, 
                          collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True))
        
        return {k: loader for k in ["span", "bywork", "chunk", "practical_line", "practical_chunk", "val"]}
    
    loaders_map = {"span": None, "bywork": None, "chunk": None, "practical_line": None, "practical_chunk": None}
    all_val_datasets = []

    for path, ftype in zip(config.file_paths, config.file_types):
        # 0: Span, 1: ByWork, 2: Practical
        if ftype == 0:
            en, ja = load_jsonl(path, max_samples=config.max_samples_per_span_file)
            ds = TranslationDatasetRandomSpan(en, ja, tokenizer, max_len=config.max_len)
            loaders_map["span"] = ds # 後でLoader化
        
        elif ftype == 1:
            en, ja = load_jsonl(path)
            loaders_map["bywork"] = TranslationDatasetByWork(en, ja, tokenizer, config.max_len)
            cen, cja = load_chunks(path)
            loaders_map["chunk"] = TranslationDatasetByWork(cen, cja, tokenizer, config.max_len * 4)

        elif ftype == 2:
            en, ja = load_jsonl(path)
            # 20倍は多すぎたので、configで制御可能にするか5倍程度に抑える
            factor = getattr(config, 'practical_upsample', 2)
            loaders_map["practical_line"] = ConcatDataset([TranslationDatasetByWork(en, ja, tokenizer, config.max_len)] * factor)
            
            cen, cja = load_chunks(path)
            loaders_map["practical_chunk"] = ConcatDataset([TranslationDatasetByWork(cen, cja, tokenizer, config.max_len * 4)] * factor)

    # 各 Dataset を DataLoader に変換 (簡易化のためここでは一括処理)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    
    final_loaders = {}
    for key, ds in loaders_map.items():
        if ds:
            # ここで Train/Val 分割を入れるのが理想的
            final_loaders[key] = DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=collator)

    return final_loaders
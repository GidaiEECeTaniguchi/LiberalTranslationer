import json
from tqdm import tqdm
import os

def load_datasets(file_paths, max_samples=None):
    """複数 JSONL データセットをまとめて読み込む"""
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


if __name__ == "__main__":
    # data/ フォルダ内の JSONL ファイルを指定
    files = [
        "data/OpenSubtitles_sample_40000.jsonl",
        "data/TED_sample_40000.jsonl",
        "data/Tatoeba_sample_40000.jsonl"
    ]

    # 最大読み込み件数（任意）
    max_samples = None  # すべて読み込みたい場合は None

    en_list, ja_list = load_datasets(files, max_samples=max_samples)
    print(f"\n✅ Total loaded examples: {len(en_list)}")

    # 読み込めたデータの確認
    for i in range(min(5, len(en_list))):
        print(f"EN: {en_list[i]}")
        print(f"JA: {ja_list[i]}")
        print("-" * 40)

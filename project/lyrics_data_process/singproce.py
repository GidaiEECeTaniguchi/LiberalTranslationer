import json
import re

def parse_sing_txt_to_jsonl(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 最初の行はタイトル（スキップ）
    content_lines = lines[1:]  # タイトル行を除外

    # 英語と日本語のチャンクを抽出
    chunks = []
    current_chunk = {"en": "", "ja": ""}
    current_lang = None

    for line in content_lines:
        # 全角スペースを半角スペースに変換
        line = line.replace('\u3000', ' ')
        
        # 英語か日本語かを判別
        if line.isascii():
            # 英語行
            if current_lang == "ja":
                # 日本語チャンクが終了、新しい英語チャンク開始
                chunks.append(current_chunk)
                current_chunk = {"en": "", "ja": ""}
            current_chunk["en"] += line + " "
            current_lang = "en"
        else:
            # 日本語行
            if current_lang == "en":
                # 英語チャンクが終了、新しい日本語チャンク開始
                pass
            current_chunk["ja"] += line + " "
            current_lang = "ja"

    # 最後のチャンクを追加
    if current_chunk["en"] or current_chunk["ja"]:
        chunks.append(current_chunk)

    # 前後の空白を削除し、連続するスペースを1つに正規化
    for chunk in chunks:
        chunk["en"] = re.sub(r'\s+', ' ', chunk["en"].strip())
        chunk["ja"] = re.sub(r'\s+', ' ', chunk["ja"].strip())

    # JSONL形式で出力
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"変換完了: {len(chunks)} 件の対訳を {output_file_path} に保存しました。")
    return chunks

# 使用例
input_file = "./lyrics_rawdata/Sing.txt"
output_file = "sing_training.jsonl"

parsed_chunks = parse_sing_txt_to_jsonl(input_file, output_file)

# 確認のためすべてのチャンクを表示
for i, chunk in enumerate(parsed_chunks):
    print(f"チャンク {i+1}: {chunk}")
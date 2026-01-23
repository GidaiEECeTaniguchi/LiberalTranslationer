import json
import re
import os
from pathlib import Path

def parse_sing_txt_to_jsonl(input_file_path, output_file_path):
    """単一の歌詞ファイルをJSONL形式に変換"""
    try:
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

        # 空のチャンクを除外
        chunks = [chunk for chunk in chunks if chunk["en"] and chunk["ja"]]

        # JSONL形式で出力
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        return len(chunks), True
    except Exception as e:
        print(f"  エラー: {e}")
        return 0, False

def process_all_lyrics_files():
    """すべての歌詞ファイルを処理"""
    raw_data_dir = Path("lyrics_rawdata")
    json_data_dir = Path("lyrics_jsondata")
    
    # 出力ディレクトリを作成（存在しない場合）
    json_data_dir.mkdir(exist_ok=True)
    
    # 入力ディレクトリの存在確認
    if not raw_data_dir.exists():
        print(f"エラー: {raw_data_dir} ディレクトリが見つかりません。")
        return
    
    # すべてのtxtファイルを取得
    txt_files = list(raw_data_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"{raw_data_dir} 内に.txtファイルが見つかりません。")
        return
    
    print(f"処理を開始します...")
    print(f"入力ディレクトリ: {raw_data_dir}")
    print(f"出力ディレクトリ: {json_data_dir}")
    print(f"処理対象ファイル数: {len(txt_files)}")
    print("-" * 50)
    
    total_chunks = 0
    success_count = 0
    
    for txt_file in txt_files:
        print(f"処理中: {txt_file.name}")
        
        # 出力ファイル名を決定（拡張子を.jsonlに変更）
        output_filename = txt_file.stem + ".jsonl"
        output_path = json_data_dir / output_filename
        
        # 変換処理
        chunk_count, success = parse_sing_txt_to_jsonl(txt_file, output_path)
        
        if success:
            print(f"  ✓ 完了: {chunk_count} 件の対訳を {output_filename} に保存")
            total_chunks += chunk_count
            success_count += 1
        else:
            print(f"  ✗ 失敗: {txt_file.name}")
        
        print()
    
    print("-" * 50)
    print(f"処理完了:")
    print(f"  成功: {success_count}/{len(txt_files)} ファイル")
    print(f"  失敗: {len(txt_files) - success_count}/{len(txt_files)} ファイル")
    print(f"  総対訳ペア数: {total_chunks}")
    


def combine_all_jsonl_files(json_data_dir, output_path):
    """すべてのJSONLファイルを1つに結合"""
    jsonl_files = list(Path(json_data_dir).glob("*.jsonl"))
    
    # 結合ファイル自体は除外
    jsonl_files = [f for f in jsonl_files if f.name != "all_lyrics.jsonl"]
    
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                outfile.writelines(lines)
                total_lines += len(lines)
    
    return total_lines

# メイン処理の実行
if __name__ == "__main__":
    process_all_lyrics_files()
    
    # サンプル表示（最初の3ファイルの内容を確認）
    print("\n" + "=" * 50)
    print("サンプル確認:")
    
    json_data_dir = Path("lyrics_jsondata")
    if json_data_dir.exists():
        jsonl_files = list(json_data_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files[:3]:  # 最初の3ファイルのみ表示
            print(f"\n{jsonl_file.name}:")
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines[:2]):  # 各ファイルの最初の2行のみ表示
                        data = json.loads(line.strip())
                        print(f"  [対訳 {i+1}]")
                        print(f"    EN: {data['en'][:50]}..." if len(data['en']) > 50 else f"    EN: {data['en']}")
                        print(f"    JA: {data['ja'][:50]}..." if len(data['ja']) > 50 else f"    JA: {data['ja']}")
            except Exception as e:
                print(f"  読み込みエラー: {e}")

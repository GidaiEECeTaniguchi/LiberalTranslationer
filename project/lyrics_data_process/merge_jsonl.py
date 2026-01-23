import os
import glob
import json

def merge_jsonl_files(directory_path, output_file="merged.jsonl", 
                      separator_en="%%%%%%%%THISWORKENDSHERE%%%%%%%%",
                      separator_ja="%%%%%%%%この作品ここまで%%%%%%%%"):
    """
    指定したディレクトリ内のすべてのjsonlファイルを結合し、
    各ファイルの間に区切り行を挿入します。
    
    Args:
        directory_path (str): jsonlファイルが格納されているディレクトリのパス
        output_file (str): 出力ファイル名（デフォルト: "merged.jsonl"）
        separator_en (str): 英語の区切りテキスト
        separator_ja (str): 日本語の区切りテキスト
    """
    
    # ディレクトリ内のすべてのjsonlファイルを取得
    jsonl_files = sorted(glob.glob(os.path.join(directory_path, "*.jsonl")))
    
    if not jsonl_files:
        print(f"ディレクトリ '{directory_path}' にjsonlファイルが見つかりませんでした。")
        return
    
    print(f"{len(jsonl_files)}個のjsonlファイルが見つかりました:")
    for i, file in enumerate(jsonl_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    # 区切り行を作成
    separator = json.dumps({"en": separator_en, "ja": separator_ja}, ensure_ascii=False)
    
    # 出力ファイルを開く
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, jsonl_file in enumerate(jsonl_files, 1):
            print(f"処理中: {os.path.basename(jsonl_file)} ({i}/{len(jsonl_files)})")
            
            # 各jsonlファイルの内容を読み込んで出力
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # 空行をスキップ
                        outfile.write(line + '\n')
            
            # 最後のファイル以外には区切り行を追加
            if i < len(jsonl_files):
                outfile.write(separator + '\n')
    
    print(f"\n完了！結合されたファイルは '{output_file}' に保存されました。")
    print(f"合計で {len(jsonl_files)} 個のファイルが結合され、{len(jsonl_files)-1} 個の区切り行が挿入されました。")

# 使用例
if __name__ == "__main__":
    # ディレクトリパスを指定
    target_directory = input("処理するディレクトリのパスを入力してください: ").strip()
    
    # ディレクトリの存在確認
    if not os.path.isdir(target_directory):
        print(f"エラー: ディレクトリ '{target_directory}' が見つかりません。")
    else:
        # 出力ファイル名を指定（オプション）
        output_name = input("出力ファイル名を入力してください（デフォルト: merged.jsonl）: ").strip()
        if not output_name:
            output_name = "merged.jsonl"
        
        merge_jsonl_files(target_directory, output_name)

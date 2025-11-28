import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# モデルとトークナイザーのロード
print("\nモデルをロード中...")
model_name = "Helsinki-NLP/opus-mt-en-jap"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    use_safetensors=True  # safetensorsフォーマットを使用
).to(device)

print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
print("ロード完了！\n")

# テスト用の英文サンプル
test_sentences = [
    "Hello, how are you?",
    "Machine learning is a subset of artificial intelligence.",
    "The weather is beautiful today.",
    "I love programming and technology.",
    "This is a test of the translation model running on GPU."
]

# 翻訳実行
print("=" * 60)
print("翻訳テスト開始")
print("=" * 60)

total_time = 0

for i, text in enumerate(test_sentences, 1):
    print(f"\n[{i}] 原文: {text}")
    
    # エンコード
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # 翻訳実行（時間計測）
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    elapsed = time.time() - start_time
    total_time += elapsed
    
    # デコード
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"    翻訳: {translation}")
    print(f"    処理時間: {elapsed:.4f}秒")

# バッチ処理テスト
print("\n" + "=" * 60)
print("バッチ処理テスト")
print("=" * 60)

batch_start = time.time()
batch_inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    batch_outputs = model.generate(**batch_inputs, max_length=512)

batch_elapsed = time.time() - batch_start

print(f"\nバッチサイズ: {len(test_sentences)}")
print(f"バッチ処理時間: {batch_elapsed:.4f}秒")
print(f"1文あたり平均: {batch_elapsed/len(test_sentences):.4f}秒")

print("\n" + "=" * 60)
print("統計情報")
print("=" * 60)
print(f"個別処理合計時間: {total_time:.4f}秒")
print(f"個別処理平均時間: {total_time/len(test_sentences):.4f}秒")
print(f"バッチ処理総時間: {batch_elapsed:.4f}秒")
print(f"高速化率: {total_time/batch_elapsed:.2f}x")

if torch.cuda.is_available():
    print(f"\nGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"GPUメモリ予約量: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

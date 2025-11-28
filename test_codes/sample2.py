import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}\n")

# テストする英日翻訳モデル（軽量で実用的）
models_to_test = [
    {
        "name": "staka/fugumt-en-ja",
        "description": "FuguMT - 実用的な英日翻訳（軽量）"
    },
    {
        "name": "Helsinki-NLP/opus-mt-en-jap",
        "description": "OPUS-MT 英→日（聖書特化版）"
    }
]

# テスト用の英文サンプル
test_sentences = [
    "Hello, how are you?",
    "Machine learning is a subset of artificial intelligence.",
    "The weather is beautiful today.",
    "I love programming and technology.",
    "This is a test of the translation model running on GPU."
]

print("=" * 70)
print("複数モデルの翻訳品質比較テスト")
print("=" * 70)

for model_info in models_to_test:
    model_name = model_info["name"]
    description = model_info["description"]
    
    print(f"\n{'=' * 70}")
    print(f"モデル: {description}")
    print(f"ID: {model_name}")
    print(f"{'=' * 70}")
    
    try:
        # モデルのロード
        print("モデルをロード中...")
        start_load = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(device)
        
        load_time = time.time() - start_load
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ ロード完了 ({load_time:.2f}秒)")
        print(f"  パラメータ数: {params:,}")
        print(f"  GPUメモリ: {torch.cuda.memory_allocated()/1024**2:.0f}MB\n")
        
        # 翻訳テスト
        total_time = 0
        for i, text in enumerate(test_sentences, 1):
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)
            elapsed = time.time() - start
            total_time += elapsed
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"[{i}] {text}")
            print(f"    → {translation}")
            print(f"    ({elapsed:.3f}秒)\n")
        
        # バッチ処理テスト
        print("バッチ処理テスト...")
        batch_inputs = tokenizer(
            test_sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        batch_start = time.time()
        with torch.no_grad():
            batch_outputs = model.generate(**batch_inputs, max_length=512)
        batch_time = time.time() - batch_start
        
        print(f"  バッチ処理時間: {batch_time:.3f}秒")
        print(f"  個別処理合計: {total_time:.3f}秒")
        print(f"  高速化率: {total_time/batch_time:.2f}x")
        
        # メモリクリア
        del model, tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ エラー: {str(e)}\n")
        torch.cuda.empty_cache()
        continue

print("\n" + "=" * 70)
print("テスト完了")
print("=" * 70)
print("\n推奨: FuguMTモデルは一般的な英日翻訳に最適です")
print("OPUS-MTは聖書などの特殊なドメインに特化しています")

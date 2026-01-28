import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import sys
import time

def load_model(model_path, use_gpu=True):
    """モデルとトークナイザーを読み込む"""
    print(f"📂 Loading model from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"🔧 Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval() # 推論モードに固定
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def translate(text, model, tokenizer, device, max_length=256, num_beams=5):
    """翻訳を実行する関数"""
    
    # 学習時と同じタグ付け処理（重要）
    # もし学習データで常に ">>jap<< " をつけていたなら、ここでも必須です
    # 不要な場合はこの行をコメントアウトしてください
    #input_text = f">>jap<< {text}" 
    input_text = text
    # トークナイズ
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    ).to(device)

    # 推論（生成）
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            min_length=5,             # 極端に短い回答を禁止
            no_repeat_ngram_size=3,   # 繰り返しを防ぎ、生成を促す
            early_stopping=False,
            do_sample=True,      # 👈 決定論的ではなく、確率的に選ばせる
            top_p=0.9,           # 👈 上位90%の候補から選ぶ
            temperature=0.7,     # 👈 少し柔らかい表現を許可する
        )
        end_time = time.time()

    # デコード
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text, (end_time - start_time)

def main():
    parser = argparse.ArgumentParser(description="Custom Translation Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--text", type=str, help="Single text to translate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()

    # モデルロード
    model, tokenizer, device = load_model(args.model_dir, use_gpu=not args.cpu)

    # 1. 単発実行モード
    if args.text:
        print(f"\n📥 Input: {args.text}")
        result, latency = translate(args.text, model, tokenizer, device)
        print(f"📤 Output: {result}")
        print(f"⏱️ Latency: {latency:.4f} sec")

    # 2. 対話モード（チャット形式）
    elif args.interactive:
        print("\n💬 Interactive Mode (Type 'exit' or 'q' to quit)")
        print("-" * 50)
        while True:
            try:
                user_input = input("EN > ")
                if user_input.lower() in ["exit", "q", "quit"]:
                    break
                if not user_input.strip():
                    continue
                
                result, latency = translate(user_input, model, tokenizer, device)
                print(f"JA > {result}")
                print(f"   (Time: {latency:.4f}s)") # 速度を見たい場合はコメントイン
                print("-" * 20)
            except KeyboardInterrupt:
                break
        print("\nBye!")
    
    else:
        print("Please provide --text 'Your text' or use --interactive")

if __name__ == "__main__":
    main()

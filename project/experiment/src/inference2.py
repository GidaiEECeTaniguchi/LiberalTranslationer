import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, model_path="./models/translation_v2/best_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_path} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def translate(self, text, max_length=128, num_beams=4):
        # モデルごとのプレフィックス対応
        if hasattr(self.tokenizer, 'supported_language_codes'):
            text = ">>jap<< " + text
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 学習済みモデルの場所を指定
    model_dir = "./models/translation_v2/best_model"
    if not os.path.exists(model_dir):
        model_dir = "./models/translation_v2/final_model"

    translator = Translator(model_dir)
    
    print("\n--- Translation Test (Type 'q' to quit) ---")
    test_sentences = [
        "I love you.",
        "Who are you?",
        "This is a beautiful day.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # まずはサンプルでテスト
    for s in test_sentences:
        print(f"EN: {s}")
        print(f"JA: {translator.translate(s)}\n")

    # 自由入力
    while True:
        text = input("Enter English: ")
        if text.lower() == 'q': break
        print(f"JA: {translator.translate(text)}\n")
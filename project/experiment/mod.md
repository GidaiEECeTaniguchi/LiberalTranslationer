# 改造

## 大きな塊での学習  

単純に,
```
src = self.en[idx]
tgt = self.ja[idx]
```
の右辺を弄って、いろいろ入るようにする.


```
    def __getitem__(self, idx):
     
　　    L = len(self.en)

　# 最大で何文つなぐか
　      max_k = 5
　      k = random.randint(1, max_k)

# idx を中心に左右へ広げるが、必ず idx を含む
        left = max(0, idx - random.randint(0, k - 1))
        right = min(L, idx + random.randint(1, k + 1))

    # 単純連結
        src = "".join(self.en[left:right])
        tgt = "".join(self.ja[left:right])
     
        
        if self.add_prefix:
            src = ">>jap<< " + src
        
        src_tok = self.tok(
            src,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        tgt_tok = self.tok(
            text_target=tgt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = tgt_tok["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        
        return {
            "input_ids": src_tok["input_ids"].squeeze(),
            "attention_mask": src_tok["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }
```
被りを削減する必要がある. だが,ここでは与えられたidxに対応するオブジェクト以外の情報を持てない?
(コンストラクタが
    def __init__(self, en_list, ja_list, tokenizer, max_len=64):
        self.en = en_list
        self.ja = ja_list
        self.tok = tokenizer
        self.max_len = max_len
        self.add_prefix = hasattr(tokenizer, 'supported_language_codes')
こうなのでもしかしたら出来るかも?


)
そこで、
dataset = TranslationDataset(en_list, ja_list, tokenizer, max_len=max_len)この処理でなんとか弄る
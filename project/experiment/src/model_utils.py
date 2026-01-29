import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianTokenizer
logger = logging.getLogger(__name__)

# ===============================
# 1. 損失関数: Focal Loss
# ===============================
class FocalLoss(nn.Module):
    """
    難しいサンプル（正解確率が低いもの）に高い重みを置く損失関数
    公式: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets, ignore_index=-100):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        mask = targets_flat != ignore_index
        valid_logits = logits_flat[mask]
        valid_targets = targets_flat[mask]
        
        if valid_logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ===============================
# 2. 正則化: EMA (Exponential Moving Average)
# ===============================
class EMA:
    """
    モデルパラメータの指数移動平均を保持し、推論時の安定性を向上させる
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        logger.info(f"✅ EMA registered (decay={self.decay})")
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ===============================
# 3. モデル制御・初期化
# ===============================

def freeze_encoder_layers(model, ratio=0.5):
    """
    Encoderの下位レイヤーを凍結し、事前学習済みの知識を保護する
    """
    encoder = getattr(model.get_encoder(), 'layers', None)
    if encoder is None:
        logger.warning("⚠️ Could not find encoder layers to freeze.")
        return

    total_layers = len(encoder)
    freeze_count = int(total_layers * ratio)
    for i, layer in enumerate(encoder):
        if i < freeze_count:
            for param in layer.parameters():
                param.requires_grad = False
    logger.info(f"🔒 Frozen {freeze_count}/{total_layers} encoder layers (ratio={ratio})")

def setup_model_and_tokenizer(config, device):
    """
    モデルとトークナイザーをロードし、初期設定を行う
    """
    try:
    # 1. まずはMarian専用トークナイザーでロードを試みる
        tokenizer = MarianTokenizer.from_pretrained(config.model_name)
    except Exception:
    # 2. ダメならAutoTokenizerに戻す（保険）
        logger.warning("⚠️ Failed to load MarianTokenizer, falling back to AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, 
    use_fast=False,   # sentencepieceを確実に使うため、あえてFalseに
    trust_remote_code=True)

# 重要：正しい言語コードを教える（MarianMTはここが肝！）
# opus-mt-en-jap の場合、ソースは 'en', ターゲットは 'ja' (または 'jpn') だが、
# MarianTokenizerは自動判定してくれることが多い。念のため確認ログを出す。
    logger.info(f"🧩 Tokenizer Vocab Size: {tokenizer.vocab_size}")
    
    
    # FP32でロード (GradScaler/AMPで動的に制御するため)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        use_safetensors=True
    ).to(device)
    
    # Label Smoothingの設定
    if config.use_label_smoothing and hasattr(model.config, 'label_smoothing'):
        model.config.label_smoothing = config.label_smoothing
        logger.info(f"✨ Label Smoothing enabled: {config.label_smoothing}")
    
    # torch.compile (利用可能な場合のみ)
    if config.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("⚡ Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile skipped: {e}")
            
    return model, tokenizer
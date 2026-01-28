import os
import torch
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

# 自作モジュールのインポート
from config import TrainingConfig
from data_utils import create_dataloaders
from model_utils import setup_model_and_tokenizer, FocalLoss, EMA, freeze_encoder_layers
from trainer import LRFinder, EarlyStopping, get_total_steps, train_epoch,evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training(config: TrainingConfig):
    # 1. 環境準備
    device = torch.device("cuda" if torch.cuda.is_available() and not config.mock_mode else "cpu")
    logger.info(f"🚀 Training on: {device} (Mock: {config.mock_mode})")
    os.makedirs(config.save_dir, exist_ok=True)

    # 2. モデル & トークナイザーのセットアップ
    model, tokenizer = setup_model_and_tokenizer(config, device)
    
    # 3. データローダーの作成 (3-Phase対応)
    loaders_map = create_dataloaders(config, tokenizer)
    
    # 4. オプティマイザ設定 (Weight Decayの分離)
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # 5. LR Finder (オプション)
    if config.use_lr_finder and loaders_map["span"]:
        finder = LRFinder(model, optimizer, device)
        suggested_lr = finder.find(loaders_map["span"], min_lr=config.lr_finder_min, max_lr=config.lr_finder_max)
        finder.plot(os.path.join(config.save_dir, "lr_finder.png"))
        config.learning_rate = suggested_lr
        for pg in optimizer.param_groups: pg['lr'] = suggested_lr

    # 6. スケジューラ設定 (ここが修正の肝: 正確な合計ステップ数)
    total_steps = get_total_steps(config.phase_epochs, loaders_map, config)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 5, # 10倍は高すぎるため5倍に抑制
        total_steps=total_steps,
        pct_start=0.1, # ウォームアップを短めにして適応を早める
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    logger.info(f"📈 Scheduler initialized with {total_steps} steps.")

    # 7. その他のツール
    criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma) if config.use_focal_loss else None
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    if ema: ema.register()
    
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp and device.type == "cuda")
    early_stopping = EarlyStopping(patience=config.patience)
    
    # 初期凍結
    freeze_encoder_layers(model, ratio=0.5)

    # 8. 学習ループ (3-Phase)
    global_step = 0
    best_val_loss = float('inf')

    for phase_idx, n_epochs in enumerate(config.phase_epochs):
        if n_epochs <= 0: continue
        
        # フェーズごとのローダー選択 (ここでもアンカーデータを保証)
        if phase_idx == 0:
            phase_loaders = [loaders_map["span"], loaders_map["bywork"]]
            logger.info("--- PHASE 1: Base Training ---")
        elif phase_idx == 1:
            phase_loaders = [loaders_map["chunk"], loaders_map["bywork"], loaders_map["span"]]
            logger.info("--- PHASE 2: Contextual Training ---")
        else:
            # Phase 3: 本命データ + アンカー(span)を混ぜて忘却防止
            phase_loaders = [loaders_map["practical_chunk"], loaders_map["practical_line"], loaders_map["span"]]
            logger.info("--- PHASE 3: Domain Specialization (with Anchor) ---")

        for epoch in range(n_epochs):
            # 1エポック学習
            avg_loss = train_epoch(model, phase_loaders, optimizer, scheduler, scaler, device, config, epoch, criterion, ema)
            
            # 特定タイミングで凍結解除
            if phase_idx == 0 and epoch == 0:
                for p in model.parameters(): p.requires_grad = True
                logger.info("🔓 Model fully unfrozen.")

            # バリデーション (簡易化のためここでは省略。必要に応じて追加)
            # if val_loss < best_val_loss: save_model(...)
            if loaders_map.get("val"): # valローダーがある場合
                val_loss = evaluate_model(model, loaders_map["val"], device, config, criterion, ema)
                logger.info(f"📊 Epoch {epoch+1} Val Loss: {val_loss:.4f}")

                # ベストモデルの保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(config.save_dir, "best_model")
                    model.save_pretrained(best_path)
                    tokenizer.save_pretrained(best_path)
                    logger.info(f"🏆 New best model saved (loss: {val_loss:.4f})")
            
    logger.info("✅ Training Finished.")
    model.save_pretrained(os.path.join(config.save_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.save_dir, "final_model"))

if __name__ == "__main__":
    # 設定例
    cfg = TrainingConfig(
        file_paths=[
            "./../../data/lyrics_dataset.jsonl",
            "./../../data/separated_literary_dataset.jsonl",
            "./../../data/OpenSubtitles_sample_40000.jsonl",
            "./../../data/TED_sample_40000.jsonl",
            "./../../data/Tatoeba_sample_40000.jsonl",
            "./../../data/all_outenjp.jsonl"
        ],
        file_types=[2,1,0,0,0,0],
        epochs=3,
        phase_epochs=[1, 1, 1],
        batch_size=8,
        mock_mode=True
    )
    run_training(cfg)
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
from trainer import LRFinder, EarlyStopping, get_total_steps,get_phase_loaders, train_epoch,evaluate_model,InterleavedLoaders

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

    # 6. スケジューラ設定
    total_steps = get_total_steps(config.phase_epochs, loaders_map, config)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 5, 
        total_steps=total_steps,
        pct_start=0.3, # ウォームアップ長め推奨
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    logger.info(f"📈 Scheduler initialized with {total_steps} steps.")

    # 7. 損失関数とツール (FocalLossを使わない場合は標準のCrossEntropyを使う)
    if config.use_focal_loss:
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        # Noneだとエラーになる可能性があるので、標準を設定
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    if ema: ema.register()
    
    # Jetson/Older PyTorch compatibility
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    early_stopping = EarlyStopping(patience=config.patience)
    
    # 初期凍結
    freeze_encoder_layers(model, ratio=0.5)

    # 8. 学習ループ (3-Phase)
    global_step = 0
    best_val_loss = float('inf')

    for phase_idx, n_epochs in enumerate(config.phase_epochs):
        if n_epochs <= 0: continue
        
        # Phase 3: Encoder再凍結
        if phase_idx == 2:
            logger.info("🔒 PHASE 3: Re-freezing Encoder to protect grammar...")
            encoder = model.get_encoder()
            for param in encoder.parameters():
                param.requires_grad = False
        
        phase_names = ["Base Training", "Contextual Training", "Domain Specialization"]
        logger.info(f"--- PHASE {phase_idx+1}: {phase_names[phase_idx]} ---")
        
        phase_loaders = get_phase_loaders(phase_idx, loaders_map)
        
        for epoch in range(n_epochs):
            # 1エポック学習
            avg_loss = train_epoch(model, phase_loaders, optimizer, scheduler, scaler, device, config, epoch, criterion, ema)
            
            # Phase 1 の最初のEpoch終わりで凍結解除
            if phase_idx == 0 and epoch == 0:
                for p in model.parameters(): p.requires_grad = True
                logger.info("🔓 Model fully unfrozen.")

            # === バリデーション & ベストモデル保存 (ここが重要！) ===
            if loaders_map.get("val"): 
                logger.info("⏳ Running Validation...")
                val_loss = evaluate_model(model, loaders_map["val"], device, config, criterion, ema)
                logger.info(f"📊 Phase {phase_idx+1} - Epoch {epoch+1} Val Loss: {val_loss:.4f}")

                # ベストモデルの更新チェック
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(config.save_dir, "best_model")
                    
                    # 保存 (EMAがある場合はEMAの重みを適用して保存するのが理想だが、
                    # ここでは簡単のため、現在のモデルを保存。evaluate_model内でEMA適用してるならOK)
                    model.save_pretrained(best_path)
                    tokenizer.save_pretrained(best_path)
                    logger.info(f"🏆 New best model saved to {best_path} (loss: {val_loss:.4f})")
            else:
                logger.warning("⚠️ No validation loader found. Skipping validation.")

    logger.info("✅ Training Finished.")
    # 最終モデルの保存
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
            "./../../data/Tatoeba_sample_40000.jsonl"
        ],
        file_types=[2,1,0,0,0],
        max_samples_per_span_file=40000,
        practical_upsample=50,
        epochs=6,
        phase_epochs=[1, 2, 3],
        batch_size=16,
        mock_mode=False,
        use_focal_loss= True,
        use_lr_finder=False
    )
    run_training(cfg)
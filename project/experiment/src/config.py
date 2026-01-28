from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    # --- モデル設定 ---
    model_name: str = "Helsinki-NLP/opus-mt-en-jap"
    save_dir: str = "./models/translation_v2"
    max_len: int = 128
    
    # --- データ設定 ---
    file_paths: List[str] = field(default_factory=list)
    # 0=Span(一般), 1=ByWork(文学/コンテキスト), 2=Practical(本命/歌詞等)
    file_types: List[int] = field(default_factory=list)
    tags: Optional[List[str]] = None
    max_samples_per_span_file: int = 30000
    val_split: float = 0.05
    
    # --- 3-Phase 学習制御 ---
    # [Phase1(基礎), Phase2(文脈), Phase3(本命特化)]
    phase_epochs: List[int] = field(default_factory=lambda: [2, 1, 1])
    # Phase3で一般知識を忘れないためのアップサンプリング調整
    practical_upsample: int = 1 
    
    # --- ハイパーパラメータ ---
    epochs: int = 4 # phase_epochsの合計と一致させる
    batch_size: int = 16
    accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    patience: int = 3
    
    # --- 最適化機能 ---
    use_amp: bool = True
    use_bfloat16: bool = True
    use_compile: bool = False
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # --- 損失関数設定 ---
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    # --- LR Finder ---
    use_lr_finder: bool = False
    lr_finder_min: float = 1e-7
    lr_finder_max: float = 1e-2
    lr_finder_num_iter: int = 100
    
    # --- Mock / デバッグモード ---
    mock_mode: bool = True
    mock_samples: int = 100
    random_seed: int = 42
    num_workers: int = 4

    def __post_init__(self):
        # バリデーション: フェーズ合計とエポック数の不一致をチェック
        if sum(self.phase_epochs) != self.epochs:
            print(f"⚠️ Warning: sum(phase_epochs) [{sum(self.phase_epochs)}] does not match total epochs [{self.epochs}]")
        
        # ファイルパスとタイプの数が一致しているか確認
        if len(self.file_paths) != len(self.file_types):
            raise ValueError("file_paths and file_types must have the same length.")
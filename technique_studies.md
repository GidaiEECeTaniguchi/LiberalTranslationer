# テクニックまとめ

機械学習に使われる、様々な手法をまとめる.

## Residual Block

ResNetで有名な手法。入力を直接出力に加えることで、深いネットワークでも勾配が消失せず、訓練しやすくなります。

## Attention機構

特徴マップのどの部分が重要かを学習し、重要な部分を強調します。数字の特徴的な部分(例:「7」の横棒)に注目できるようになります。

## Global Average Pooling

全結合層の代わりに使うことで、パラメータ数を減らし過学習を防ぎます。

## Dropout

訓練時にランダムにニューロンを無効化することで、特定のニューロンへの依存を防ぎ、ロバストなモデルを作ります。

## Label Smoothing

正解ラベルを「1」ではなく「0.9」、不正解を「0」ではなく「0.01」のように柔らかくします。モデルが過信しすぎるのを防ぎ、汎化性能が向上します。

## Batch Normalization

バッチ正規化、正規化すると安定する、あたりまえ

## Weight Decay

重みが大きくなりすぎるのを防ぐL2正則化の一種。過学習を抑制します。

## AdamW

Adamの改良版。Weight Decayの扱いを改善し、より良い汎化性能を実現します。 


## Cosine Annealing with Warm Restarts

学習率をコサインカーブに沿って周期的に変化させます。局所最適解から脱出しやすくなり、より良い解を見つけられます。


## Mixed Precision Training (AMP)

計算をFP16(半精度浮動小数点)で行い、メモリ使用量を削減し訓練を高速化します。精度への影響はほぼありません。


## Test Time Augmentation (TTA)
テスト時にも複数回推論して平均を取ることで、予測の安定性が向上します。

## モデルアンサンブル

複数の独立したモデルの予測を平均することで、単一モデルのエラーを相殺し、精度が向上します。

## シード固定

## DataLoader最適化



データ拡張系(言語)としてはこんなのがある

##  Back Translation (逆翻訳)

## 同義語置換 (Synonym Replacement)

## Random Insertion/Deletion/Swap
## Contextual Word Embeddings (BERT Masking)

## Paraphrasing (言い換え)


## Mixup for Text (TMix, WordMixup)


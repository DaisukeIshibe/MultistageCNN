#!/usr/bin/env python3
"""
Multistage CNN Inference Program / 多段階CNN推論プログラム

This program performs inference using a trained multistage CNN model for CIFAR-10 classification.
The model outputs 11 categories (10 CIFAR-10 categories + OTHER category).

このプログラムは、学習済みの多段階CNNモデルを使用してCIFAR-10分類の推論を実行します。
モデルは11カテゴリ（CIFAR-10の10カテゴリ + OTHERカテゴリ）を出力します。

Usage / 使用方法:
    python inference.py --model_path trained_model_1epoch [options]
    
    Options:
    --model_path: Path to trained model / 学習済みモデルのパス
    --batch_size: Batch size for inference (default: 32) / 推論用バッチサイズ（デフォルト: 32）
    --threshold: Confidence threshold for Stage 2 (default: 0.7) / Stage 2の信頼度閾値（デフォルト: 0.7）
    --light_mode: Use light mode (1/10 of test data) / 軽量モード（テストデータの1/10を使用）
    --output_dir: Directory to save results / 結果保存ディレクトリ
    --verbose: Enable verbose output / 詳細出力を有効化

Author: AI Assistant
Date: November 2, 2025
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras


# CIFAR-10 class names / CIFAR-10クラス名
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Final 11 categories (10 CIFAR-10 + OTHER) / 最終的な11カテゴリ（CIFAR-10の10個 + OTHER）
FINAL_CLASSES = CIFAR10_CLASSES + ['OTHER']


def load_and_preprocess_data(light_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess CIFAR-10 test data / CIFAR-10テストデータの読み込みと前処理
    
    Args:
        light_mode: Whether to use light mode (1/10 of data) / 軽量モードを使用するか（データの1/10）
        
    Returns:
        x_test: Preprocessed test images / 前処理済みテスト画像
        y_test: Test labels / テストラベル
    """
    print("Loading CIFAR-10 test data... / CIFAR-10テストデータを読み込み中...")
    
    # Load CIFAR-10 data / CIFAR-10データの読み込み
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    if light_mode:
        # Use only 1/10 of test data for light mode / 軽量モードでは1/10のテストデータのみ使用
        subset_size = len(x_test) // 10
        x_test = x_test[:subset_size]
        y_test = y_test[:subset_size]
        print(f"Light mode: Using {len(x_test)} test samples / 軽量モード: {len(x_test)}個のテストサンプルを使用")
    
    # Normalize pixel values to [0, 1] / ピクセル値を[0, 1]に正規化
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels / ラベルを1次元に変換
    y_test = y_test.flatten()
    
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_test, y_test


def load_model(model_path: str) -> keras.Model:
    """
    Load trained multistage CNN model / 学習済み多段階CNNモデルの読み込み
    
    Args:
        model_path: Path to the saved model / 保存されたモデルのパス
        
    Returns:
        model: Loaded model / 読み込まれたモデル
    """
    print(f"Loading model from {model_path}... / {model_path}からモデルを読み込み中...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path} / モデルが見つかりません: {model_path}")
    
    try:
        # Load the saved model / 保存されたモデルの読み込み
        model = keras.models.load_model(model_path)
        print("Model loaded successfully! / モデルの読み込みが完了しました！")
        
        # Print model summary / モデルサマリーの表示
        print("\nModel Summary / モデルサマリー:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)} / モデル読み込みエラー: {str(e)}")
        raise


def predict_final_categories(model: keras.Model, 
                           x_data: np.ndarray, 
                           threshold: float = 0.7,
                           batch_size: int = 32,
                           verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Perform inference with the multistage CNN model / 多段階CNNモデルで推論を実行
    
    Args:
        model: Trained multistage CNN model / 学習済み多段階CNNモデル
        x_data: Input data / 入力データ
        threshold: Confidence threshold for Stage 2 / Stage 2の信頼度閾値
        batch_size: Batch size for inference / 推論用バッチサイズ
        verbose: Whether to show progress / 進行状況を表示するか
        
    Returns:
        final_predictions: Final prediction results (0-10) / 最終予測結果（0-10）
        prediction_info: Additional prediction information / 追加の予測情報
    """
    if verbose:
        print(f"Performing inference with batch size {batch_size}... / バッチサイズ{batch_size}で推論を実行中...")
        print(f"Confidence threshold: {threshold} / 信頼度閾値: {threshold}")
    
    num_samples = len(x_data)
    final_predictions = []
    stage1_predictions = []
    stage2_confidences = []
    
    # Process data in batches / データをバッチごとに処理
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = x_data[start_idx:end_idx]
        
        if verbose and batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx + 1}/{num_batches} / バッチ{batch_idx + 1}/{num_batches}を処理中")
        
        # Get model predictions / モデル予測の取得
        stage1_output, stage2_outputs = model(batch_data, training=False)
        
        # Convert to numpy arrays / numpy配列に変換
        stage1_probs = stage1_output.numpy()
        stage1_pred = np.argmax(stage1_probs, axis=-1)
        
        # Process each sample in the batch / バッチ内の各サンプルを処理
        for i in range(len(batch_data)):
            predicted_category = stage1_pred[i]
            stage1_predictions.append(predicted_category)
            
            # Get Stage 2 confidence for the predicted category / 予測されたカテゴリのStage 2信頼度を取得
            stage2_key = f'category_{predicted_category}'
            if stage2_key in stage2_outputs:
                stage2_confidence = stage2_outputs[stage2_key][i, 1].numpy()  # Probability of correctness / 正解の確率
                stage2_confidences.append(stage2_confidence)
                
                if stage2_confidence >= threshold:
                    # High confidence: use Stage 1 prediction / 高信頼度: Stage 1の予測を採用
                    final_predictions.append(predicted_category)
                else:
                    # Low confidence: classify as OTHER / 低信頼度: OTHERに分類
                    final_predictions.append(10)  # OTHER category
            else:
                # Fallback: use Stage 1 prediction / フォールバック: Stage 1の予測を使用
                stage2_confidences.append(0.0)
                final_predictions.append(predicted_category)
    
    inference_time = time.time() - start_time
    
    if verbose:
        print(f"Inference completed in {inference_time:.2f} seconds / 推論が{inference_time:.2f}秒で完了")
        print(f"Average time per sample: {inference_time/num_samples*1000:.2f} ms / サンプル当たり平均時間: {inference_time/num_samples*1000:.2f} ms")
    
    # Prepare prediction information / 予測情報の準備
    prediction_info = {
        'stage1_predictions': np.array(stage1_predictions),
        'stage2_confidences': np.array(stage2_confidences),
        'threshold': threshold,
        'inference_time': inference_time,
        'batch_size': batch_size,
        'num_samples': num_samples
    }
    
    return np.array(final_predictions), prediction_info


def evaluate_predictions(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        prediction_info: Dict,
                        output_dir: Optional[str] = None,
                        verbose: bool = True) -> Dict:
    """
    Evaluate prediction results and generate reports / 予測結果の評価とレポート生成
    
    Args:
        y_true: Ground truth labels (0-9 for CIFAR-10) / 正解ラベル（CIFAR-10の0-9）
        y_pred: Final predictions (0-10, where 10 is OTHER) / 最終予測（0-10、10はOTHER）
        prediction_info: Additional prediction information / 追加の予測情報
        output_dir: Directory to save evaluation results / 評価結果の保存ディレクトリ
        verbose: Whether to print detailed results / 詳細結果を出力するか
        
    Returns:
        evaluation_results: Dictionary containing evaluation metrics / 評価指標を含む辞書
    """
    if verbose:
        print("\n" + "="*70)
        print("INFERENCE EVALUATION RESULTS / 推論評価結果")
        print("="*70)
    
    # Convert CIFAR-10 labels to 11-category labels (keep original + add OTHER possibility)
    # CIFAR-10ラベルを11カテゴリラベルに変換（元のラベルを保持 + OTHER可能性を追加）
    y_true_11cat = y_true.copy()  # Original CIFAR-10 labels remain the same / 元のCIFAR-10ラベルはそのまま
    
    # Calculate accuracy / 精度の計算
    overall_accuracy = accuracy_score(y_true_11cat, y_pred)
    
    # Calculate stage-wise metrics / ステージ別メトリクスの計算
    stage1_predictions = prediction_info['stage1_predictions']
    stage1_accuracy = accuracy_score(y_true, stage1_predictions)
    
    # Count OTHER predictions / OTHER予測の数をカウント
    other_count = np.sum(y_pred == 10)
    other_ratio = other_count / len(y_pred)
    
    # Count correct predictions that were kept vs rejected / 保持された正解予測と拒否された正解予測の数
    correct_stage1 = (stage1_predictions == y_true)
    kept_correct = np.sum((y_pred == y_true) & (y_pred != 10))
    rejected_correct = np.sum(correct_stage1 & (y_pred == 10))
    
    if verbose:
        print(f"\nPerformance Summary / 性能サマリー:")
        print(f"  Overall Accuracy (11-cat): {overall_accuracy:.4f} / 総合精度（11カテゴリ）: {overall_accuracy:.4f}")
        print(f"  Stage 1 Accuracy (10-cat): {stage1_accuracy:.4f} / Stage 1精度（10カテゴリ）: {stage1_accuracy:.4f}")
        print(f"  OTHER predictions: {other_count}/{len(y_pred)} ({other_ratio:.2%}) / OTHER予測: {other_count}/{len(y_pred)} ({other_ratio:.2%})")
        print(f"  Correct kept: {kept_correct} / 保持された正解: {kept_correct}")
        print(f"  Correct rejected: {rejected_correct} / 拒否された正解: {rejected_correct}")
        
        print(f"\nInference Statistics / 推論統計:")
        print(f"  Threshold used: {prediction_info['threshold']} / 使用された閾値: {prediction_info['threshold']}")
        print(f"  Batch size: {prediction_info['batch_size']} / バッチサイズ: {prediction_info['batch_size']}")
        print(f"  Total inference time: {prediction_info['inference_time']:.2f}s / 総推論時間: {prediction_info['inference_time']:.2f}秒")
        print(f"  Average confidence: {np.mean(prediction_info['stage2_confidences']):.4f} / 平均信頼度: {np.mean(prediction_info['stage2_confidences']):.4f}")
    
    # Generate classification report / 分類レポートの生成
    if verbose:
        print(f"\nClassification Report (11 categories) / 分類レポート（11カテゴリ）:")
        report = classification_report(y_true_11cat, y_pred, 
                                     target_names=FINAL_CLASSES, 
                                     zero_division=0)
        print(report)
    
    # Generate and save confusion matrix / 混同行列の生成と保存
    cm = confusion_matrix(y_true_11cat, y_pred, labels=list(range(11)))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix plot / 混同行列プロットの保存
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=FINAL_CLASSES, yticklabels=FINAL_CLASSES)
        plt.title('Confusion Matrix - Inference Results / 混同行列 - 推論結果')
        plt.xlabel('Predicted / 予測')
        plt.ylabel('Actual / 実際')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'inference_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path} / 混同行列を保存: {cm_path}")
        
        # Save confidence distribution plot / 信頼度分布プロットの保存
        plt.figure(figsize=(10, 6))
        plt.hist(prediction_info['stage2_confidences'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(prediction_info['threshold'], color='red', linestyle='--', 
                   label=f"Threshold: {prediction_info['threshold']}")
        plt.xlabel('Stage 2 Confidence / Stage 2信頼度')
        plt.ylabel('Frequency / 頻度')
        plt.title('Distribution of Stage 2 Confidence Scores / Stage 2信頼度スコアの分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        conf_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence distribution saved to: {conf_path} / 信頼度分布を保存: {conf_path}")
    
    # Prepare evaluation results / 評価結果の準備
    evaluation_results = {
        'overall_accuracy': overall_accuracy,
        'stage1_accuracy': stage1_accuracy,
        'other_ratio': other_ratio,
        'kept_correct': kept_correct,
        'rejected_correct': rejected_correct,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true_11cat, y_pred, 
                                                     target_names=FINAL_CLASSES, 
                                                     zero_division=0, output_dict=True),
        'prediction_info': prediction_info
    }
    
    return evaluation_results


def main():
    """
    Main function for inference program / 推論プログラムのメイン関数
    """
    parser = argparse.ArgumentParser(
        description='Multistage CNN Inference Program / 多段階CNN推論プログラム',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model / 学習済みモデルのパス')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference / 推論用バッチサイズ')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for Stage 2 / Stage 2の信頼度閾値')
    parser.add_argument('--light_mode', action='store_true',
                       help='Use light mode (1/10 of test data) / 軽量モード（テストデータの1/10を使用）')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save results / 結果保存ディレクトリ')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output / 詳細出力を有効化')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTISTAGE CNN INFERENCE / 多段階CNN推論")
    print("="*70)
    print(f"Model path: {args.model_path} / モデルパス: {args.model_path}")
    print(f"Batch size: {args.batch_size} / バッチサイズ: {args.batch_size}")
    print(f"Threshold: {args.threshold} / 閾値: {args.threshold}")
    print(f"Light mode: {args.light_mode} / 軽量モード: {args.light_mode}")
    print(f"Output directory: {args.output_dir} / 出力ディレクトリ: {args.output_dir}")
    print("-" * 70)
    
    try:
        # Load test data / テストデータの読み込み
        x_test, y_test = load_and_preprocess_data(light_mode=args.light_mode)
        
        # Load trained model / 学習済みモデルの読み込み
        model = load_model(args.model_path)
        
        # Perform inference / 推論の実行
        final_predictions, prediction_info = predict_final_categories(
            model=model,
            x_data=x_test,
            threshold=args.threshold,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        # Evaluate results / 結果の評価
        evaluation_results = evaluate_predictions(
            y_true=y_test,
            y_pred=final_predictions,
            prediction_info=prediction_info,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        print(f"\nInference completed successfully! / 推論が正常に完了しました！")
        print(f"Results saved to: {args.output_dir} / 結果保存先: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)} / 推論中にエラーが発生しました: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
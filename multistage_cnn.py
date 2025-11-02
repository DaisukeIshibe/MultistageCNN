"""
多段階CNN（2段階）学習プログラム
Multi-Stage CNN (2-stage) Training Program

このプログラムは以下の機能を実装：
This program implements the following features:
1. 1段目CNN: CIFAR-10の10カテゴリ分類
   Stage 1 CNN: 10-category classification for CIFAR-10
2. 2段目CNN: カテゴリ毎の正誤判定（正解/OTHER）
   Stage 2 CNN: Correctness judgment for each category (correct/OTHER)
3. 最終出力: 11カテゴリ（10カテゴリ + OTHER）
   Final output: 11 categories (10 categories + OTHER)
4. 再学習オプション: 1段目を固定し2段目のみ学習可能
   Retraining option: Stage 1 can be frozen while only Stage 2 is trainable
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os
from typing import Tuple, Dict, List


# CIFAR-10のカテゴリ名 / CIFAR-10 category names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 最終出力カテゴリ名（10カテゴリ + OTHER） / Final output category names (10 categories + OTHER)
FINAL_CLASSES = CIFAR10_CLASSES + ['OTHER']


def load_and_preprocess_data(light_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CIFAR-10データを読み込み、前処理を行う
    Load and preprocess CIFAR-10 data
    
    Args:
        light_mode: 軽量モードでデータサイズを削減するかどうか
                   Whether to reduce data size in light mode
    
    Returns:
        x_train, y_train, x_test, y_test: 前処理済みデータ
                                         Preprocessed data
    """
    # CIFAR-10データの読み込み / Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # 軽量モードの場合、データサイズを削減 / Reduce data size in light mode
    if light_mode:
        # 訓練データを1/10に削減 / Reduce training data to 1/10
        train_size = len(x_train) // 10
        test_size = len(x_test) // 10
        
        indices_train = np.random.choice(len(x_train), train_size, replace=False)
        indices_test = np.random.choice(len(x_test), test_size, replace=False)
        
        x_train = x_train[indices_train]
        y_train = y_train[indices_train]
        x_test = x_test[indices_test]
        y_test = y_test[indices_test]
        
        print(f"Light mode: Using {train_size} training samples and {test_size} test samples")
    
    # データ型をfloat32に変換し、0-1に正規化 / Convert to float32 and normalize to 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # ラベルを1次元に変換 / Convert labels to 1D
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test


def create_stage1_cnn() -> keras.Model:
    """
    1段目CNN: CIFAR-10の10カテゴリ分類モデルを作成
    Create Stage 1 CNN: 10-category classification model for CIFAR-10
    
    Returns:
        stage1_model: 1段目CNNモデル / Stage 1 CNN model
    """
    model = keras.Sequential([
        # 1st Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 2nd Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 3rd Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', name='stage1_output')
    ], name='stage1_cnn')
    
    return model


def create_stage2_cnn(category_idx: int) -> keras.Model:
    """
    2段目CNN: 特定カテゴリの正誤判定モデルを作成
    Create Stage 2 CNN: Correctness judgment model for specific category
    
    Args:
        category_idx: 対象カテゴリのインデックス / Target category index
        
    Returns:
        stage2_model: 2段目CNNモデル / Stage 2 CNN model
    """
    model = keras.Sequential([
        # Feature extraction layers
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Classification head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax', name=f'stage2_category_{category_idx}_output')
    ], name=f'stage2_cnn_category_{category_idx}')
    
    return model


class MultiStageCNN(keras.Model):
    """
    2段階CNN統合モデル
    Multi-Stage CNN Integrated Model
    """
    
    def __init__(self):
        super(MultiStageCNN, self).__init__()
        
        # 1段目CNN / Stage 1 CNN
        self.stage1_model = create_stage1_cnn()
        
        # 2段目CNN（各カテゴリ用） / Stage 2 CNN (for each category)
        self.stage2_models = {}
        for i in range(10):
            self.stage2_models[f'category_{i}'] = create_stage2_cnn(i)
    
    def call(self, inputs, training=None, stage1_only=False):
        """
        フォワードパス
        Forward pass
        
        Args:
            inputs: 入力画像 / Input images
            training: 学習モードかどうか / Whether in training mode
            stage1_only: 1段目のみ実行するかどうか / Whether to execute only Stage 1
            
        Returns:
            stage1_output: 1段目の出力 / Stage 1 output
            stage2_outputs: 2段目の出力（stage1_onlyがTrueの場合はNone）
                           Stage 2 outputs (None if stage1_only is True)
        """
        # 1段目の実行 / Execute Stage 1
        stage1_output = self.stage1_model(inputs, training=training)
        
        if stage1_only:
            return stage1_output, None
        
        # 2段目の実行 / Execute Stage 2
        stage2_outputs = {}
        for i in range(10):
            stage2_outputs[i] = self.stage2_models[f'category_{i}'](inputs, training=training)
        
        return stage1_output, stage2_outputs
    
    def set_stage1_trainable(self, trainable: bool):
        """
        1段目の学習可能フラグを設定
        Set trainable flag for Stage 1
        
        Args:
            trainable: 学習可能にするかどうか / Whether to make trainable
        """
        self.stage1_model.trainable = trainable
        for layer in self.stage1_model.layers:
            layer.trainable = trainable
    
    def set_stage2_category_trainable(self, category_indices: List[int] = None, trainable: bool = True):
        """
        特定カテゴリの2段目CNNの学習可能フラグを設定
        Set trainable flag for Stage 2 CNNs of specific categories
        
        Args:
            category_indices: 学習対象のカテゴリインデックスリスト（Noneの場合は全カテゴリ）
                             List of category indices to train (all categories if None)
            trainable: 学習可能にするかどうか / Whether to make trainable
        """
        if category_indices is None:
            # 全カテゴリの設定 / Set all categories
            category_indices = list(range(10))
        
        for i in range(10):
            model_key = f'category_{i}'
            if i in category_indices:
                # 指定されたカテゴリは指定された状態に設定 / Set specified categories to specified state
                self.stage2_models[model_key].trainable = trainable
                for layer in self.stage2_models[model_key].layers:
                    layer.trainable = trainable
            else:
                # 指定されていないカテゴリは学習を無効化 / Disable training for unspecified categories
                self.stage2_models[model_key].trainable = False
                for layer in self.stage2_models[model_key].layers:
                    layer.trainable = False
    
    def get_trainable_summary(self):
        """
        各段階の学習可能状態を表示
        Display trainable status of each stage
        
        Returns:
            summary: 学習可能状態のサマリー / Summary of trainable status
        """
        summary = {
            'stage1_trainable': self.stage1_model.trainable,
            'stage2_trainable': {}
        }
        
        for i in range(10):
            model_key = f'category_{i}'
            category_name = CIFAR10_CLASSES[i]
            summary['stage2_trainable'][category_name] = self.stage2_models[model_key].trainable
            
        return summary


def create_multistage_loss():
    """
    多段階学習用の損失関数を作成
    
    Returns:
        loss_function: カスタム損失関数
    """
    def multistage_loss(y_true, outputs):
        """
        多段階損失関数
        
        Args:
            y_true: 真のラベル
            outputs: (stage1_output, stage2_outputs)のタプル
            
        Returns:
            total_loss: 総損失
        """
        stage1_output, stage2_outputs = outputs
        
        # 1段目の損失（カテゴリカル分類）
        stage1_loss = keras.losses.sparse_categorical_crossentropy(y_true, stage1_output)
        
        # 2段目の損失（各カテゴリの正誤判定）
        stage2_loss = 0
        for i in range(10):
            # カテゴリiの正誤ラベルを作成（正解なら1、不正解なら0）
            category_correct = tf.cast(tf.equal(y_true, i), tf.float32)
            category_correct = tf.expand_dims(category_correct, -1)
            category_labels = tf.concat([1 - category_correct, category_correct], axis=-1)
            
            # カテゴリiの2段目損失
            category_loss = keras.losses.categorical_crossentropy(category_labels, stage2_outputs[i])
            stage2_loss += category_loss
        
        # 総損失（重み付き和）
        total_loss = stage1_loss + 0.5 * stage2_loss
        
        return total_loss
    
    return multistage_loss


def predict_final_categories(model: MultiStageCNN, x_data: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    最終的な11カテゴリ予測を実行
    Execute final 11-category prediction
    
    Args:
        model: 学習済みモデル / Trained model
        x_data: 入力データ / Input data
        threshold: 2段目の信頼度閾値 / Confidence threshold for Stage 2
        
    Returns:
        final_predictions: 最終予測結果（0-10の11カテゴリ）
                          Final prediction results (11 categories: 0-10)
    """
    # 1段目と2段目の予測を取得 / Get Stage 1 and Stage 2 predictions
    stage1_output, stage2_outputs = model(x_data, training=False)
    
    # 1段目の予測カテゴリ / Stage 1 predicted categories
    stage1_pred = tf.argmax(stage1_output, axis=-1).numpy()
    
    final_predictions = []
    
    for i in range(len(x_data)):
        predicted_category = stage1_pred[i]
        
        # 該当カテゴリの2段目モデルで正誤判定 / Judge correctness with Stage 2 model for the category
        stage2_confidence = stage2_outputs[predicted_category][i, 1]  # 正解の確率 / Probability of correctness
        
        if stage2_confidence >= threshold:
            # 信頼度が閾値以上なら、1段目の予測を採用 / Adopt Stage 1 prediction if confidence >= threshold
            final_predictions.append(predicted_category)
        else:
            # 信頼度が閾値未満なら、OTHERカテゴリ（10） / Use OTHER category (10) if confidence < threshold
            final_predictions.append(10)
    
    return np.array(final_predictions)


def train_model(model: MultiStageCNN, 
                x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray,
                epochs: int = 50,
                batch_size: int = 32,
                stage2_only: bool = False,
                stage2_categories: List[int] = None) -> Dict:
    """
    モデルの学習を実行
    Execute model training
    
    Args:
        model: 学習対象モデル / Target model for training
        x_train, y_train: 訓練データ / Training data
        x_test, y_test: テストデータ / Test data
        epochs: エポック数 / Number of epochs
        batch_size: バッチサイズ / Batch size
        stage2_only: 2段目のみ学習するかどうか / Whether to train only Stage 2
        stage2_categories: 学習対象の2段目カテゴリ（Noneの場合は全カテゴリ）
                          Stage 2 categories to train (all categories if None)
        
    Returns:
        history: 学習履歴 / Training history
    """
    if stage2_only:
        # 1段目を固定
        model.set_stage1_trainable(False)
        
        if stage2_categories is not None:
            # 特定カテゴリの2段目のみ学習
            model.set_stage2_category_trainable(stage2_categories, trainable=True)
            category_names = [CIFAR10_CLASSES[i] for i in stage2_categories]
            print(f"Category-specific Stage 2 training mode:")
            print(f"  - Stage 1: Frozen")
            print(f"  - Stage 2 trainable categories: {category_names}")
            print(f"  - Stage 2 frozen categories: {[CIFAR10_CLASSES[i] for i in range(10) if i not in stage2_categories]}")
        else:
            # 全ての2段目を学習
            model.set_stage2_category_trainable(None, trainable=True)
            print("Stage 2 only training mode: Stage 1 is frozen, all Stage 2 models are trainable")
    else:
        # 全体を学習
        model.set_stage1_trainable(True)
        model.set_stage2_category_trainable(None, trainable=True)
        print("Full training mode: Both stages are trainable")
    
    # オプティマイザーの設定
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # カスタム学習ループ
    train_loss_metric = keras.metrics.Mean()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 訓練ループ
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()
        
        num_batches = len(x_train) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                # フォワードパス
                stage1_output, stage2_outputs = model(x_batch, training=True)
                
                # 1段目の損失
                stage1_loss = keras.losses.sparse_categorical_crossentropy(y_batch, stage1_output)
                stage1_loss = tf.reduce_mean(stage1_loss)
                
                # 2段目の損失
                stage2_loss = 0
                for i in range(10):
                    category_correct = tf.cast(tf.equal(y_batch, i), tf.float32)
                    category_incorrect = 1 - category_correct
                    category_labels = tf.stack([category_incorrect, category_correct], axis=-1)
                    
                    category_loss = keras.losses.categorical_crossentropy(category_labels, stage2_outputs[i])
                    stage2_loss += tf.reduce_mean(category_loss)
                
                total_loss = stage1_loss + 0.3 * stage2_loss
            
            # バックプロパゲーション
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # メトリクス更新
            train_loss_metric.update_state(total_loss)
            train_acc_metric.update_state(y_batch, stage1_output)
        
        # 検証
        val_stage1_output, val_stage2_outputs = model(x_test, training=False)
        val_stage1_loss = keras.losses.sparse_categorical_crossentropy(y_test, val_stage1_output)
        val_stage1_loss = tf.reduce_mean(val_stage1_loss)
        
        val_stage2_loss = 0
        for i in range(10):
            val_category_correct = tf.cast(tf.equal(y_test, i), tf.float32)
            val_category_incorrect = 1 - val_category_correct
            val_category_labels = tf.stack([val_category_incorrect, val_category_correct], axis=-1)
            
            val_category_loss = keras.losses.categorical_crossentropy(val_category_labels, val_stage2_outputs[i])
            val_stage2_loss += tf.reduce_mean(val_category_loss)
        
        val_total_loss = val_stage1_loss + 0.3 * val_stage2_loss
        val_accuracy = keras.metrics.sparse_categorical_accuracy(y_test, val_stage1_output)
        val_accuracy = tf.reduce_mean(val_accuracy)
        
        # 履歴に記録
        history['loss'].append(float(train_loss_metric.result()))
        history['accuracy'].append(float(train_acc_metric.result()))
        history['val_loss'].append(float(val_total_loss))
        history['val_accuracy'].append(float(val_accuracy))
        
        print(f"Loss: {train_loss_metric.result():.4f} - "
              f"Accuracy: {train_acc_metric.result():.4f} - "
              f"Val Loss: {val_total_loss:.4f} - "
              f"Val Accuracy: {val_accuracy:.4f}")
    
    return history


def evaluate_model(model: MultiStageCNN, x_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.7):
    """
    モデルの評価を実行
    
    Args:
        model: 評価対象モデル
        x_test, y_test: テストデータ
        threshold: 2段目の信頼度閾値
    """
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # 最終予測の実行
    final_predictions = predict_final_categories(model, x_test, threshold)
    
    # 真のラベルを11カテゴリ版に変換（正解は元のラベル、不正解の場合の対応は複雑なので簡易版）
    # ここでは、テスト用に1段目の予測と真のラベルを比較してOTHERを決定
    stage1_output, _ = model(x_test, training=False)
    stage1_pred = tf.argmax(stage1_output, axis=-1).numpy()
    
    # 評価用の真のラベル（11カテゴリ版）を作成
    true_labels_11cat = []
    for i in range(len(y_test)):
        if stage1_pred[i] == y_test[i]:
            true_labels_11cat.append(y_test[i])
        else:
            true_labels_11cat.append(10)  # OTHER
    
    true_labels_11cat = np.array(true_labels_11cat)
    
    # 分類レポート
    print("\nClassification Report (11 categories):")
    print(classification_report(true_labels_11cat, final_predictions, 
                                target_names=FINAL_CLASSES, zero_division=0,
                                labels=list(range(11))))
    
    # 混同行列
    cm = confusion_matrix(true_labels_11cat, final_predictions)
    
    # 混同行列の可視化
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (11 Categories)')
    plt.colorbar()
    
    tick_marks = np.arange(len(FINAL_CLASSES))
    plt.xticks(tick_marks, FINAL_CLASSES, rotation=45)
    plt.yticks(tick_marks, FINAL_CLASSES)
    
    # 数値を表示
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 精度の計算
    accuracy = np.sum(final_predictions == true_labels_11cat) / len(true_labels_11cat)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    return accuracy, final_predictions, true_labels_11cat


def plot_training_history(history: Dict):
    """
    学習履歴をプロット
    
    Args:
        history: 学習履歴
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    メイン関数
    Main function
    """
    parser = argparse.ArgumentParser(description='Multi-Stage CNN for CIFAR-10')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--stage2_only', action='store_true', 
                        help='Train only stage 2 (freeze stage 1)')
    parser.add_argument('--threshold', type=float, default=0.7, 
                        help='Confidence threshold for stage 2')
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Path to load pre-trained model')
    parser.add_argument('--save_model', type=str, default='multistage_cnn_model', 
                        help='Path to save trained model')
    parser.add_argument('--light_mode', action='store_true', 
                        help='Use light mode with reduced data size')
    parser.add_argument('--stage2_categories', type=str, 
                        help='Comma-separated category names or indices for Stage 2 training (e.g., "ship,truck" or "8,9")')
    
    args = parser.parse_args()
    
    # stage2_categoriesの解析 / Parse stage2_categories
    stage2_category_indices = None
    if args.stage2_categories:
        category_specs = [spec.strip() for spec in args.stage2_categories.split(',')]
        stage2_category_indices = []
        
        for spec in category_specs:
            if spec.isdigit():
                # 数値インデックス / Numeric index
                idx = int(spec)
                if 0 <= idx <= 9:
                    stage2_category_indices.append(idx)
                else:
                    print(f"Warning: Invalid category index {idx}. Must be 0-9.")
            else:
                # カテゴリ名 / Category name
                if spec in CIFAR10_CLASSES:
                    stage2_category_indices.append(CIFAR10_CLASSES.index(spec))
                else:
                    print(f"Warning: Invalid category name '{spec}'. Available: {CIFAR10_CLASSES}")
        
        if not stage2_category_indices:
            print("Error: No valid categories specified. Ignoring --stage2_categories option.")
            stage2_category_indices = None
    
    print("Multi-Stage CNN for CIFAR-10")
    print("="*50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stage 2 only: {args.stage2_only}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Light mode: {args.light_mode}")
    if stage2_category_indices:
        category_names = [CIFAR10_CLASSES[i] for i in stage2_category_indices]
        print(f"Stage 2 categories: {category_names} (indices: {stage2_category_indices})")
    
    # データの読み込み / Load data
    print("\nLoading and preprocessing data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data(light_mode=args.light_mode)
    
    # モデルの作成 / Create model
    print("\nCreating multi-stage CNN model...")
    model = MultiStageCNN()
    
    # モデルを初期化（最初に一度呼び出す必要がある） / Initialize model (must be called once first)
    _ = model(x_train[:1])
    
    # 事前学習済みモデルのロード / Load pre-trained model
    if args.load_model and os.path.exists(args.load_model):
        print(f"\nLoading pre-trained model from {args.load_model}")
        model.load_weights(args.load_model)
    
    # モデルの学習 / Train model
    print("\nStarting training...")
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage2_only=args.stage2_only,
        stage2_categories=stage2_category_indices
    )
    
    # 学習可能状態の表示
    print("\nTraining Status Summary:")
    summary = model.get_trainable_summary()
    print(f"Stage 1 trainable: {summary['stage1_trainable']}")
    print("Stage 2 trainable status:")
    for category, trainable in summary['stage2_trainable'].items():
        status = "✓" if trainable else "✗"
        print(f"  {status} {category}: {trainable}")
    
    # 学習履歴のプロット / Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # モデルの評価 / Evaluate model
    print("\nEvaluating model...")
    accuracy, predictions, true_labels = evaluate_model(model, x_test, y_test, args.threshold)
    
    # モデルの保存 / Save model
    print(f"\nSaving model to {args.save_model}")
    model.save(args.save_model)
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

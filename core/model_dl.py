"""
Common_Model_DL.py
DL 모델 (1D-CNN LSTM / 1D-CNN Seq2Seq) 학습·예측·평가 공통 모듈

History
  2024-04-05  Created
  2026-03-30  Refactored - import 정리, 중복 함수 제거 (Common_Model_ML에서 import)
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold

from core.model_ml import (
    model_sk_metrics,
    model_visualization,
    _aggregate_kfold_scores,
)


# =========================================================================== #
#  Data Split (DL 전용: 시퀀스 윈도우 생성)
# =========================================================================== #

def _build_sequences(X, Y, seq_len: int):
    """슬라이딩 윈도우로 (X_seq, Y_seq) 배열 생성."""
    x_data, y_data = [], []
    for i in range(len(X) - seq_len + 1):
        x_data.append(X.iloc[i:i + seq_len].values)
        y_data.append(Y.iloc[i + seq_len - 1].values)
    return np.array(x_data), np.array(y_data)


def split_build_dataset(df: pd.DataFrame, test_ratio: float,
                        target_col: str, seq_len: int):
    """Train/Test 분할 후 시퀀스 윈도우 생성."""
    train, test = train_test_split(df, test_size=test_ratio, shuffle=False)
    train_x, train_y = train.drop(columns=[target_col]), train[[target_col]]
    test_x, test_y = test.drop(columns=[target_col]), test[[target_col]]

    train_X, train_Y = _build_sequences(train_x, train_y, seq_len)
    test_X, test_Y = _build_sequences(test_x, test_y, seq_len)
    return train_X, test_X, train_Y, test_Y


# =========================================================================== #
#  Model Builders
# =========================================================================== #

def build_1dcnn_lstm(n_features: int, seq_len: int = 3,
                     activation: str = "swish"):
    """KIER M02 — 1D-CNN + LSTM 모델."""
    inp = tf.keras.layers.Input(shape=(seq_len, n_features))

    x = tf.keras.layers.Conv1D(512, 1, activation=activation)(inp)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(1024, 1, activation=activation)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LSTM(1024, activation=activation, dropout=0.15, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(512,  activation=activation, dropout=0.15, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(256,  activation=activation, dropout=0.15, return_sequences=True)(x)

    for units in [256, 128, 64]:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units, activation=activation)(x)

    out = tf.keras.layers.Dense(1)(tf.keras.layers.BatchNormalization()(x))
    return "1D-CNN_LSTM", tf.keras.models.Model(inp, out)


def build_1dcnn_seq2seq(input_shape, activation: str = "swish"):
    """KIER M02 — 1D-CNN + Seq2Seq 모델."""
    inp = tf.keras.layers.Input(shape=input_shape)

    c1 = tf.keras.layers.Conv1D(1024, 1, activation=activation)(inp)
    p1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(c1)
    b1 = tf.keras.layers.BatchNormalization()(p1)
    c2 = tf.keras.layers.Conv1D(512, 1, activation=activation)(b1)
    p2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(c2)
    b2 = tf.keras.layers.BatchNormalization()(p2)

    # Encoder
    enc1 = tf.keras.layers.LSTM(256, return_sequences=True, activation=activation)(b2)
    enc2 = tf.keras.layers.LSTM(512, return_sequences=True, activation=activation)(b1)
    enc3_out, state_h, state_c = tf.keras.layers.LSTM(
        1024, return_state=True, return_sequences=True, activation=activation)(enc2)

    # Decoder
    dec1 = tf.keras.layers.LSTM(1024, return_sequences=True, activation=activation)(
        enc3_out, initial_state=[state_h, state_c])
    dec2 = tf.keras.layers.LSTM(512, return_sequences=True, activation=activation)(dec1)
    dec3 = tf.keras.layers.LSTM(256, return_sequences=True, activation=activation)(dec2)

    flat = tf.keras.layers.Flatten()(dec3)
    out = tf.keras.layers.Dense(1)(flat)
    return "1D-CNN_Seq2Seq", tf.keras.models.Model(inp, out)


# =========================================================================== #
#  DL: Predict
# =========================================================================== #

def model_dl_predict(train_X, train_Y, test_X, test_Y,
                     model, epochs: int = 500, batch_size: int = 32,
                     patience: int = 15, lr: float = 3e-4):
    """
    DL 모델 학습·예측.

    Returns : (y_actual, y_preds, elapsed_sec)
    """
    t0 = time.time()
    es = EarlyStopping(monitor="loss", patience=patience)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adamax(learning_rate=lr, clipnorm=1.0),
        metrics=["mae"],
    )
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, callbacks=[es])
    elapsed = time.time() - t0

    preds = model.predict(test_X).reshape(-1, 1)
    actual = test_Y.reshape(-1, 1)
    return actual, preds, elapsed


# =========================================================================== #
#  DL: Single Analysis
# =========================================================================== #

def model_dl_analysis_single(df: pd.DataFrame, model,
                             test_ratio: float, target_col: str,
                             seq_len: int):
    train_X, test_X, train_Y, test_Y = split_build_dataset(
        df, test_ratio, target_col, seq_len)
    return model_dl_predict(train_X, train_Y, test_X, test_Y, model)


# =========================================================================== #
#  DL: KFold Analysis
# =========================================================================== #

def model_dl_analysis_with_KFold(df: pd.DataFrame, model_builder,
                                 test_ratio: float, target_col: str,
                                 n_folds: int, seq_len: int,
                                 shuffle: bool = False):
    """
    K-Fold 교차검증 DL 분석.

    Parameters
    ----------
    model_builder : 매 fold마다 새 모델을 반환하는 callable (가중치 초기화를 위해)

    Returns
    -------
    kf_scores : [MAE, MAPE, MSE, RMSE, MSLE, MBE, R2, AvgTime] 평균값
    kf_hists  : 각 지표별 fold 이력 리스트
    """
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    fold_scores, fold_times = [], []

    for _ in kf.split(df):
        model = model_builder()
        actual, preds, elapsed = model_dl_analysis_single(
            df, model, test_ratio, target_col, seq_len)
        fold_scores.append(model_sk_metrics(actual, preds))
        fold_times.append(elapsed)

    return _aggregate_kfold_scores(fold_scores, fold_times)

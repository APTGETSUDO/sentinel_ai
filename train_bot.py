import os

# --- CONFIGURATION SPÉCIALE MAC M1/M2/M3 ---
# 1. Force le mode Legacy pour Keras (Compatibilité Code)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# 2. DÉSACTIVE LE GPU (Force le CPU) pour éviter le blocage "Epoch 1/10"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Attention, Concatenate, Bidirectional
# Utilisation explicite de l'optimiseur Legacy pour Mac
from tensorflow.keras.optimizers.legacy import Adam 
from sklearn.preprocessing import MinMaxScaler
import warnings

# Import du moteur de calcul
from quant_features import HedgeFundIndicators

warnings.filterwarnings("ignore")

# --- PARAMÈTRES ---
TICKER = "NVDA" 
SEQ_LEN = 60    
PREDICT_HORIZON = 1 

SERIES_COLS = ['Close', 'Log_Vol', 'VPIN', 'Returns', 'MFI', 'Tape_Speed']
CONTEXT_COLS = ['Hurst', 'Entropy', 'Volatility', 'ADX', 'Impulse_ATR', 'VBP_Density']

def fix_yfinance_data(df):
    """Nettoyage chirurgical des données"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in required:
        if c not in df.columns:
            if 'Adj Close' in df.columns and c == 'Close':
                df['Close'] = df['Adj Close']
            else:
                return pd.DataFrame()
    return df.dropna(subset=['Close', 'Volume'])

def create_sentinel_model(input_shape_series, input_shape_context):
    """Architecture IA"""
    # Branche Série
    input_series = Input(shape=input_shape_series, name="Time_Series_Input")
    x = Bidirectional(LSTM(64, return_sequences=True))(input_series)
    x = BatchNormalization()(x)
    attn = Attention()([x, x])
    x = LSTM(32, return_sequences=False)(attn)
    
    # Branche Contexte
    input_context = Input(shape=input_shape_context, name="Context_Input")
    y = Dense(16, activation="relu")(input_context)
    y = Dropout(0.2)(y)
    
    # Fusion
    combined = Concatenate()([x, y])
    z = Dense(32, activation="relu")(combined)
    output = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=[input_series, input_context], outputs=output)
    
    # Optimiseur Legacy pour éviter les lenteurs Mac
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def prepare_data(df):
    if df.empty or len(df) < SEQ_LEN + 10: return np.array([]), np.array([]), np.array([])

    scaler_series = MinMaxScaler()
    scaler_context = MinMaxScaler()
    
    try:
        data_series = scaler_series.fit_transform(df[SERIES_COLS])
        data_context = scaler_context.fit_transform(df[CONTEXT_COLS])
    except: return np.array([]), np.array([]), np.array([])
    
    X_series, X_context, y = [], [], []
    
    for i in range(SEQ_LEN, len(df) - PREDICT_HORIZON):
        X_series.append(data_series[i-SEQ_LEN:i])
        X_context.append(data_context[i])
        label = 1 if df['Close'].iloc[i + PREDICT_HORIZON] > df['Close'].iloc[i] else 0
        y.append(label)
        
    return np.array(X_series), np.array(X_context), np.array(y)

def main():
    print(f"🚀 Démarrage Entraînement (CPU MODE - ANTI FREEZE)...")
    
    df = yf.download(TICKER, period="2y", interval="1h", progress=False, auto_adjust=True)
    df = fix_yfinance_data(df)
    
    if df.empty: return
    
    try:
        df = HedgeFundIndicators.add_all_features(df)
    except: return

    if len(df) == 0:
        print("❌ Erreur Calculs.")
        return
        
    X_s, X_c, y = prepare_data(df)
    if len(X_s) == 0: return

    split = int(len(y) * 0.8)
    X_s_train, X_s_val = X_s[:split], X_s[split:]
    X_c_train, X_c_val = X_c[:split], X_c[split:]
    y_train, y_val = y[:split], y[split:]
    
    print("🧠 Initialisation Modèle...")
    model = create_sentinel_model((SEQ_LEN, len(SERIES_COLS)), (len(CONTEXT_COLS),))
    
    print("🏋️ Entraînement en cours (Rapide)...")
    # Verbose = 2 pour éviter de saturer le terminal si ça bloque
    model.fit(
        [X_s_train, X_c_train], y_train,
        validation_data=([X_s_val, X_c_val], y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    model.save("sentinel_v1.keras")
    print("\n✅ SUCCÈS TOTAL. Le modèle est sauvegardé.")
    print("👉 Lancez maintenant : python -m streamlit run terminal.py")

if __name__ == "__main__":
    main()
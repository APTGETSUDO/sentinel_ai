import os
import sys

# --- CONFIGURATION CLOUD (LINUX/GITHUB ACTIONS) ---
# 1. Force le CPU (Les serveurs GitHub gratuits n'ont pas de GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2. Réduit le bruit des logs TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# --------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Attention, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import MinMaxScaler
import warnings

# Import du moteur de calcul (doit être présent dans le même dossier)
from quant_features import HedgeFundIndicators

warnings.filterwarnings("ignore")

# --- PARAMÈTRES ---
TICKER = "NVDA" 
SEQ_LEN = 60    
PREDICT_HORIZON = 1 

SERIES_COLS = ['Close', 'Log_Vol', 'VPIN', 'Returns', 'MFI', 'Tape_Speed']
CONTEXT_COLS = ['Hurst', 'Entropy', 'Volatility', 'ADX', 'Impulse_ATR', 'VBP_Density']

def create_sentinel_model(input_shape_series, input_shape_context):
    """Architecture IA (Standard Keras pour Linux)"""
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
    
    # Optimiseur Standard (Pas Legacy, car on est sur Linux)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def prepare_data(df):
    if df.empty or len(df) < SEQ_LEN + 10: return np.array([]), np.array([]), np.array([])

    scaler_series = MinMaxScaler()
    scaler_context = MinMaxScaler()
    
    # Gestion robuste des NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    try:
        # On s'assure que les colonnes existent
        avail_series = [c for c in SERIES_COLS if c in df.columns]
        avail_context = [c for c in CONTEXT_COLS if c in df.columns]
        
        if len(avail_series) != len(SERIES_COLS) or len(avail_context) != len(CONTEXT_COLS):
            print("❌ Colonnes manquantes dans quant_features.")
            return np.array([]), np.array([]), np.array([])

        data_series = scaler_series.fit_transform(df[avail_series])
        data_context = scaler_context.fit_transform(df[avail_context])
    except Exception as e: 
        print(f"❌ Erreur Scaling: {e}")
        return np.array([]), np.array([]), np.array([])
    
    X_series, X_context, y = [], [], []
    
    for i in range(SEQ_LEN, len(df) - PREDICT_HORIZON):
        X_series.append(data_series[i-SEQ_LEN:i])
        X_context.append(data_context[i])
        # Target : 1 si le prix monte, 0 sinon
        label = 1 if df['Close'].iloc[i + PREDICT_HORIZON] > df['Close'].iloc[i] else 0
        y.append(label)
        
    return np.array(X_series), np.array(X_context), np.array(y)

def main():
    print(f"🚀 GITHUB RUNNER: Démarrage Entraînement sur {TICKER}...")
    
    # 1. Téléchargement Données (Avec Retry Logic implicite via yfinance)
    try:
        df = yf.download(TICKER, period="2y", interval="1h", progress=False, auto_adjust=True)
        # Fix MultiIndex si nécessaire
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        print(f"❌ Erreur YFinance: {e}")
        sys.exit(1) # Force l'échec pour le robot GitHub
    
    if df.empty:
        print("❌ Données vides.")
        sys.exit(1)
    
    # 2. Calcul Indicateurs
    print("⚙️ Calcul des indicateurs quantitatifs...")
    try:
        df = HedgeFundIndicators.add_all_features(df)
    except Exception as e:
        print(f"❌ Erreur Features: {e}")
        sys.exit(1)

    # 3. Préparation
    X_s, X_c, y = prepare_data(df)
    if len(X_s) == 0:
        print("❌ Pas assez de données pour l'entraînement.")
        sys.exit(1)

    # Split 80/20
    split = int(len(y) * 0.8)
    X_s_train, X_s_val = X_s[:split], X_s[split:]
    X_c_train, X_c_val = X_c[:split], X_c[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"🧠 Entraînement sur {len(y_train)} échantillons...")
    
    # 4. Modèle
    model = create_sentinel_model((SEQ_LEN, len(SERIES_COLS)), (len(CONTEXT_COLS),))
    
    # 5. Entraînement "Clean Log"
    # verbose=2 est CRITIQUE pour GitHub : Affiche 1 ligne par époque au lieu d'une barre animée
    model.fit(
        [X_s_train, X_c_train], y_train,
        validation_data=([X_s_val, X_c_val], y_val),
        epochs=15,          # Augmenté légèrement car serveurs rapides
        batch_size=32,
        verbose=2           # <-- L'optimisation clé pour les logs
    )
    
    # 6. Sauvegarde
    model.save("sentinel_v1.keras")
    print("✅ MODÈLE SAUVEGARDÉ : sentinel_v1.keras")

if __name__ == "__main__":
    main()

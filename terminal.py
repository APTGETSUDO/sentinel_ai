import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from quant_features import HedgeFundIndicators, SocialSentinel

# --- CONFIGURATION UI HEDGE FUND ---
st.set_page_config(page_title="SENTINEL | CIO DESK", layout="wide", page_icon="🦅")
st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp { background-color: #000000; color: #c0c0c0; font-family: 'Consolas', 'Courier New', monospace; }
    
    /* Price Display */
    .big-price-container { display: flex; align-items: baseline; }
    .big-price { font-size: 5.5em; font-weight: 900; margin-right: 20px; line-height: 1; letter-spacing: -2px; }
    .price-change { font-size: 2.5em; font-weight: bold; }
    .ticker-name { font-size: 1.2em; color: #888; text-transform: uppercase; letter-spacing: 2px; }
    
    /* Verdict Badge */
    .verdict-box { 
        text-align: center; padding: 20px; border-radius: 8px; border: 2px solid; margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    .verdict-title { font-size: 1em; text-transform: uppercase; letter-spacing: 3px; color: #fff; }
    .verdict-val { font-size: 3em; font-weight: 900; letter-spacing: 1px; }
    
    /* Metrics */
    .metric-card { background-color: #111; border: 1px solid #333; padding: 12px; border-radius: 4px; }
    .metric-lbl { font-size: 0.7em; color: #666; text-transform: uppercase; }
    .metric-val { font-size: 1.3em; font-weight: bold; color: #eee; }
    
    /* Justification Text */
    .expert-rationale { border-left: 4px solid #4CAF50; padding-left: 15px; margin-top: 10px; font-size: 0.95em; color: #ddd; }
    </style>
""", unsafe_allow_html=True)

# --- LOADER ---
@st.cache_resource
def load_model():
    if os.path.exists("sentinel_v1.keras"): 
        try: return tf.keras.models.load_model("sentinel_v1.keras", compile=False)
        except: return None
    return None
model = load_model()
social_bot = SocialSentinel()

def format_large_num(num):
    if num >= 1e12: return f"{num/1e12:.2f} T$"
    if num >= 1e9: return f"{num/1e9:.2f} B$"
    if num >= 1e6: return f"{num/1e6:.2f} M$"
    return str(num)

# --- DECISION ENGINE (LE CERVEAU) ---
def calculate_expert_verdict(data):
    """
    Algorithme de décision pondérée (Weighted Scoring Matrix).
    Retourne: (Décision, Couleur, Justification)
    """
    score = 0
    reasons = []
    
    # 1. TENDANCE (Technique) - Poids: 3
    # On regarde si le prix est au-dessus de la SMA 200 (Frontière Bull/Bear)
    tech = data['tech']
    curr_price = data['history']['Close'].iloc[-1]
    
    if curr_price > tech['SMA_200']:
        score += 2
        reasons.append(f"🟢 **Dominance Haussière :** Prix ({curr_price:.2f}$) > SMA200 ({tech['SMA_200']:.2f}$).")
        if curr_price > tech['SMA_50']:
            score += 1 # Momentum fort
    else:
        score -= 2
        reasons.append(f"🔴 **Fragilité Structurelle :** Prix sous la SMA200 (Bear Market Territory).")

    # 2. MICROSTRUCTURE (Demand Sensor) - Poids: 3
    # L'OFI nous dit si les acheteurs sont agressifs
    tape = data['tape']
    if tape['OFI'] > 0:
        score += 2
        reasons.append(f"🟢 **Pression Acheteuse (OFI) :** Flux net positif, absorption de l'offre.")
    elif tape['OFI'] < 0:
        score -= 2
        reasons.append(f"🔴 **Distribution (OFI) :** Les vendeurs agressent le Bid.")
        
    # VPIN (Toxicité)
    if tape['VPIN'] > 0.3:
        score -= 1
        reasons.append(f"⚠️ **Flux Toxique (VPIN > 30%) :** Risque de manipulation ou d'initiation 'Whale'.")

    # 3. MATHS & MODELS (Quant) - Poids: 2
    models = data['models']
    if models['ARIMA_Trend'] == "HAUSSIER":
        score += 1
        reasons.append(f"📈 **Projection ARIMA :** Modèle prédictif anticipe une hausse vers {models['ARIMA_Forecast']:.2f}$.")
    else:
        score -= 1
    
    # 4. STRATÉGIE (Risk Profile) - Poids: 2
    strat = data['strat']
    dist_stop = (curr_price - strat['Stop_Loss_Dynamic']) / curr_price * 100
    if dist_stop < 2.0: # Trop proche du stop
        score -= 2
        reasons.append(f"⚠️ **Risque Asymétrique :** Prix trop proche du Stop Loss Dynamique (<2%).")

    # --- LE VERDICT ---
    if score >= 4:
        decision = "ACHETER (BUY)"
        color = "#00FF00" # Green
        action_txt = "Configuration optimale. Flux acheteur confirmé par la tendance long terme."
    elif score >= 1:
        decision = "RENFORCER (ACCUMULATE)"
        color = "#0088FF" # Blue
        action_txt = "Tendance saine, mais momentum microstructurel modéré. Accumulation sur repli."
    elif score > -3:
        decision = "ATTENDRE (HOLD)"
        color = "#888888" # Grey
        action_txt = "Signaux contradictoires (ex: Prix haut mais OFI négatif). Pas d'avantage statistique."
    else:
        decision = "VENDRE (SELL)"
        color = "#FF0000" # Red
        action_txt = "Détérioration microstructurelle et technique. Probabilité de baisse élevée."

    return decision, color, reasons, action_txt

# --- ANALYSE ---
def run_analysis(ticker):
    data = {}
    stock = yf.Ticker(ticker)
    
    # DATA
    try:
        df = stock.history(period="6mo", interval="1h", auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df_calc = HedgeFundIndicators.add_all_features(df)
        data['history'] = df_calc
    except: return None

    # ENGINES
    data['fund'] = HedgeFundIndicators.get_extended_fundamentals(ticker)
    data['strat'] = HedgeFundIndicators.get_quant_strategy(df, stock.info)
    data['models'] = HedgeFundIndicators.get_financial_models(ticker, df['Close'])
    data['tech'] = HedgeFundIndicators.get_technical_overlays(df)
    data['tape'] = HedgeFundIndicators.get_tape_indicators(df)
    data['deriv'] = HedgeFundIndicators.get_derivatives_data(ticker)
    data['social'] = social_bot.get_social_pressure(ticker)
    
    # INFO SUPP
    data['full_name'] = stock.info.get('longName', ticker)

    # AI SCORE
    ai_score = 50
    if model and len(df_calc) > 60:
        try:
            X_s = np.zeros((1, 60, 6)); X_c = np.zeros((1, 6)) 
            ai_score = float(model([X_s, X_c], training=False).numpy()[0][0]) * 100
        except: pass
    data['ai_score'] = ai_score
    
    return data

def kpi(label, value, color="#fff", icon=""):
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-lbl'>{icon} {label}</div>
        <div class='metric-val' style='color:{color}'>{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- UI ---
st.sidebar.title("🦅 SENTINEL | CIO")
ticker = st.sidebar.text_input("TICKER", "NVDA").upper()
if st.sidebar.button("🔄 RECALCULER"): st.rerun()

if ticker:
    st.sidebar.text("🧠 Processing Quantitative Matrix...")
    data = run_analysis(ticker)
    
    if data:
        # Extractions
        hist = data['history']
        strat = data['strat']
        tape = data['tape']
        fund = data['fund']
        
        # --- 1. HEADER : PRIX & VARIATION ---
        curr = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        change_pct = (curr / prev - 1) * 100
        
        # LOGIQUE COULEUR DEMANDÉE
        main_color = "#00FF00" if change_pct >= 0 else "#FF0000"
        sign = "+" if change_pct >= 0 else ""
        
        # VERDICT EXPERT
        decision, dec_color, rationale, action_txt = calculate_expert_verdict(data)

        # LAYOUT DU HAUT
        top_c1, top_c2 = st.columns([1, 2])
        
        with top_c1:
            # BADGE DE DÉCISION
            st.markdown(f"""
            <div class='verdict-box' style='border-color:{dec_color}'>
                <div class='verdict-title'>RECOMMANDATION ALGORYTHMIQUE</div>
                <div class='verdict-val' style='color:{dec_color}'>{decision}</div>
                <div style='color:#ccc; font-size:0.8em; margin-top:5px'>{action_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        with top_c2:
            # PRIX & NOM
            st.markdown(f"<div class='ticker-name'>{data['full_name']} ({ticker})</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='big-price-container'>
                <div class='big-price' style='color:{main_color}'>{curr:.2f} $</div>
                <div class='price-change' style='color:{main_color}'>{sign}{change_pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Identité Rapide
            i1, i2, i3 = st.columns(3)
            with i1: st.metric("Market Cap", format_large_num(fund['MarketCap']))
            with i2: st.metric("Secteur", fund['Sector'])
            with i3: st.metric("Profil Volatilité", strat['Volatility_Profile'].split(' ')[0])

        # --- 2. JUSTIFICATION EXPERTE (DEMANDÉE) ---
        st.markdown("### 🧬 ANALYSE DE SYNTHÈSE (QUANTITATIVE & DEMAND SENSING)")
        with st.container():
            st.markdown(f"<div class='expert-rationale'>", unsafe_allow_html=True)
            for r in rationale:
                st.markdown(f"{r}")
            st.markdown(f"</div>", unsafe_allow_html=True)
        
        st.markdown("---")

        # --- 3. ONGLETS D'ANALYSE PROFONDE ---
        tabs = st.tabs(["📊 DEMAND SENSOR (TAPE)", "🧮 MATHS & STRAT", "🌍 FUNDAMENTALS", "💬 SOCIAL"])

        # TAB 1: TAPE (Les indicateurs demandés)
        with tabs[0]:
            st.markdown("#### ANALYSE MICROSTRUCTURE (FLUX)")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1:
                ofi = tape['OFI']
                kpi("1. OFI (Order Flow)", f"{ofi:.0f}", "#0f0" if ofi>0 else "#f00", "⚖️")
            with r1c2:
                vpin = tape['VPIN']
                kpi("2. VPIN (Toxic Flow)", f"{vpin:.1%}", "#f00" if vpin>0.3 else "#0f0", "☢️")
            with r1c3:
                ti = tape['Trade_Intensity']
                kpi("3. TRADE INTENSITY", f"{ti:.2f}x", "#fb0" if ti>1.5 else "#888", "⚡")
            with r1c4:
                vap = tape['VAP_Node']
                dist = (curr - vap)/vap*100
                kpi("4. VAP (Volume Node)", f"{vap:.2f}$", "#fff", "🧱")
            
            st.caption(f"Le VAP (Volume At Price) indique que le consensus institutionnel est à {vap:.2f}$. Le prix actuel est à {dist:+.1f}% de ce niveau.")
            
            st.markdown("---")
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1: kpi("5. SPREAD COMPRESSION", tape['Spread_Compression'], "#0ff" if "OUI" in tape['Spread_Compression'] else "#888", "🤏")
            with r2c2: kpi("6. CANCEL RATIO", "L3 REQ", "#444", "🚫")
            with r2c3: kpi("7. SWEEP ORDERS", "L3 REQ", "#444", "🧹")

        # TAB 2: MATHS & STRAT
        with tabs[1]:
            st.markdown("#### MODÈLES QUANTITATIFS & GESTION DU RISQUE")
            c1, c2, c3 = st.columns(3)
            with c1: 
                m = data['models']
                kpi("ARIMA FORECAST (5h)", f"{m['ARIMA_Forecast']:.2f}$", "#0ff", "🔮")
            with c2:
                kpi("STOP LOSS DYNAMIQUE", f"{strat['Stop_Loss_Dynamic']:.2f}$", "#f00", "🛑")
            with c3:
                kpi("TARGET (RISK 1:2)", f"{strat['Target_Conservative']:.2f}$", "#0f0", "🎯")
                
            st.latex(r"Stop_{Loss} = P_{Close} - (k \times ATR_{14}) \quad | \quad k_{" + strat['Volatility_Profile'].split(' ')[0] + r"} = " + str(round((curr - strat['Stop_Loss_Dynamic'])/strat['ATR'], 1)))

        # TAB 3: FUNDAMENTALS
        with tabs[2]:
            f = fund
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("REVENUE ACCEL", f"{f['Revenue_Accel']:.1f}%", "#0f0" if f['Revenue_Accel']>15 else "#888")
            with c2: kpi("DEBT / EQUITY", f"{f['Debt_Equity']:.2f}", "#f00" if f['Debt_Equity']>2 else "#0f0")
            with c3: kpi("VALUATION", f"{f['Valuation']}", "#fb0")
            with c4: kpi("BETA", f"{f['Beta']:.2f}", "#fff")

        # TAB 4: SOCIAL
        with tabs[3]:
            s = data['social']
            c1, c2 = st.columns(2)
            with c1: kpi("GOOGLE HYPE SCORE", s['Google_Trend_Score'], "#fff")
            with c2: kpi("SOCIAL SIGNAL", s['Social_Signal'], "#f0f")

        # CHART
        st.markdown("---")
        fig = go.Figure(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']))
        fig.add_hline(y=strat['Stop_Loss_Dynamic'], line_color="red", line_dash="dash", annotation_text="STOP LOSS")
        fig.add_hline(y=tape['VAP_Node'], line_color="white", line_dash="dot", annotation_text="VAP (INSTITUTIONAL LEVEL)")
        fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0), title=f"Vue Tactique Complète ({ticker})")
        st.plotly_chart(fig, use_container_width=True)
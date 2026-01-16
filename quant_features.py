import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import linregress
import warnings
import os

# --- SOCIAL IMPORTS ---
try:
    import praw
    from pytrends.request import TrendReq
except ImportError:
    praw = None
    TrendReq = None

warnings.filterwarnings("ignore")
os.environ["TF_USE_LEGACY_KERAS"] = "1"

class SocialSentinel:
    def __init__(self):
        self.reddit_id = "YOUR_ID"
        self.reddit_secret = "YOUR_SECRET"
        self.reddit = None
        if praw and self.reddit_id != "YOUR_ID":
            try: self.reddit = praw.Reddit(client_id=self.reddit_id, client_secret=self.reddit_secret, user_agent="Sentinel")
            except: pass

    def get_social_pressure(self, ticker):
        data = {"Google_Trend_Score": 0, "Google_Trend_Slope": "NEUTRE", "Social_Signal": "CALME", "Reddit_Mentions": 0}
        try:
            if TrendReq:
                pt = TrendReq(hl='en-US', tz=360)
                pt.build_payload([ticker], cat=0, timeframe='now 7-d')
                df = pt.interest_over_time()
                if not df.empty:
                    data["Google_Trend_Score"] = int(df[ticker].iloc[-1])
                    if df[ticker].iloc[-1] > df[ticker].iloc[-2]: data["Google_Trend_Slope"] = "HAUSSE"
        except: pass
        
        if data["Google_Trend_Score"] > 75: data["Social_Signal"] = "🔥 HYPE (Retail)"
        return data

class HedgeFundIndicators:

    # ==============================================================================
    # 1. MICROSTRUCTURE AVANCÉE (TAPE READING)
    # ==============================================================================
    @staticmethod
    def get_tape_indicators(df):
        tape = {
            "OFI": 0.0, "VPIN": 0.0, "Trade_Intensity": 0.0, 
            "Spread_Compression": "NON", "VAP_Node": 0.0,
            "Sweep_Risk": "N/A (L3 Req)", "Cancel_Ratio": "N/A (L3 Req)"
        }
        try:
            # 1. OFI (Order Flow Imbalance) - Proxy
            # Formule : Delta Prix relatif au Range * Volume
            rb = (df['High'] - df['Low']).replace(0, 1e-9)
            tape["OFI"] = (((df['Close'] - df['Open']) / rb) * df['Volume']).rolling(3).mean().iloc[-1]
            
            # 2. VPIN (Toxic Flow Proxy)
            # Volume-Synchronized Probability of Informed Trading
            # Estimation via la volatilité du volume pondéré
            buy_vol = df['Volume'] * (df['Close'] - df['Low']) / rb
            sell_vol = df['Volume'] * (df['High'] - df['Close']) / rb
            diff = (buy_vol - sell_vol).abs().rolling(24).sum() # Bucket 24h
            total = df['Volume'].rolling(24).sum().replace(0, 1e-9)
            tape["VPIN"] = (diff / total).iloc[-1]
            
            # 3. TRADE INTENSITY (Vitesse du Tape)
            # Volume par unité de mouvement (Resistance du carnet)
            intensity = df['Volume'] / rb
            avg_intensity = intensity.rolling(50).mean()
            tape["Trade_Intensity"] = (intensity.iloc[-1] / avg_intensity.iloc[-1]) if avg_intensity.iloc[-1] > 0 else 1.0
            
            # 4. SPREAD COMPRESSION (Bollinger Bandwidth sur le Range)
            # Si le range (High-Low) s'écrase, une explosion arrive
            avg_range = rb.rolling(20).mean().iloc[-1]
            curr_range = rb.iloc[-1]
            if curr_range < (avg_range * 0.5):
                tape["Spread_Compression"] = "OUI (SQUEEZE)"
            
            # 5. VAP (Volume At Price - Point of Control)
            # Histogramme des volumes sur les 100 dernières bougies
            prices = df['Close'].values[-100:]
            volumes = df['Volume'].values[-100:]
            hist, bin_edges = np.histogram(prices, bins=20, weights=volumes)
            max_idx = np.argmax(hist)
            tape["VAP_Node"] = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
            
        except: pass
        return tape

    # ==============================================================================
    # 2. GENERATEUR DE THÈSE (SYNTHÈSE TEXTUELLE)
    # ==============================================================================
    @staticmethod
    def generate_executive_summary(ticker, data):
        """
        Rédige une analyse humaine basée sur les signaux.
        """
        reasons_pos = []
        reasons_neg = []
        
        # A. ANALYSE TECHNIQUE
        tech = data.get('tech', {})
        hist = data.get('history', pd.DataFrame())
        last_price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        if last_price > tech.get('SMA_200', 99999):
            reasons_pos.append("Le prix évolue au-dessus de la SMA 200 (Tendance de fond haussière).")
        else:
            reasons_neg.append("Le prix est sous la SMA 200 (Risque de marché baissier).")
            
        if tech.get('MACD_Val', 0) > tech.get('MACD_Signal', 0):
            reasons_pos.append("Croisement MACD haussier (Momentum positif court terme).")
            
        # B. MICROSTRUCTURE
        tape = data.get('tape', {})
        if tape.get('OFI', 0) > 0:
            reasons_pos.append("L'OFI est positif : Les acheteurs agressifs dominent le carnet d'ordres.")
        elif tape.get('OFI', 0) < 0:
            reasons_neg.append("L'OFI est négatif : Pression vendeuse invisible malgré le prix.")
            
        if tape.get('VPIN', 0) > 0.30:
            reasons_neg.append("VPIN élevé (>30%) : Présence de flux toxiques (Ventes informées probables).")
            
        if tape.get('VAP_Node', 0) > last_price * 1.02:
            reasons_neg.append(f"Mur de volume institutionnel au-dessus du prix ({tape['VAP_Node']:.2f}$), agissant comme résistance.")
        elif tape.get('VAP_Node', 0) < last_price * 0.98:
            reasons_pos.append(f"Support institutionnel solide (VAP) identifié à {tape['VAP_Node']:.2f}$.")

        # C. FONDAMENTAL
        fund = data.get('fund', {})
        if "SOUS" in fund.get('Valuation', ''):
            reasons_pos.append("L'actif est considéré comme Sous-évalué par rapport à sa croissance (PEG < 1).")
        if fund.get('Revenue_Accel', 0) > 20:
            reasons_pos.append("Forte accélération du Chiffre d'Affaires (>20%).")
            
        # SYNTHÈSE FINALE
        score = data.get('final_score', 50)
        sentiment = "POSITIF" if score > 60 else ("NÉGATIF" if score < 40 else "NEUTRE")
        
        txt = f"**THÈSE D'INVESTISSEMENT : {sentiment}**\n\n"
        
        if reasons_pos:
            txt += "✅ **FACTEURS DE HAUSSE :**\n"
            for r in reasons_pos: txt += f"- {r}\n"
        
        txt += "\n"
        if reasons_neg:
            txt += "⚠️ **FACTEURS DE RISQUE :**\n"
            for r in reasons_neg: txt += f"- {r}\n"
            
        if not reasons_pos and not reasons_neg:
            txt += "Aucun signal directionnel fort détecté (Marché en range)."
            
        return txt

    # ==============================================================================
    # 3. STRATÉGIE QUANT (EXISTANT)
    # ==============================================================================
    @staticmethod
    def get_quant_strategy(df, info):
        strat = {}
        try:
            close = df['Close'].iloc[-1]
            
            # ATR
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift(1)).abs()
            tr3 = (df['Low'] - df['Close'].shift(1)).abs()
            atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
            
            # Profil
            mkt_cap = info.get('marketCap', 0)
            k_stop = 3.0 if mkt_cap < 2e9 else (1.5 if mkt_cap > 100e9 else 2.0)
            profile = "AGRESSIF (Small)" if mkt_cap < 2e9 else "CONSERVATEUR (Blue Chip)"
            
            stop_loss = close - (atr * k_stop)
            risk = close - stop_loss
            
            strat = {
                "ATR": round(atr, 2), "Volatility_Profile": profile,
                "Stop_Loss_Dynamic": round(stop_loss, 2),
                "Entry_Zone": round(close, 2),
                "Target_Conservative": round(close + risk*2, 2),
                "Target_Aggressive": round(close + risk*3.5, 2),
                "Risk_Reward_Ratio": "1:2"
            }
        except: strat = {"ATR": 0, "Stop_Loss_Dynamic": 0}
        return strat

    # ==============================================================================
    # 4. ENGINES AUXILIAIRES (MACRO, MODELS, TECH, FUND)
    # ==============================================================================
    @staticmethod
    def get_financial_models(ticker, df_close):
        out = {"Beta": 1.0, "ROI_1Y": 0.0, "ARIMA_Forecast": 0.0, "ARIMA_Trend": "NEUTRE"}
        try:
            if len(df_close) > 252: out["ROI_1Y"] = (df_close.iloc[-1] / df_close.iloc[-252] - 1) * 100
            # ARIMA
            try:
                model = ARIMA(df_close.values, order=(5,1,0))
                model_fit = model.fit()
                f = model_fit.forecast(steps=5)[-1]
                out["ARIMA_Forecast"] = f
                out["ARIMA_Trend"] = "HAUSSIER" if f > df_close.iloc[-1] else "BAISSIER"
            except: pass
            # BETA
            sp500 = yf.download("^GSPC", period="1y", progress=False)['Close']
            if isinstance(sp500.columns, pd.MultiIndex): sp500 = sp500.iloc[:, 0]
            if len(sp500) > 10:
                s_ret = df_close.pct_change().dropna().tail(len(sp500))
                m_ret = sp500.pct_change().dropna().tail(len(s_ret))
                if len(s_ret) == len(m_ret):
                    slope, _, _, _, _ = linregress(m_ret, s_ret)
                    out["Beta"] = round(slope, 2)
        except: pass
        return out

    @staticmethod
    def get_technical_overlays(df):
        tech = {}
        c = df['Close']
        tech['SMA_50'] = c.rolling(50).mean().iloc[-1]
        tech['SMA_200'] = c.rolling(200).mean().iloc[-1]
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        tech['MACD_Val'] = (ema12 - ema26).iloc[-1]
        tech['MACD_Signal'] = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
        return tech

    @staticmethod
    def get_derivatives_data(ticker):
        d = {"Put_Call_Ratio": 0.0, "Gamma_Risk": "N/A"}
        try:
            s = yf.Ticker(ticker)
            if s.options:
                c = s.option_chain(s.options[0])
                vc = c.calls['volume'].sum(); vp = c.puts['volume'].sum()
                if vc > 0: d["Put_Call_Ratio"] = round(vp / vc, 2)
        except: pass
        return d

    @staticmethod
    def get_extended_fundamentals(ticker):
        out = {"Score_Fund": 5, "Sector": "N/A", "Industry": "N/A", "MarketCap": 0, "Beta": 1, "Debt_Equity": 0, "Revenue_Accel": 0, "PE_Ratio": 0, "Valuation": "N/A"}
        try:
            info = yf.Ticker(ticker).info
            out["Sector"] = info.get('sector', 'N/A')
            out["Industry"] = info.get('industry', 'N/A')
            out["MarketCap"] = info.get('marketCap', 0)
            out["Beta"] = info.get('beta', 1)
            out["PE_Ratio"] = info.get('trailingPE', 0)
            peg = info.get('pegRatio', 0)
            if peg > 0:
                if peg < 1.0: out["Valuation"] = "SOUS-ÉVALUÉ (Cheap)"
                elif peg > 2.0: out["Valuation"] = "SUR-ÉVALUÉ (Expensive)"
                else: out["Valuation"] = "JUSTE PRIX (Fair)"
            out["Debt_Equity"] = info.get('debtToEquity', 0)
            out["Revenue_Accel"] = info.get('revenueGrowth', 0) * 100
            score = 5
            if out["Revenue_Accel"] > 20: score += 2
            if "SOUS" in out["Valuation"]: score += 2
            out["Score_Fund"] = min(10, score)
        except: pass
        return out

    @staticmethod
    def add_all_features(df):
        df = df.copy()
        df.columns = [c.capitalize() for c in df.columns]
        for c in ['Open','High','Low','Close','Volume']:
            if c not in df.columns: df[c] = 0.0
            else: df[c] = pd.to_numeric(df[c], errors='coerce').ffill().fillna(0)
        df['Returns'] = df['Close'].pct_change().fillna(0)
        df['Log_Vol'] = np.log(df['Volume'] + 1.0)
        
        # Microstructure Proxy Calculation for History
        rb = (df['High'] - df['Low']).replace(0, 1e-9)
        df['OFI'] = (((df['Close'] - df['Open']) / rb) * df['Volume']).rolling(3).mean().fillna(0)
        
        # Placeholders
        df['MFI']=50; df['VPIN']=0; df['Tape_Speed']=1.0; df['Hurst']=0.5
        df['Entropy']=0; df['ADX']=20; df['Impulse_ATR']=1; df['Volatility']=0.01
        df['VBP_Density']=0; df['Block_Trade']=0; df['Dist_VWAP']=0
        return df.fillna(0)
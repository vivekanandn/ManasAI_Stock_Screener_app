import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Screener - Swing & SMC")

# --- Helper Functions (Data Acquisition, Indicators, Screening) ---

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_nifty500_symbols():
    """Fetches Nifty 500 symbols from a local CSV file."""
    try:
        # Assuming ind_nifty500list.csv is in the same directory as the app
        df = pd.read_csv("ind_nifty500list.csv")
        return [symbol + ".NS" for symbol in df["Symbol"].tolist()]
    except FileNotFoundError:
        st.error("Nifty 500 symbol list (ind_nifty500list.csv) not found. Please place it in the app directory.")
        return []
    except Exception as e:
        st.error(f"Error loading Nifty 500 symbols: {e}")
        return []

@st.cache_data(ttl=900) # Cache for 15 minutes
def get_historical_data_yf(symbol, period="1y", interval="1d"):
    """Fetches historical stock data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        # Ensure correct column names (yfinance might use 'Datetime' or 'Date')
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        if "Date" not in df.columns:
             st.warning(f"Date column not found for {symbol}. Available: {df.columns}")
             return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        # st.warning(f"Could not fetch data for {symbol} using yfinance: {e}")
        return pd.DataFrame()

def calculate_swing_signals(df: pd.DataFrame, swing_length: int = 3) -> pd.DataFrame:
    """
    Calculates buy and sell signals based on the Accurate Swing Trading System Pine Script logic.
    """
    if df.empty or len(df) < swing_length + 1:
        return df.assign(Buy_Signal=False, Sell_Signal=False, Resistance=np.nan, Support=np.nan, TSL=np.nan)

    df_calc = df.copy()
    df_calc["Resistance"] = df_calc["High"].rolling(window=swing_length).max().shift(1)
    df_calc["Support"] = df_calc["Low"].rolling(window=swing_length).min().shift(1)

    avd = []
    for i in range(len(df_calc)):
        if i == 0 or pd.isna(df_calc["Resistance"].iloc[i]) or pd.isna(df_calc["Support"].iloc[i]):
            avd.append(0)
            continue
        current_close = df_calc["Close"].iloc[i]
        prev_resistance = df_calc["Resistance"].iloc[i]
        prev_support = df_calc["Support"].iloc[i]
        if current_close > prev_resistance:
            avd.append(1)
        elif current_close < prev_support:
            avd.append(-1)
        else:
            avd.append(0)
    df_calc["avd"] = avd
    df_calc["avn"] = df_calc["avd"].replace(0, pd.NA).ffill().fillna(0)

    tsl = []
    for i in range(len(df_calc)):
        current_avn = df_calc["avn"].iloc[i]
        current_support = df_calc["Support"].iloc[i]
        current_resistance = df_calc["Resistance"].iloc[i]
        if current_avn == 1 and not pd.isna(current_support):
            tsl.append(current_support)
        elif current_avn == -1 and not pd.isna(current_resistance):
            tsl.append(current_resistance)
        else:
            tsl.append(np.nan)
    df_calc["TSL"] = tsl
    df_calc["TSL"] = df_calc["TSL"].ffill()

    df_calc["Buy_Signal"] = False
    df_calc["Sell_Signal"] = False
    for i in range(1, len(df_calc)):
        prev_close = df_calc["Close"].iloc[i-1]
        curr_close = df_calc["Close"].iloc[i]
        prev_tsl = df_calc["TSL"].iloc[i-1]
        curr_tsl = df_calc["TSL"].iloc[i]
        if not pd.isna(prev_tsl) and not pd.isna(curr_tsl):
            if prev_close < prev_tsl and curr_close > curr_tsl:
                df_calc.loc[df_calc.index[i], "Buy_Signal"] = True
            if prev_close > prev_tsl and curr_close < curr_tsl:
                df_calc.loc[df_calc.index[i], "Sell_Signal"] = True
    
    return df_calc[["Buy_Signal", "Sell_Signal", "Resistance", "Support", "TSL"]]

def find_pivots(df, length=5):
    """Finds pivot highs and lows."""
    df["PivotHigh"] = df["High"].rolling(window=2*length+1, center=True, min_periods=1).apply(lambda x: x.iloc[length] if len(x) == 2*length+1 and x.iloc[length] == x.max() else np.nan, raw=True)
    df["PivotLow"] = df["Low"].rolling(window=2*length+1, center=True, min_periods=1).apply(lambda x: x.iloc[length] if len(x) == 2*length+1 and x.iloc[length] == x.min() else np.nan, raw=True)
    return df

def smc_lite_analysis(df: pd.DataFrame, pivot_length: int = 5) -> pd.DataFrame:
    """Simplified SMC analysis for BOS, CHOCH, Supply/Demand."""
    if df.empty or len(df) < pivot_length * 2 + 1:
        return df.assign(BOS_Bullish=False, BOS_Bearish=False, CHOCH_Bullish=False, CHOCH_Bearish=False, SupplyZone=np.nan, DemandZone=np.nan)

    df_smc = df.copy()
    df_smc = find_pivots(df_smc, length=pivot_length)

    df_smc["BOS_Bullish"] = False
    df_smc["BOS_Bearish"] = False
    df_smc["CHOCH_Bullish"] = False
    df_smc["CHOCH_Bearish"] = False
    df_smc["SupplyZone"] = np.nan
    df_smc["DemandZone"] = np.nan

    last_pivot_high = None
    last_pivot_low = None
    trend = 0 # 0: undefined, 1: uptrend, -1: downtrend

    for i in range(len(df_smc)):
        current_high = df_smc["High"].iloc[i]
        current_low = df_smc["Low"].iloc[i]
        current_pivot_high = df_smc["PivotHigh"].iloc[i]
        current_pivot_low = df_smc["PivotLow"].iloc[i]

        if not pd.isna(current_pivot_high):
            if last_pivot_high is not None and current_pivot_high > last_pivot_high:
                if trend == 1: # Uptrend, BOS
                    df_smc.loc[df_smc.index[i], "BOS_Bullish"] = True
                elif trend == -1: # Downtrend, CHOCH
                    df_smc.loc[df_smc.index[i], "CHOCH_Bullish"] = True
                    df_smc.loc[df_smc.index[i], "DemandZone"] = last_pivot_low # Mark previous low as demand
                trend = 1
            last_pivot_high = current_pivot_high
            df_smc.loc[df_smc.index[i], "SupplyZone"] = current_pivot_high # Mark current high as supply

        if not pd.isna(current_pivot_low):
            if last_pivot_low is not None and current_pivot_low < last_pivot_low:
                if trend == -1: # Downtrend, BOS
                    df_smc.loc[df_smc.index[i], "BOS_Bearish"] = True
                elif trend == 1: # Uptrend, CHOCH
                    df_smc.loc[df_smc.index[i], "CHOCH_Bearish"] = True
                    df_smc.loc[df_smc.index[i], "SupplyZone"] = last_pivot_high # Mark previous high as supply
                trend = -1
            last_pivot_low = current_pivot_low
            df_smc.loc[df_smc.index[i], "DemandZone"] = current_pivot_low # Mark current low as demand
            
    # Forward fill zones for plotting/latest value
    df_smc["SupplyZone"] = df_smc["SupplyZone"].ffill()
    df_smc["DemandZone"] = df_smc["DemandZone"].ffill()

    return df_smc[["BOS_Bullish", "BOS_Bearish", "CHOCH_Bullish", "CHOCH_Bearish", "SupplyZone", "DemandZone"]]

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if df.empty or len(df) < period:
        return df.assign(ATR=np.nan)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    return df.assign(ATR=atr)

def calculate_target_stoploss(df_with_indicators: pd.DataFrame, latest_data: pd.Series) -> dict:
    target = None
    stop_loss = None
    risk_reward_ratio = None
    current_price = latest_data["Close"]
    atr = latest_data.get("ATR", current_price * 0.02)  # Default 2% ATR if not available
    if atr is None or pd.isna(atr) or atr == 0: atr = current_price * 0.02

    # Use last identified supply/demand zones from the full history
    last_supply_zone = df_with_indicators["SupplyZone"].dropna().iloc[-1] if not df_with_indicators["SupplyZone"].dropna().empty else None
    last_demand_zone = df_with_indicators["DemandZone"].dropna().iloc[-1] if not df_with_indicators["DemandZone"].dropna().empty else None

    if latest_data.get("Combined_Buy_Signal", False):
        if last_demand_zone is not None:
            stop_loss = min(last_demand_zone - (atr * 0.5), current_price - (atr * 1.5)) # Tighter SL
        else:
            stop_loss = current_price - (atr * 1.5)
        
        if last_supply_zone is not None and last_supply_zone > current_price:
            target = last_supply_zone
        else:
            target = current_price + (atr * 3) # Default 1:3 R:R if no clear supply
    
    elif latest_data.get("Combined_Sell_Signal", False):
        if last_supply_zone is not None:
            stop_loss = max(last_supply_zone + (atr * 0.5), current_price + (atr * 1.5))
        else:
            stop_loss = current_price + (atr * 1.5)

        if last_demand_zone is not None and last_demand_zone < current_price:
            target = last_demand_zone
        else:
            target = current_price - (atr * 3)

    if target is not None and stop_loss is not None and current_price != stop_loss:
        if latest_data.get("Combined_Buy_Signal", False):
            risk_reward_ratio = (target - current_price) / (current_price - stop_loss)
        elif latest_data.get("Combined_Sell_Signal", False):
             risk_reward_ratio = (current_price - target) / (stop_loss - current_price)

    return {
        "target": round(target, 2) if target is not None else None,
        "stop_loss": round(stop_loss, 2) if stop_loss is not None else None,
        "risk_reward_ratio": round(risk_reward_ratio, 2) if risk_reward_ratio is not None else None
    }

@st.cache_data(ttl=900) # Cache results for 15 mins
def screen_stocks_streamlit(symbols_to_scan, period, interval, swing_length, smc_pivot_length, atr_period):
    suggestions = []
    progress_bar = st.progress(0)
    total_symbols = len(symbols_to_scan)

    for i, symbol in enumerate(symbols_to_scan):
        try:
            df_hist = get_historical_data_yf(symbol, period=period, interval=interval)
            if df_hist.empty or len(df_hist) < max(swing_length + 1, smc_pivot_length * 2 + 1, atr_period):
                continue

            df_swing = calculate_swing_signals(df_hist.copy(), swing_length=swing_length)
            df_smc = smc_lite_analysis(df_hist.copy(), pivot_length=smc_pivot_length)
            df_atr = calculate_atr(df_hist.copy(), period=atr_period)
            
            # Combine results - ensure indices align if they were modified
            df_combined = pd.concat([df_hist, df_swing, df_smc, df_atr["ATR"]], axis=1)
            
            if df_combined.empty:
                continue
            latest_data = df_combined.iloc[-1]

            swing_buy = latest_data.get("Buy_Signal", False)
            swing_sell = latest_data.get("Sell_Signal", False)
            smc_bullish = latest_data.get("CHOCH_Bullish", False) or latest_data.get("BOS_Bullish", False)
            smc_bearish = latest_data.get("CHOCH_Bearish", False) or latest_data.get("BOS_Bearish", False)

            combined_buy_signal = swing_buy and smc_bullish
            combined_sell_signal = swing_sell and smc_bearish
            
            latest_data_for_tsl = latest_data.copy() # Create a mutable copy
            latest_data_for_tsl["Combined_Buy_Signal"] = combined_buy_signal
            latest_data_for_tsl["Combined_Sell_Signal"] = combined_sell_signal

            target_sl_data = calculate_target_stoploss(df_combined, latest_data_for_tsl)

            score = 0
            signal_type = "NEUTRAL"
            if combined_buy_signal:
                score += 2
                if not pd.isna(latest_data.get("DemandZone")):
                    score +=1
                signal_type = "BUY"
            elif combined_sell_signal:
                score += 2
                if not pd.isna(latest_data.get("SupplyZone")):
                    score +=1
                signal_type = "SELL"
            elif swing_buy:
                score +=1
                signal_type = "SWING_BUY"
            elif swing_sell:
                score +=1
                signal_type = "SWING_SELL"
            elif smc_bullish:
                score +=1
                signal_type = "SMC_BULLISH"
            elif smc_bearish:
                score +=1
                signal_type = "SMC_BEARISH"

            if score > 0:
                suggestions.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Signal_Type": signal_type,
                    "Latest_Close": latest_data["Close"],
                    "Swing_Buy": swing_buy,
                    "Swing_Sell": swing_sell,
                    "SMC_Bullish": smc_bullish,
                    "SMC_Bearish": smc_bearish,
                    "BOS_Bullish": latest_data.get("BOS_Bullish", False),
                    "BOS_Bearish": latest_data.get("BOS_Bearish", False),
                    "CHOCH_Bullish": latest_data.get("CHOCH_Bullish", False),
                    "CHOCH_Bearish": latest_data.get("CHOCH_Bearish", False),
                    "Supply_Zone": latest_data.get("SupplyZone"),
                    "Demand_Zone": latest_data.get("DemandZone"),
                    "Target": target_sl_data["target"],
                    "Stop_Loss": target_sl_data["stop_loss"],
                    "Risk_Reward": target_sl_data["risk_reward_ratio"],
                    "Score": score,
                    "Date": latest_data["Date"].strftime("%Y-%m-%d")
                })
        except Exception as e:
            # st.warning(f"Error processing {symbol}: {e}")
            pass # Suppress individual errors during scan for cleaner UI
        finally:
            progress_bar.progress((i + 1) / total_symbols)
    
    progress_bar.empty() # Clear progress bar after completion
    return pd.DataFrame(suggestions)

# --- Streamlit UI ---
st.title("📈 Stock Screener for Swing Trading & SMC")
st.markdown("Analyzes Nifty 500 stocks for potential buy/sell opportunities based on Accurate Swing Trading and Smart Money Concepts.")

# Sidebar for Inputs
st.sidebar.header("Scan Parameters")

# Stock Selection
all_nifty500_symbols = get_nifty500_symbols()
if not all_nifty500_symbols:
    st.sidebar.error("Could not load Nifty 500 symbols. Screening disabled.")
    st.stop()

scan_mode = st.sidebar.radio("Stock Universe", ("Nifty 500 (Sample - First 50)", "Nifty 500 (Full Scan)", "Custom List"), index=0)

custom_symbols_input = ""
if scan_mode == "Custom List":
    custom_symbols_input = st.sidebar.text_area("Enter stock symbols (comma-separated, e.g., RELIANCE,TCS,INFY)", "RELIANCE,TCS")

# Time Period and Interval
period_options = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
selected_period_label = st.sidebar.selectbox("Historical Data Period", list(period_options.keys()), index=2)
period = period_options[selected_period_label]

interval_options = {"Daily": "1d", "Weekly": "1wk"}
selected_interval_label = st.sidebar.selectbox("Data Interval", list(interval_options.keys()), index=0)
interval = interval_options[selected_interval_label]

# Indicator Parameters
st.sidebar.subheader("Indicator Settings")
swing_len = st.sidebar.slider("Swing Indicator Length", 1, 20, 3)
smc_pivot_len = st.sidebar.slider("SMC Pivot Length", 3, 15, 5)
atr_len = st.sidebar.slider("ATR Period for T/SL", 5, 30, 14)

# Scan Button
if st.sidebar.button("🚀 Screen Stocks", use_container_width=True, type="primary"):
    symbols_to_scan_final = []
    if scan_mode == "Nifty 500 (Sample - First 50)":
        symbols_to_scan_final = all_nifty500_symbols[:50]
        st.info("Scanning a sample of the first 50 Nifty 500 stocks.")
    elif scan_mode == "Nifty 500 (Full Scan)":
        symbols_to_scan_final = all_nifty500_symbols
        st.info(f"Scanning all {len(all_nifty500_symbols)} Nifty 500 stocks. This may take some time...")
    elif scan_mode == "Custom List":
        custom_symbols = [s.strip().upper() + ".NS" for s in custom_symbols_input.split(",") if s.strip()]
        if not custom_symbols:
            st.error("Please enter valid stock symbols for custom scan.")
            st.stop()
        symbols_to_scan_final = custom_symbols
        st.info(f"Scanning custom list: {', '.join([s.replace('.NS','') for s in custom_symbols_final])}")

    if not symbols_to_scan_final:
        st.warning("No symbols selected for scanning.")
    else:
        with st.spinner("Analyzing stocks... Please wait."):
            results_df = screen_stocks_streamlit(symbols_to_scan_final, period, interval, swing_len, smc_pivot_len, atr_len)
        
        st.session_state.results = results_df
        st.session_state.last_scan_params = {
            "period": selected_period_label,
            "interval": selected_interval_label,
            "swing_len": swing_len,
            "smc_pivot_len": smc_pivot_len,
            "atr_len": atr_len,
            "scan_mode": scan_mode,
            "num_symbols": len(symbols_to_scan_final)
        }

# Display Results
if "results" in st.session_state:
    results_df = st.session_state.results
    last_params = st.session_state.last_scan_params
    
    st.subheader(f"Scan Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    st.markdown(f"**Parameters**: Period: `{last_params['period']}`, Interval: `{last_params['interval']}`, Swing: `{last_params['swing_len']}`, SMC Pivot: `{last_params['smc_pivot_len']}`, ATR: `{last_params['atr_len']}`, Mode: `{last_params['scan_mode']}` ({last_params['num_symbols']} symbols scanned)")

    if results_df.empty:
        st.warning("No stocks met the screening criteria with the current settings.")
    else:
        # Sort and filter
        sort_by = st.selectbox("Sort results by", ["Score", "Symbol", "Signal_Type", "Risk_Reward"], index=0)
        sort_ascending = st.checkbox("Sort Ascending", False if sort_by in ["Score", "Risk_Reward"] else True)
        
        results_df_sorted = results_df.sort_values(by=sort_by, ascending=sort_ascending)
        
        st.metric("Total Suggestions", len(results_df_sorted))
        
        # Display as cards
        cols_per_row = st.slider("Suggestions per row", 1, 4, 3)
        cols = st.columns(cols_per_row)
        
        for i, row in enumerate(results_df_sorted.iterrows()):
            idx, stock = row
            col = cols[i % cols_per_row]
            
            with col.container(border=True):
                st.markdown(f"#### {stock['Symbol']} <span style='font-size: small; color: {'green' if stock['Signal_Type'] in ['BUY', 'SWING_BUY', 'SMC_BULLISH'] else 'red' if stock['Signal_Type'] in ['SELL', 'SWING_SELL', 'SMC_BEARISH'] else 'gray'};'>({stock['Signal_Type']})</span>", unsafe_allow_html=True)
                st.markdown(f"**Close:** ₹{stock['Latest_Close']:.2f} | **Score:** {stock['Score']}")
                
                with st.expander("Details & Indicators"):
                    st.markdown(f"**Date:** {stock['Date']}")
                    st.markdown("**Swing Signals:**")
                    c1, c2 = st.columns(2)
                    c1.metric("Buy Signal", "✅ Yes" if stock['Swing_Buy'] else "❌ No")
                    c2.metric("Sell Signal", "✅ Yes" if stock['Swing_Sell'] else "❌ No")
                    
                    st.markdown("**SMC Indicators:**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("BOS Bull", "✅" if stock['BOS_Bullish'] else "-")
                    c2.metric("BOS Bear", "✅" if stock['BOS_Bearish'] else "-")
                    c3.metric("CHOCH Bull", "✅" if stock['CHOCH_Bullish'] else "-")
                    c4.metric("CHOCH Bear", "✅" if stock['CHOCH_Bearish'] else "-")
                    
                    if not pd.isna(stock['Supply_Zone']):
                        st.markdown(f"**Supply Zone:** ~₹{stock['Supply_Zone']:.2f}")
                    if not pd.isna(stock['Demand_Zone']):
                        st.markdown(f"**Demand Zone:** ~₹{stock['Demand_Zone']:.2f}")

                if not pd.isna(stock['Target']) or not pd.isna(stock['Stop_Loss']):
                    with st.expander("Target & Stop Loss"):
                        if not pd.isna(stock['Target']):
                            st.markdown(f"**Target:** <span style='color: green;'>₹{stock['Target']:.2f}</span>", unsafe_allow_html=True)
                        if not pd.isna(stock['Stop_Loss']):
                            st.markdown(f"**Stop Loss:** <span style='color: red;'>₹{stock['Stop_Loss']:.2f}</span>", unsafe_allow_html=True)
                        if not pd.isna(stock['Risk_Reward']):
                            st.markdown(f"**Risk/Reward Ratio:** 1 : {stock['Risk_Reward']:.2f}")
        
        st.subheader("Tabular Data")
        st.dataframe(results_df_sorted, use_container_width=True)

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(results_df_sorted)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"stock_screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
else:
    st.info("Click 'Screen Stocks' in the sidebar to begin analysis.")

st.sidebar.markdown("---_" * 10)
st.sidebar.markdown("**Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. Always do your own research.")

# For deployment, ensure ind_nifty500list.csv is in the root directory of your Streamlit app.
# Example: Create a file named `ind_nifty500list.csv` with a header "Symbol" and list stock symbols like:
# Symbol
# RELIANCE
# TCS
# HDFCBANK
# ...etc.



import os
import math
from pathlib import Path
import json

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from neo_api_client import NeoAPI


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

app = FastAPI(title="StockScore Live - 100pt Model")

neo = None
logged_in = False


# =========================================================
# REQUEST MODELS
# =========================================================

class LoginRequest(BaseModel):
    totp: str


class StockRequest(BaseModel):
    company: str


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def num(v):
    try:
        if v is None:
            return None
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def clean_token(val):
    if val is None:
        return None
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val).strip()


def extract_fallback_metrics(t: yf.Ticker, info: dict):
    """
    Fallback extractor for NSE metrics missing in yfinance .info
    Calculates Current Ratio, FCF, ROE, Earnings Growth, and PEG directly
    from financial statements DataFrames.
    """
    bs = getattr(t, "balance_sheet", pd.DataFrame())
    cf = getattr(t, "cashflow", pd.DataFrame())
    fin = getattr(t, "financials", pd.DataFrame())
    q_fin = getattr(t, "quarterly_financials", pd.DataFrame())

    # 1. Current Ratio Fallback
    current_ratio = num(info.get("currentRatio"))
    if current_ratio is None and not bs.empty:
        try:
            curr_assets = num(bs.loc["Current Assets"].iloc[0]) if "Current Assets" in bs.index else None
            curr_liab = num(bs.loc["Current Liabilities"].iloc[0]) if "Current Liabilities" in bs.index else None
            if curr_assets and curr_liab and curr_liab > 0:
                current_ratio = round(curr_assets / curr_liab, 2)
        except Exception:
            pass

    # 2. Free Cash Flow (FCF) Fallback
    fcf = num(info.get("freeCashflow"))
    if fcf is None and not cf.empty:
        try:
            ocf = num(cf.loc["Operating Cash Flow"].iloc[0]) if "Operating Cash Flow" in cf.index else None
            capex = num(cf.loc["Capital Expenditure"].iloc[0]) if "Capital Expenditure" in cf.index else 0
            if ocf is not None:
                # Capex is usually reported as negative in yfinance
                fcf = ocf + capex if capex < 0 else ocf - capex
        except Exception:
            pass

    # 3. ROE Fallback
    roe = num(info.get("returnOnEquity"))
    if roe is None and not fin.empty and not bs.empty:
        try:
            net_income = num(fin.loc["Net Income Common Stockholders"].iloc[0]) if "Net Income Common Stockholders" in fin.index else num(fin.loc["Net Income"].iloc[0]) if "Net Income" in fin.index else None
            equity = num(bs.loc["Stockholders Equity"].iloc[0]) if "Stockholders Equity" in bs.index else None
            if net_income and equity and equity > 0:
                roe = net_income / equity
        except Exception:
            pass

    # 4. Earnings Growth Fallback (YoY Quarterly Growth)
    eg = num(info.get("earningsGrowth"))
    if eg is None and not q_fin.empty and q_fin.shape[1] >= 5:
        try:
            row_key = "Net Income Common Stockholders" if "Net Income Common Stockholders" in q_fin.index else "Net Income"
            if row_key in q_fin.index:
                recent_q = num(q_fin.loc[row_key].iloc[0])
                prev_year_q = num(q_fin.loc[row_key].iloc[4])
                if recent_q and prev_year_q and prev_year_q > 0:
                    eg = (recent_q - prev_year_q) / abs(prev_year_q)
        except Exception:
            pass

    # 5. PEG Ratio Fallback
    peg = num(info.get("pegRatio"))
    pe = num(info.get("trailingPE"))
    if peg is None and pe and eg and eg > 0:
        eg_pct = eg * 100
        peg = round(pe / eg_pct, 2)

    return {
        "current_ratio": current_ratio,
        "fcf": fcf,
        "roe": roe,
        "earnings_growth": eg,
        "peg": peg
    }


# =========================================================
# KOTAK NEO LOGIN
# =========================================================

def login_neo(totp):
    global neo, logged_in

    if len(totp) != 6 or not totp.isdigit():
        raise HTTPException(
            status_code=400,
            detail="TOTP must be exactly 6 digits."
        )

    neo = NeoAPI(
        environment="prod",
        access_token=None,
        neo_fin_key=None,
        consumer_key=os.getenv("KOTAK_CONSUMER_KEY")
    )

    try:
        r = neo.totp_login(
            mobile_number=os.getenv("KOTAK_MOBILE"),
            ucc=os.getenv("KOTAK_UCC"),
            totp=totp
        )

        if isinstance(r, dict) and "error" in r:
            raise HTTPException(
                status_code=401,
                detail=str(r)
            )

        r = neo.totp_validate(
            mpin=os.getenv("KOTAK_MPIN")
        )

        if isinstance(r, dict) and "error" in r:
            raise HTTPException(
                status_code=401,
                detail=str(r)
            )

        logged_in = True
        return {
            "ok": True,
            "message": "Kotak Neo login successful"
        }

    except HTTPException:
        raise

    except Exception as e:
        logged_in = False
        raise HTTPException(
            status_code=401,
            detail=f"Kotak Neo login failed: {str(e)}"
        )


# =========================================================
# FIND NSE STOCK (UPDATED MATCHING LOGIC & BLOCK DEAL FILTER)
# =========================================================

def find_stock(symbol):
    symbol = symbol.upper().strip()
    print(f"\n--- Searching Kotak Neo for: '{symbol}' ---")
    
    queries_to_try = [f"{symbol}-EQ", symbol]
    segments_to_try = ["nse_cm", "NSE", "bse_cm", "BSE"]
    
    valid_items = []

    for seg in segments_to_try:
        for q in queries_to_try:
            try:
                res = neo.search_scrip(
                    exchange_segment=seg,
                    symbol=q,
                    expiry="",
                    option_type="",
                    strike_price=""
                )
                if not res:
                    continue

                if isinstance(res, str):
                    try:
                        res = json.loads(res)
                    except Exception:
                        continue

                items = []
                if isinstance(res, dict):
                    for k in ("data", "result", "scrips", "values", "items"):
                        if k in res and isinstance(res[k], list):
                            items = res[k]
                            break
                    if not items and ("pTrdSymbol" in res or "pSymbol" in res):
                        items = [res]
                elif isinstance(res, list):
                    items = res

                if items:
                    valid_items.extend(items)
            except Exception as e:
                print(f"[SEARCH ERROR] seg='{seg}', query='{q}': {e}")

    if not valid_items:
        print(f"[SEARCH FAILED] No response from Kotak Neo for '{symbol}'")
        return None

    # Deduplicate results
    unique_items = []
    seen = set()
    for item in valid_items:
        if isinstance(item, dict):
            key = item.get("pTrdSymbol") or item.get("pSymbol") or str(item)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

    def get_token(item):
        raw = (item.get("pSymbol") or item.get("instrument_token") or 
               item.get("token") or item.get("tok") or item.get("pCode") or item.get("pScripRefKey"))
        return clean_token(raw)

    def get_trd_sym(item):
        return str(item.get("pTrdSymbol") or item.get("trading_symbol") or item.get("trdSym") or item.get("symbol") or "").upper()

    def get_sym_name(item):
        return str(item.get("pSymbolName") or item.get("pDesc") or item.get("symbol") or item.get("name") or "").upper()

    def get_exch(item):
        return str(item.get("pExchSeg") or item.get("exchange_segment") or item.get("exch") or "nse_cm").lower()

    target_eq = f"{symbol}-EQ"

    # Pass 1: Strict Exact Match for Main Equity Trading Symbol (SYMBOL-EQ)
    for item in unique_items:
        token = get_token(item)
        trd_sym = get_trd_sym(item)
        exch = get_exch(item)

        if token and trd_sym == target_eq:
            print(f"[MATCH FOUND - Pass 1 (Exact EQ)] {trd_sym} | Token: {token} | Exch: {exch}")
            return {
                "pSymbolName": symbol,
                "pTrdSymbol": trd_sym,
                "pSymbol": token,
                "pExchSeg": exch if "nse" in exch else "nse_cm"
            }

    # Pass 2: Fallback Exact Match on Symbol / Symbol Name (Excluding Derivatives & Block Deals)
    for item in unique_items:
        token = get_token(item)
        trd_sym = get_trd_sym(item)
        sym_name = get_sym_name(item)
        exch = get_exch(item)

        is_non_equity = any(trd_sym.endswith(ext) or ext in trd_sym for ext in ("-BL", "-FUT", "-CE", "-PE", "-N1", "-N2", "-E1", "-BE", "-BZ"))

        if token and not is_non_equity and (trd_sym == symbol or sym_name == symbol):
            print(f"[MATCH FOUND - Pass 2 (Clean Symbol)] {trd_sym} | Token: {token} | Exch: {exch}")
            return {
                "pSymbolName": symbol,
                "pTrdSymbol": trd_sym,
                "pSymbol": token,
                "pExchSeg": exch if "nse" in exch else "nse_cm"
            }

    # Pass 3: General Cash Market Equity Fallback
    for item in unique_items:
        token = get_token(item)
        trd_sym = get_trd_sym(item)
        sym_name = get_sym_name(item)
        exch = get_exch(item)

        is_non_equity = any(trd_sym.endswith(ext) or ext in trd_sym for ext in ("-BL", "-FUT", "-CE", "-PE", "-N1", "-N2", "-E1"))

        if token and not is_non_equity and (symbol in trd_sym or symbol in sym_name):
            print(f"[MATCH FOUND - Pass 3 (Fallback)] {trd_sym} | Token: {token} | Exch: {exch}")
            return {
                "pSymbolName": symbol,
                "pTrdSymbol": trd_sym,
                "pSymbol": token,
                "pExchSeg": exch if "nse" in exch else "nse_cm"
            }

    print(f"[SEARCH FAILED] No valid main equity token extracted for '{symbol}'")
    return None


# =========================================================
# INTELLIGENT 100-POINT SCORING ENGINE
# =========================================================

def calculate_rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    return num(result.iloc[-1])


def analyze_fundamentals(info, ticker):
    pts = 0
    max_pts = 40
    data_points_found = 0
    total_metrics = 7
    positives = []
    warnings = []

    fallbacks = extract_fallback_metrics(ticker, info)

    pe = num(info.get("trailingPE"))
    ind_pe = num(info.get("trailingPegRatio")) or num(info.get("forwardPE"))
    roe = fallbacks["roe"]
    rg = num(info.get("revenueGrowth"))
    eg = fallbacks["earnings_growth"]
    de = num(info.get("debtToEquity"))
    net_income = num(info.get("netIncomeToCommon"))
    ocf = num(info.get("operatingCashflow"))

    if pe is not None:
        data_points_found += 1
        if ind_pe and ind_pe > 0:
            pe_diff = ((pe - ind_pe) / ind_pe) * 100
            if pe_diff < -15:
                pts += 8
                positives.append(f"PE ({pe:.1f}) is significantly lower than Forward/Industry PE ({ind_pe:.1f})")
            elif pe_diff <= 5:
                pts += 6
                positives.append(f"PE ({pe:.1f}) is reasonably aligned with Sector/Forward PE ({ind_pe:.1f})")
            elif pe_diff <= 25:
                pts += 3
            else:
                warnings.append(f"PE ({pe:.1f}) is trading at a high premium vs Sector ({ind_pe:.1f})")
        else:
            if pe <= 18:
                pts += 8
                positives.append(f"Attractive PE ratio of {pe:.1f}")
            elif pe <= 28:
                pts += 6
            elif pe <= 40:
                pts += 3
            else:
                warnings.append(f"High PE valuation ({pe:.1f})")

    if eg is not None:
        data_points_found += 1
        eg_pct = eg * 100
        if eg_pct >= 25:
            pts += 7
            positives.append(f"Exceptional quarterly EPS Growth (+{eg_pct:.1f}%)")
        elif eg_pct >= 15:
            pts += 5
            positives.append(f"Solid EPS Growth (+{eg_pct:.1f}%)")
        elif eg_pct >= 5:
            pts += 3
        elif eg_pct < 0:
            warnings.append(f"Earnings contraction (-{abs(eg_pct):.1f}%)")

    if rg is not None:
        data_points_found += 1
        rg_pct = rg * 100
        if rg_pct >= 15:
            pts += 5
            positives.append(f"Strong Revenue Growth (+{rg_pct:.1f}%)")
        elif rg_pct >= 8:
            pts += 3
        elif rg_pct < 0:
            warnings.append(f"Declining Revenue (-{abs(rg_pct):.1f}%)")

    if roe is not None:
        data_points_found += 1
        roe_pct = roe * 100
        if roe_pct >= 20:
            pts += 8
            positives.append(f"High Return on Equity (ROE {roe_pct:.1f}%)")
        elif roe_pct >= 14:
            pts += 6
            positives.append(f"Healthy ROE of {roe_pct:.1f}%")
        elif roe_pct >= 8:
            pts += 4
        else:
            warnings.append(f"Low ROE ({roe_pct:.1f}%) indicates lower capital efficiency")

    if de is not None:
        data_points_found += 1
        if de <= 30:
            pts += 6
            positives.append(f"Low Debt/Equity ratio ({de:.1f}) - Minimal leverage risk")
        elif de <= 70:
            pts += 4
        elif de <= 120:
            pts += 2
        else:
            warnings.append(f"High Debt to Equity ratio ({de:.1f})")

    if ocf is not None and net_income is not None:
        data_points_found += 2
        if ocf > 0 and net_income > 0:
            quality_ratio = ocf / net_income
            if quality_ratio >= 0.9:
                pts += 6
                positives.append("High Earnings Quality (Operating Cash Flow aligns well with Net Profit)")
            elif quality_ratio >= 0.6:
                pts += 4
            else:
                warnings.append("Profit-Cash Flow Mismatch: Cash flow from operations is significantly lower than reported Net Profit")
        elif ocf <= 0 and net_income > 0:
            warnings.append("Negative Operating Cash Flow despite reported profits")

    conf = (data_points_found / total_metrics)

    return {
        "score": pts,
        "max": max_pts,
        "confidence": conf,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "pe": pe,
            "industry_pe": ind_pe,
            "roe_pct": round(roe * 100, 2) if roe is not None else None,
            "revenue_growth_pct": round(rg * 100, 2) if rg is not None else None,
            "earnings_growth_pct": round(eg * 100, 2) if eg is not None else None,
            "debt_to_equity": de
        }
    }


def analyze_technicals(hist):
    if hist is None or hist.empty or len(hist) < 50:
        return {"score": 0, "max": 25, "confidence": 0.0, "positives": [], "warnings": [], "data": {}}

    positives = []
    warnings = []
    pts = 0

    c = hist["Close"].dropna()
    v = hist["Volume"].dropna()
    last_price = float(c.iloc[-1])

    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else sma50
    rv = calculate_rsi(c)

    if last_price > sma50: pts += 3
    if last_price > sma200: pts += 3
    if sma50 > sma200: pts += 2

    if last_price > sma50 > sma200:
        positives.append("Strong Bullish Trend Structure (Price > SMA50 > SMA200)")
    elif last_price < sma50 < sma200:
        warnings.append("Bearish Trend Alignment (Price < SMA50 < SMA200)")

    if rv is not None:
        if 55 <= rv <= 70:
            pts += 5
            positives.append(f"Healthy Bullish RSI zone ({rv:.1f})")
        elif 45 <= rv < 55:
            pts += 3
        elif rv > 75:
            pts += 2
            warnings.append(f"RSI Overbought ({rv:.1f}) - Short-term pullback risk")
        elif rv < 30:
            pts += 1
            warnings.append(f"RSI Oversold ({rv:.1f}) - Strong weakness but potential bounce zone")

    m20 = ((last_price / float(c.iloc[-21])) - 1) * 100 if len(c) >= 21 else 0
    m60 = ((last_price / float(c.iloc[-61])) - 1) * 100 if len(c) >= 61 else 0

    if m20 >= 4:
        pts += 3
        positives.append(f"Positive 20-day momentum (+{m20:.1f}%)")
    elif m20 < -5:
        warnings.append(f"Negative short-term momentum ({m20:.1f}% 20D return)")

    if m60 >= 8:
        pts += 3
    elif m60 < 0:
        pts += 0

    avg_vol_20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
    latest_vol = float(v.iloc[-1])
    if avg_vol_20 > 0:
        vol_ratio = latest_vol / avg_vol_20
        if vol_ratio >= 1.5 and m20 > 0:
            pts += 4
            positives.append(f"High Volume Expansion ({vol_ratio:.1f}x 20-day avg volume)")
        elif vol_ratio >= 1.0:
            pts += 2

    high_52 = float(c.max())
    low_52 = float(c.min())
    dist_high = ((last_price - high_52) / high_52) * 100
    dist_low = ((last_price - low_52) / low_52) * 100

    if dist_high >= -5:
        warnings.append("Trading near 52-Week High resistance zone")
    elif dist_low <= 8:
        positives.append("Trading close to 52-Week Low support base")
        pts += 2

    return {
        "score": min(pts, 25),
        "max": 25,
        "confidence": 1.0,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "rsi": round(rv, 2) if rv is not None else None,
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "momentum_20d_pct": round(m20, 2),
            "momentum_60d_pct": round(m60, 2),
            "dist_52w_high_pct": round(dist_high, 2),
            "dist_52w_low_pct": round(dist_low, 2)
        }
    }


def analyze_valuation(info, ticker):
    positives = []
    warnings = []
    pts = 0
    max_pts = 20
    found = 0

    fallbacks = extract_fallback_metrics(ticker, info)
    peg = fallbacks["peg"]
    pb = num(info.get("priceToBook"))
    fcf = fallbacks["fcf"]
    mcap = num(info.get("marketCap"))

    if peg is not None and peg > 0:
        found += 1
        if peg <= 1.0:
            pts += 7
            positives.append(f"Attractive PEG ratio ({peg:.2f} <= 1.0) - Growth at reasonable price")
        elif peg <= 1.5:
            pts += 5
        elif peg <= 2.2:
            pts += 3
        else:
            warnings.append(f"High PEG ratio ({peg:.2f}) indicates overvaluation relative to growth")

    if fcf is not None and mcap is not None and mcap > 0:
        found += 1
        fcf_yield = (fcf / mcap) * 100
        if fcf_yield >= 4.5:
            pts += 7
            positives.append(f"Strong Free Cash Flow Yield ({fcf_yield:.2f}%)")
        elif fcf_yield >= 2.0:
            pts += 5
        elif fcf_yield > 0:
            pts += 2
        else:
            warnings.append("Negative Free Cash Flow Yield")

    if pb is not None and pb > 0:
        found += 1
        if pb <= 2.5:
            pts += 6
            positives.append(f"Reasonable Price to Book value ({pb:.2f})")
        elif pb <= 5.0:
            pts += 4
        elif pb <= 8.0:
            pts += 2
        else:
            warnings.append(f"Elevated Price-to-Book multiple ({pb:.2f})")

    conf = (found / 3.0)

    return {
        "score": min(pts, 20),
        "max": max_pts,
        "confidence": conf,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "peg_ratio": peg,
            "price_to_book": pb,
            "fcf_yield_pct": round((fcf / mcap) * 100, 2) if (fcf and mcap) else None
        }
    }


def analyze_risk(info, hist, ticker):
    positives = []
    warnings = []
    pts = 0

    fallbacks = extract_fallback_metrics(ticker, info)
    beta = num(info.get("beta"))
    current_ratio = fallbacks["current_ratio"]

    if beta is not None:
        if 0.5 <= beta <= 1.1:
            pts += 5
            positives.append(f"Moderate Market Beta ({beta:.2f}) - Lower systematic risk")
        elif beta < 0.5:
            pts += 4
        elif beta <= 1.5:
            pts += 2
        else:
            warnings.append(f"High Beta ({beta:.2f}) - Elevated volatility relative to the market")

    if hist is not None and not hist.empty and len(hist) > 30:
        c = hist["Close"].dropna()
        roll_max = c.cummax()
        drawdown = (c - roll_max) / roll_max
        max_dd = abs(float(drawdown.min())) * 100
        if max_dd <= 20:
            pts += 5
            positives.append(f"Low Maximum Drawdown over 1Y ({max_dd:.1f}%)")
        elif max_dd <= 35:
            pts += 3
        else:
            warnings.append(f"High 1-Year Maximum Drawdown ({max_dd:.1f}%)")

    if current_ratio is not None:
        if current_ratio >= 1.5:
            pts += 5
            positives.append(f"Strong Current Ratio ({current_ratio:.2f})")
        elif current_ratio >= 1.0:
            pts += 3
        else:
            warnings.append(f"Low Current Ratio ({current_ratio:.2f}) - Working capital pressure")

    return {
        "score": min(pts, 15),
        "max": 15,
        "confidence": 1.0,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "beta": beta,
            "current_ratio": current_ratio
        }
    }


# =========================================================
# FAIR VALUE & DECISION ENGINE
# =========================================================

def calculate_fair_value(info, ltp):
    eps = num(info.get("trailingEps")) or num(info.get("forwardEps"))
    pe = num(info.get("trailingPE"))
    ind_pe = num(info.get("trailingPegRatio")) or num(info.get("forwardPE")) or 20.0

    if eps and eps > 0:
        target_pe = min(pe if pe else ind_pe, ind_pe * 1.1) if pe else ind_pe
        fair_val = round(eps * target_pe, 2)
        margin_of_safety = round(((fair_val - ltp) / fair_val) * 100, 2) if fair_val > 0 else 0
        buy_zone_upper = round(fair_val * 0.90, 2)
        buy_zone_lower = round(fair_val * 0.75, 2)

        return {
            "fair_value": fair_val,
            "margin_of_safety_pct": margin_of_safety,
            "buy_zone": f"₹{buy_zone_lower} - ₹{buy_zone_upper}",
            "valuation_status": "Undervalued" if margin_of_safety > 10 else ("Fairly Valued" if margin_of_safety >= -10 else "Overvalued")
        }

    return {
        "fair_value": None,
        "margin_of_safety_pct": None,
        "buy_zone": "N/A",
        "valuation_status": "Neutral"
    }


def make_decision(total_score, f_score, t_score, warnings):
    critical_warnings = len([w for w in warnings if "high" in w.lower() or "contraction" in w.lower()])

    if total_score >= 76 and f_score >= 26 and t_score >= 15 and critical_warnings == 0:
        return "STRONG BUY"
    elif total_score >= 60 and f_score >= 20:
        return "BUY"
    elif total_score >= 45:
        return "HOLD"
    else:
        return "AVOID"


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def home():
    frontend = (Path(__file__).parent / "../frontend/index.html").resolve()
    return FileResponse(str(frontend))


@app.post("/api/login")
def api_login(x: LoginRequest):
    return login_neo(x.totp)


@app.post("/api/score")
def api_score(x: StockRequest):
    if not logged_in or neo is None:
        raise HTTPException(status_code=401, detail="Login to Kotak Neo first.")

    symbol = x.company.strip().upper()
    stock = find_stock(symbol)

    if not stock:
        raise HTTPException(
            status_code=404,
            detail=f"NSE equity not found for '{symbol}'."
        )

    def extract_ltp(obj):
        preferred_keys = (
            "ltp", "LTP", "last_price", "lastPrice",
            "lastTradedPrice", "last_traded_price",
            "pLtp", "pLTP", "lp"
        )

        def walk(value):
            if isinstance(value, dict):
                for key in preferred_keys:
                    if key in value:
                        n = num(value.get(key))
                        if n is not None and n > 0:
                            return n

                for key in ("data", "result", "quotes", "quote", "response"):
                    if key in value:
                        found = walk(value[key])
                        if found is not None:
                            return found

                for child in value.values():
                    found = walk(child)
                    if found is not None:
                        return found

            elif isinstance(value, list):
                for child in value:
                    found = walk(child)
                    if found is not None:
                        return found

            return None

        return walk(obj)

    token = clean_token(stock.get("pSymbol"))
    exchange = str(stock.get("pExchSeg") or "nse_cm").strip().lower()

    if not token:
        raise HTTPException(status_code=502, detail="Kotak Neo instrument token is missing.")

    ltp = None
    try:
        q = neo.quotes(
            instrument_tokens=[{
                "instrument_token": token,
                "exchange_segment": exchange
            }],
            quote_type="all"
        )
        ltp = extract_ltp(q)
    except Exception as e:
        print("KOTAK QUOTE ERROR:", repr(e))

    if ltp is None:
        raise HTTPException(status_code=502, detail="Unable to retrieve live LTP from Kotak Neo.")

    try:
        t = yf.Ticker(symbol + ".NS")
        info = t.info or {}
        hist = t.history(period="1y", auto_adjust=False)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Yahoo Finance data fetch error: {str(e)}"
        )

    f_res = analyze_fundamentals(info, t)
    t_res = analyze_technicals(hist)
    v_res = analyze_valuation(info, t)
    r_res = analyze_risk(info, hist, t)

    data_confidence_pct = round(
        ((f_res["confidence"] * 0.40) +
         (t_res["confidence"] * 0.25) +
         (v_res["confidence"] * 0.20) +
         (r_res["confidence"] * 0.15)) * 100,
        1
    )

    total_score = f_res["score"] + t_res["score"] + v_res["score"] + r_res["score"]
    total_score = max(0, min(100, total_score))

    positives = f_res["positives"] + t_res["positives"] + v_res["positives"] + r_res["positives"]
    warnings = f_res["warnings"] + t_res["warnings"] + v_res["warnings"] + r_res["warnings"]

    decision = make_decision(total_score, f_res["score"], t_res["score"], warnings)
    valuation_metrics = calculate_fair_value(info, ltp)

    return {
        "company": stock.get("pSymbolName"),
        "symbol": stock.get("pTrdSymbol"),
        "token": token,
        "ltp": ltp,
        "score": total_score,
        "score_model": "100-Point Intelligent Model: Fundamental (40) + Technical (25) + Valuation (20) + Risk (15)",
        "decision": decision,
        "data_confidence_pct": data_confidence_pct,
        "breakdown": {
            "fundamental": {"score": f_res["score"], "out_of": 40, **f_res["data"]},
            "technical": {"score": t_res["score"], "out_of": 25, **t_res["data"]},
            "valuation": {"score": v_res["score"], "out_of": 20, **v_res["data"]},
            "risk": {"score": r_res["score"], "out_of": 15, **r_res["data"]}
        },
        "why_buy": positives,
        "risk_warnings": warnings,
        "fair_value_analysis": valuation_metrics,
        "data_note": "Live LTP: Kotak Neo. Financials & Technical History: Yahoo Finance.",
        "disclaimer": "Decision-support algorithm; not personal financial or investment advice."
    }
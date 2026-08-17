import os
import math
from pathlib import Path
import json
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from neo_api_client import NeoAPI

# =========================================================
# CONFIG & SESSION SETUP
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

# =========================================================
# TRADINGVIEW DATA ENGINE (REPLACES YAHOO FINANCE)
# =========================================================

def fetch_tradingview_data(symbol):
    """
    Fetches both technical and fundamental data directly from TradingView Scanner API.
    Bypasses Yahoo Finance completely and does not require an API key.
    """
    url = "https://scanner.tradingview.com/india/scan"
    
    # TradingView symbol format fix (e.g., M&M becomes M_M)
    tv_symbol = symbol.upper().replace("&", "_").replace("-", "_")
    ticker = f"NSE:{tv_symbol}"
    
    payload = {
        "symbols": {"tickers": [ticker]},
        "columns": [
            "close", "volume", 
            "price_earnings_ttm", "price_book_fq", "price_book_ratio", 
            "return_on_equity", "debt_to_equity_fq", "debt_to_equity",
            "total_revenue_yoy_growth", "revenue_growth_yoy", 
            "earnings_per_share_basic_ttm", "earnings_growth",
            "RSI", "SMA20", "SMA50", "SMA200", 
            "beta_1_year", "current_ratio_fq", "current_ratio",
            "market_cap_basic", "Perf.1M", "Perf.3M", "Perf.Y",
            "price_52_week_high", "price_52_week_low", "High.52", "Low.52",
            "free_cash_flow_margin_ttm"
        ]
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    if not data.get("data"):
        raise Exception(f"Stock '{symbol}' not found on TradingView India scanner.")
        
    row = data["data"][0]["d"]
    cols = payload["columns"]
    return dict(zip(cols, row))


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
            raise HTTPException(status_code=401, detail=str(r))

        r = neo.totp_validate(mpin=os.getenv("KOTAK_MPIN"))

        if isinstance(r, dict) and "error" in r:
            raise HTTPException(status_code=401, detail=str(r))

        logged_in = True
        return {"ok": True, "message": "Kotak Neo login successful"}

    except HTTPException:
        raise
    except Exception as e:
        logged_in = False
        raise HTTPException(status_code=401, detail=f"Kotak Neo login failed: {str(e)}")


# =========================================================
# FIND NSE STOCK (KOTAK NEO)
# =========================================================

def find_stock(symbol):
    symbol = symbol.upper().strip()
    queries_to_try = [f"{symbol}-EQ", symbol]
    segments_to_try = ["nse_cm", "NSE", "bse_cm", "BSE"]
    
    valid_items = []

    for seg in segments_to_try:
        for q in queries_to_try:
            try:
                res = neo.search_scrip(exchange_segment=seg, symbol=q, expiry="", option_type="", strike_price="")
                if not res: continue

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
            except Exception:
                pass

    if not valid_items:
        return None

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

    def get_exch(item):
        return str(item.get("pExchSeg") or item.get("exchange_segment") or item.get("exch") or "nse_cm").lower()

    target_eq = f"{symbol}-EQ"

    for item in unique_items:
        token = get_token(item)
        trd_sym = get_trd_sym(item)
        exch = get_exch(item)

        if token and trd_sym == target_eq:
            return {
                "pSymbolName": symbol,
                "pTrdSymbol": trd_sym,
                "pSymbol": token,
                "pExchSeg": exch if "nse" in exch else "nse_cm"
            }
    
    # Fallback to direct symbol matching
    for item in unique_items:
        token = get_token(item)
        trd_sym = get_trd_sym(item)
        exch = get_exch(item)

        is_non_equity = any(trd_sym.endswith(ext) or ext in trd_sym for ext in ("-BL", "-FUT", "-CE", "-PE", "-N1", "-N2", "-E1", "-BE", "-BZ"))
        if token and not is_non_equity and (trd_sym == symbol):
            return {
                "pSymbolName": symbol,
                "pTrdSymbol": trd_sym,
                "pSymbol": token,
                "pExchSeg": exch if "nse" in exch else "nse_cm"
            }

    return None

# =========================================================
# INTELLIGENT 100-POINT SCORING ENGINE (TradingView Edition)
# =========================================================

def analyze_fundamentals(tv_data):
    pts = 0
    max_possible = 0
    positives = []
    warnings = []

    pe = num(tv_data.get("price_earnings_ttm"))
    ind_pe = 25.0 
    roe_pct = num(tv_data.get("return_on_equity")) 
    rg_pct = num(tv_data.get("total_revenue_yoy_growth")) or num(tv_data.get("revenue_growth_yoy"))
    de = num(tv_data.get("debt_to_equity_fq")) or num(tv_data.get("debt_to_equity"))

    if pe is not None:
        max_possible += 8
        if pe <= 18:
            pts += 8
            positives.append(f"Attractive PE ratio of {pe:.1f}")
        elif pe <= 28:
            pts += 5
            positives.append(f"Reasonable PE ratio of {pe:.1f}")
        elif pe <= 35:
            pts += 2
        else:
            warnings.append(f"High PE valuation ({pe:.1f})")

    if rg_pct is not None:
        max_possible += 8
        if rg_pct >= 15:
            pts += 8 
            positives.append(f"Strong Revenue Growth (+{rg_pct:.1f}%)")
        elif rg_pct >= 8:
            pts += 4
        elif rg_pct < 0:
            warnings.append(f"Declining Revenue (-{abs(rg_pct):.1f}%)")

    if roe_pct is not None:
        max_possible += 8
        if roe_pct >= 20:
            pts += 8
            positives.append(f"High Return on Equity (ROE {roe_pct:.1f}%)")
        elif roe_pct >= 14:
            pts += 5
            positives.append(f"Healthy ROE of {roe_pct:.1f}%")
        elif roe_pct >= 8:
            pts += 2
        else:
            warnings.append(f"Low ROE ({roe_pct:.1f}%) indicates lower capital efficiency")

    if de is not None:
        max_possible += 8
        if de <= 0.3: 
            pts += 8
            positives.append(f"Low Debt/Equity ratio ({de:.2f})")
        elif de <= 0.7:
            pts += 5
        elif de <= 1.2:
            pts += 1
        else:
            warnings.append(f"High Debt to Equity ratio ({de:.2f})")

    # STRICT SCALING: Calculate percentage of actual points vs possible points
    if max_possible > 0:
        pts = int((pts / max_possible) * 40)
        
    conf = (max_possible / 32) if max_possible > 0 else 0

    return {
        "score": min(pts, 40),
        "max": 40,
        "confidence": conf,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "pe": pe,
            "industry_pe": ind_pe,
            "roe_pct": round(roe_pct, 2) if roe_pct is not None else None,
            "revenue_growth_pct": round(rg_pct, 2) if rg_pct is not None else None,
            "debt_to_equity": de
        }
    }

def analyze_technicals(tv_data):
    positives = []
    warnings = []
    pts = 0

    last_price = num(tv_data.get("close"))
    if not last_price:
        return {"score": 0, "max": 25, "confidence": 0.0, "positives": [], "warnings": [], "data": {}}

    sma20 = num(tv_data.get("SMA20"))
    sma50 = num(tv_data.get("SMA50"))
    sma200 = num(tv_data.get("SMA200"))
    rv = num(tv_data.get("RSI"))
    m20 = num(tv_data.get("Perf.1M"))
    m60 = num(tv_data.get("Perf.3M"))

    high_52 = num(tv_data.get("price_52_week_high")) or num(tv_data.get("High.52"))
    low_52 = num(tv_data.get("price_52_week_low")) or num(tv_data.get("Low.52"))

    if sma50 and sma200:
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
            pts += 2
        elif rv > 75:
            warnings.append(f"RSI Overbought ({rv:.1f}) - Short-term pullback risk")
        elif rv < 35:
            warnings.append(f"RSI Oversold ({rv:.1f}) - Strong weakness")

    if m20 is not None:
        if m20 >= 5:
            pts += 4
            positives.append(f"Positive 1-Month momentum (+{m20:.1f}%)")
        elif m20 < -5:
            warnings.append(f"Negative short-term momentum ({m20:.1f}% 1M return)")

    if m60 is not None and m60 >= 10:
        pts += 3

    dist_high, dist_low = None, None
    if high_52 and low_52 and last_price:
        dist_high = ((last_price - high_52) / high_52) * 100
        dist_low = ((last_price - low_52) / low_52) * 100
        
        if dist_high >= -5:
            warnings.append("Trading near 52-Week High resistance zone")
        elif dist_low <= 10:
            positives.append("Trading close to 52-Week Low support base")
            pts += 3

    return {
        "score": min(pts, 25),
        "max": 25,
        "confidence": 1.0,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "rsi": round(rv, 2) if rv else None,
            "sma20": round(sma20, 2) if sma20 else None,
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "momentum_1m_pct": round(m20, 2) if m20 else None,
            "momentum_3m_pct": round(m60, 2) if m60 else None,
            "dist_52w_high_pct": round(dist_high, 2) if dist_high else None,
            "dist_52w_low_pct": round(dist_low, 2) if dist_low else None
        }
    }

def analyze_valuation(tv_data):
    positives = []
    warnings = []
    pts = 0
    max_possible = 0

    pe = num(tv_data.get("price_earnings_ttm"))
    eg_pct = num(tv_data.get("earnings_growth")) or num(tv_data.get("total_revenue_yoy_growth"))
    peg = round(pe / eg_pct, 2) if pe and eg_pct and eg_pct > 0 else None
    pb = num(tv_data.get("price_book_fq")) or num(tv_data.get("price_book_ratio"))
    fcf_margin = num(tv_data.get("free_cash_flow_margin_ttm"))

    if peg is not None and peg > 0:
        max_possible += 8
        if peg <= 1.0:
            pts += 8
            positives.append(f"Attractive PEG ratio ({peg:.2f})")
        elif peg <= 1.5:
            pts += 4
        elif peg <= 2.0:
            pts += 1
        else:
            warnings.append(f"High PEG ratio ({peg:.2f})")

    if fcf_margin is not None:
        max_possible += 6
        if fcf_margin >= 15:
            pts += 6
            positives.append(f"Strong Free Cash Flow Margin ({fcf_margin:.1f}%)")
        elif fcf_margin >= 5:
            pts += 3
        elif fcf_margin < 0:
            warnings.append("Negative Free Cash Flow Margin")

    if pb is not None and pb > 0:
        max_possible += 6
        if pb <= 2.5:
            pts += 6
            positives.append(f"Reasonable Price to Book value ({pb:.2f})")
        elif pb <= 5.0:
            pts += 2
        else:
            warnings.append(f"Elevated Price-to-Book multiple ({pb:.2f})")

    # STRICT SCALING
    if max_possible > 0:
        pts = int((pts / max_possible) * 20) 
        
    conf = (max_possible / 20) if max_possible > 0 else 0

    return {
        "score": min(pts, 20),
        "max": 20,
        "confidence": conf,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "peg_ratio": peg,
            "price_to_book": pb,
            "fcf_margin_pct": fcf_margin
        }
    }

def analyze_risk(tv_data):
    positives = []
    warnings = []
    pts = 0

    beta = num(tv_data.get("beta_1_year"))
    current_ratio = num(tv_data.get("current_ratio_fq")) or num(tv_data.get("current_ratio"))

    if beta is not None:
        if 0.5 <= beta <= 1.1:
            pts += 7
            positives.append(f"Moderate Market Beta ({beta:.2f}) - Lower systematic risk")
        elif beta < 0.5:
            pts += 5
        elif beta <= 1.5:
            pts += 3
        else:
            warnings.append(f"High Beta ({beta:.2f}) - Elevated volatility relative to the market")

    if current_ratio is not None:
        if current_ratio >= 1.5:
            pts += 8
            positives.append(f"Strong Current Ratio ({current_ratio:.2f})")
        elif current_ratio >= 1.0:
            pts += 4
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

def calculate_fair_value(tv_data, ltp):
    eps = num(tv_data.get("earnings_per_share_basic_ttm"))
    pe = num(tv_data.get("price_earnings_ttm"))
    ind_pe = 25.0

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
    # Added "negative" to critical warnings to catch bad cash flows/growth
    critical_warnings = len([w for w in warnings if "high" in w.lower() or "contraction" in w.lower() or "negative" in w.lower()])

    # Raised cut-offs significantly
    if total_score >= 80 and f_score >= 28 and t_score >= 18 and critical_warnings == 0:
        return "STRONG BUY"
    elif total_score >= 65 and f_score >= 24:
        return "BUY"
    elif total_score >= 48:
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
        raise HTTPException(status_code=404, detail=f"NSE equity not found for '{symbol}'.")

    def extract_ltp(obj):
        preferred_keys = ("ltp", "LTP", "last_price", "lastPrice", "lastTradedPrice", "last_traded_price", "pLtp", "pLTP", "lp")
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
            instrument_tokens=[{"instrument_token": token, "exchange_segment": exchange}],
            quote_type="all"
        )
        ltp = extract_ltp(q)
    except Exception as e:
        print("KOTAK QUOTE ERROR:", repr(e))

    if ltp is None:
        raise HTTPException(status_code=502, detail="Unable to retrieve live LTP from Kotak Neo.")

    # --- Fetch Everything via TradingView ---
    try:
        tv_data = fetch_tradingview_data(symbol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TradingView data fetch error: {str(e)}")

    f_res = analyze_fundamentals(tv_data)
    t_res = analyze_technicals(tv_data)
    v_res = analyze_valuation(tv_data)
    r_res = analyze_risk(tv_data)

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
    valuation_metrics = calculate_fair_value(tv_data, ltp)

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
        "data_note": "Live LTP: Kotak Neo. Financials & Technical History: TradingView Scanner.",
        "disclaimer": "Decision-support algorithm; not personal financial or investment advice."
    }

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
    """
    Fundamental score: fixed 40-point model.
    Missing data gets no points; available metrics do NOT get rescaled to 40.
    """
    positives = []
    warnings = []
    pts = 0
    data_points_found = 0
    total_metrics = 4

    pe = num(tv_data.get("price_earnings_ttm"))
    roe_pct = num(tv_data.get("return_on_equity"))
    rg_pct = num(tv_data.get("total_revenue_yoy_growth")) or num(tv_data.get("revenue_growth_yoy"))
    de = num(tv_data.get("debt_to_equity_fq")) or num(tv_data.get("debt_to_equity"))

    # PE: 10 points
    if pe is not None and pe > 0:
        data_points_found += 1
        if pe <= 15:
            pts += 10
            positives.append(f"Very attractive PE ratio ({pe:.1f})")
        elif pe <= 20:
            pts += 8
            positives.append(f"Attractive PE ratio ({pe:.1f})")
        elif pe <= 30:
            pts += 6
            positives.append(f"Reasonable PE ratio ({pe:.1f})")
        elif pe <= 40:
            pts += 3
            warnings.append(f"High PE valuation ({pe:.1f})")
        else:
            warnings.append(f"Very high PE valuation ({pe:.1f})")

    # Revenue Growth: 10 points
    if rg_pct is not None:
        data_points_found += 1
        if rg_pct >= 20:
            pts += 10
            positives.append(f"Excellent Revenue Growth (+{rg_pct:.1f}%)")
        elif rg_pct >= 12:
            pts += 8
            positives.append(f"Strong Revenue Growth (+{rg_pct:.1f}%)")
        elif rg_pct >= 5:
            pts += 5
        elif rg_pct >= 0:
            pts += 2
            warnings.append(f"Low Revenue Growth (+{rg_pct:.1f}%)")
        else:
            warnings.append(f"Declining Revenue ({rg_pct:.1f}%)")

    # ROE: 10 points
    if roe_pct is not None:
        data_points_found += 1
        if roe_pct >= 25:
            pts += 10
            positives.append(f"Excellent ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 18:
            pts += 8
            positives.append(f"Strong ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 12:
            pts += 5
        elif roe_pct >= 8:
            pts += 2
            warnings.append(f"Low ROE ({roe_pct:.1f}%)")
        else:
            warnings.append(f"Very low ROE ({roe_pct:.1f}%)")

    # Debt/Equity: 10 points
    if de is not None and de >= 0:
        data_points_found += 1
        if de <= 0.30:
            pts += 10
            positives.append(f"Very low Debt/Equity ({de:.2f})")
        elif de <= 0.60:
            pts += 8
            positives.append(f"Low Debt/Equity ({de:.2f})")
        elif de <= 1.00:
            pts += 5
        elif de <= 1.50:
            pts += 2
            warnings.append(f"Elevated Debt/Equity ({de:.2f})")
        else:
            warnings.append(f"High Debt/Equity ({de:.2f})")

    return {
        "score": min(pts, 40),
        "max": 40,
        "confidence": data_points_found / total_metrics,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "pe": pe,
            "industry_pe": None,
            "roe_pct": round(roe_pct, 2) if roe_pct is not None else None,
            "revenue_growth_pct": round(rg_pct, 2) if rg_pct is not None else None,
            "debt_to_equity": de
        }
    }

def analyze_technicals(tv_data):
    """Technical score: fixed 25-point model."""
    positives = []
    warnings = []
    pts = 0
    found = 0
    total_metrics = 4

    last_price = num(tv_data.get("close"))
    sma20 = num(tv_data.get("SMA20"))
    sma50 = num(tv_data.get("SMA50"))
    sma200 = num(tv_data.get("SMA200"))
    rv = num(tv_data.get("RSI"))
    m20 = num(tv_data.get("Perf.1M"))
    m60 = num(tv_data.get("Perf.3M"))

    high_52 = num(tv_data.get("price_52_week_high")) or num(tv_data.get("High.52"))
    low_52 = num(tv_data.get("price_52_week_low")) or num(tv_data.get("Low.52"))

    if not last_price:
        return {
            "score": 0, "max": 25, "confidence": 0.0,
            "positives": [], "warnings": ["Live/TradingView price unavailable."], "data": {}
        }

    # Trend: 8 points
    if sma50 is not None and sma200 is not None:
        found += 1
        if last_price > sma50 > sma200:
            pts += 8
            positives.append("Strong Bullish Trend (Price > SMA50 > SMA200)")
        elif last_price > sma50 and last_price > sma200:
            pts += 6
            positives.append("Bullish Trend (Price above SMA50 and SMA200)")
        elif last_price > sma50:
            pts += 4
        elif last_price > sma200:
            pts += 2
        else:
            warnings.append("Price is below both SMA50 and SMA200.")

    # RSI: 5 points
    if rv is not None:
        found += 1
        if 55 <= rv <= 68:
            pts += 5
            positives.append(f"Healthy RSI ({rv:.1f})")
        elif 50 <= rv < 55:
            pts += 3
        elif 68 < rv <= 75:
            pts += 2
            warnings.append(f"RSI elevated ({rv:.1f})")
        elif rv > 75:
            pts += 0
            warnings.append(f"RSI Overbought ({rv:.1f})")
        elif 40 <= rv < 50:
            pts += 1
        else:
            warnings.append(f"Weak RSI ({rv:.1f})")

    # Momentum: 7 points
    if m20 is not None and m60 is not None:
        found += 1
        if m20 >= 5 and m60 >= 10:
            pts += 7
            positives.append(f"Strong momentum (1M +{m20:.1f}%, 3M +{m60:.1f}%)")
        elif m20 >= 3 and m60 >= 5:
            pts += 5
        elif m20 >= 0 and m60 >= 0:
            pts += 3
        elif m20 < -8 or m60 < -12:
            warnings.append(f"Negative momentum (1M {m20:.1f}%, 3M {m60:.1f}%)")
        else:
            pts += 1

    # 52-week position: 5 points
    if high_52 is not None and low_52 is not None and high_52 > low_52:
        found += 1
        dist_high = ((last_price - high_52) / high_52) * 100
        dist_low = ((last_price - low_52) / low_52) * 100

        if dist_high >= -5:
            pts += 1
            warnings.append("Trading near 52-Week High resistance zone.")
        elif dist_low <= 10:
            pts += 2
            positives.append("Trading near 52-Week Low support zone.")
        elif dist_high >= -20:
            pts += 4
        else:
            pts += 5

    return {
        "score": min(pts, 25),
        "max": 25,
        "confidence": found / total_metrics,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "rsi": round(rv, 2) if rv is not None else None,
            "sma20": round(sma20, 2) if sma20 is not None else None,
            "sma50": round(sma50, 2) if sma50 is not None else None,
            "sma200": round(sma200, 2) if sma200 is not None else None,
            "momentum_1m_pct": round(m20, 2) if m20 is not None else None,
            "momentum_3m_pct": round(m60, 2) if m60 is not None else None,
            "dist_52w_high_pct": round(((last_price - high_52) / high_52) * 100, 2)
                if high_52 else None,
            "dist_52w_low_pct": round(((last_price - low_52) / low_52) * 100, 2)
                if low_52 else None
        }
    }

def analyze_valuation(tv_data):
    """
    Valuation score: fixed 20-point model.
    Missing data gets no points; available metrics are never rescaled.
    """
    positives = []
    warnings = []
    pts = 0
    found = 0
    total_metrics = 3

    pe = num(tv_data.get("price_earnings_ttm"))
    eg_pct = num(tv_data.get("earnings_growth")) or num(tv_data.get("total_revenue_yoy_growth"))

    peg = None
    if pe is not None and eg_pct is not None and eg_pct > 0:
        peg = round(pe / eg_pct, 2)

    pb = num(tv_data.get("price_book_fq")) or num(tv_data.get("price_book_ratio"))
    fcf_margin = num(tv_data.get("free_cash_flow_margin_ttm"))

    # PEG: 7 points
    if peg is not None and peg > 0:
        found += 1
        if peg <= 0.8:
            pts += 7
            positives.append(f"Very attractive PEG ({peg:.2f})")
        elif peg <= 1.2:
            pts += 6
            positives.append(f"Attractive PEG ({peg:.2f})")
        elif peg <= 1.8:
            pts += 4
        elif peg <= 2.5:
            pts += 2
            warnings.append(f"Elevated PEG ({peg:.2f})")
        else:
            warnings.append(f"High PEG ({peg:.2f}) indicates overvaluation")

    # P/B: 6 points
    if pb is not None and pb > 0:
        found += 1
        if pb <= 1.5:
            pts += 6
            positives.append(f"Attractive Price-to-Book ({pb:.2f})")
        elif pb <= 2.5:
            pts += 5
        elif pb <= 4.0:
            pts += 3
        elif pb <= 6.0:
            pts += 1
            warnings.append(f"Elevated Price-to-Book ({pb:.2f})")
        else:
            warnings.append(f"High Price-to-Book ({pb:.2f})")

    # FCF Margin: 7 points
    if fcf_margin is not None:
        found += 1
        if fcf_margin >= 15:
            pts += 7
            positives.append(f"Strong Free Cash Flow Margin ({fcf_margin:.1f}%)")
        elif fcf_margin >= 8:
            pts += 5
        elif fcf_margin >= 3:
            pts += 3
        elif fcf_margin >= 0:
            pts += 1
        else:
            warnings.append(f"Negative Free Cash Flow Margin ({fcf_margin:.1f}%)")

    return {
        "score": min(pts, 20),
        "max": 20,
        "confidence": found / total_metrics,
        "positives": positives,
        "warnings": warnings,
        "data": {
            "peg_ratio": peg,
            "price_to_book": pb,
            "fcf_margin_pct": fcf_margin
        }
    }

def analyze_risk(tv_data):
    """Risk score: fixed 15-point model."""
    positives = []
    warnings = []
    pts = 0
    found = 0
    total_metrics = 2

    beta = num(tv_data.get("beta_1_year"))
    current_ratio = num(tv_data.get("current_ratio_fq")) or num(tv_data.get("current_ratio"))

    # Beta: 7 points
    if beta is not None:
        found += 1
        if 0.5 <= beta <= 1.0:
            pts += 7
            positives.append(f"Moderate Market Beta ({beta:.2f})")
        elif beta < 0.5:
            pts += 5
            positives.append(f"Low Beta ({beta:.2f})")
        elif beta <= 1.3:
            pts += 4
        elif beta <= 1.7:
            pts += 2
            warnings.append(f"Elevated Beta ({beta:.2f})")
        else:
            warnings.append(f"High Beta ({beta:.2f}) - Elevated volatility")

    # Current Ratio: 8 points
    if current_ratio is not None and current_ratio > 0:
        found += 1
        if current_ratio >= 2.0:
            pts += 8
            positives.append(f"Strong Current Ratio ({current_ratio:.2f})")
        elif current_ratio >= 1.5:
            pts += 6
        elif current_ratio >= 1.0:
            pts += 3
        else:
            warnings.append(f"Low Current Ratio ({current_ratio:.2f}) - Working capital pressure")

    return {
        "score": min(pts, 15),
        "max": 15,
        "confidence": found / total_metrics,
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

def make_decision(total_score, f_score, t_score, v_score, r_score, warnings):
    """
    Conservative decision engine.
    BUY requires both fundamental and technical strength.
    STRONG BUY additionally requires valuation and risk support.
    """
    critical_warnings = len([
        w for w in warnings
        if any(word in w.lower() for word in (
            "high", "very high", "declining", "negative", "overbought",
            "weak", "pressure", "overvaluation"
        ))
    ])

    if (
        total_score >= 80
        and f_score >= 28
        and t_score >= 17
        and v_score >= 12
        and r_score >= 8
        and critical_warnings == 0
    ):
        return "STRONG BUY"

    if (
        total_score >= 65
        and f_score >= 22
        and t_score >= 13
        and critical_warnings <= 1
    ):
        return "BUY"

    if total_score >= 50:
        return "HOLD"

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

    # Do not hide missing-data risk behind a high score.
    if data_confidence_pct < 70:
        warnings.append(
            f"Limited data confidence ({data_confidence_pct:.1f}%) - decision should be treated cautiously."
        )

    decision = make_decision(
        total_score,
        f_res["score"],
        t_res["score"],
        v_res["score"],
        r_res["score"],
        warnings
    )
    valuation_metrics = calculate_fair_value(tv_data, ltp)

    return {
        "company": stock.get("pSymbolName"),
        "symbol": stock.get("pTrdSymbol"),
        "token": token,
        "ltp": ltp,
        "score": total_score,
        "score_model": "100-Point Fixed Model: Fundamental (40) + Technical (25) + Valuation (20) + Risk (15)",
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


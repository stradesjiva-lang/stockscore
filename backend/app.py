import os
import math
import secrets
import time
from pathlib import Path
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote_plus
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo_api_client import NeoAPI

# =========================================================
# CONFIG & SESSION SETUP
# =========================================================

load_dotenv()

app = FastAPI(title="StockMeter - 100pt Stock Analysis")

# CORS: explicit origins when the frontend is hosted separately.
# Same-origin Render deployment does not require CORS, but localhost development does.
cors_origins = [
    x.strip() for x in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://localhost:5500"
    ).split(",") if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Each login receives its own in-memory session.
# The actual Kotak Neo client remains server-side and is never sent to the browser.
sessions: Dict[str, Dict[str, Any]] = {}
score_history: List[Dict[str, Any]] = []
alerts_store: Dict[str, Dict[str, Any]] = {}

# =========================================================
# REQUEST MODELS
# =========================================================

class LoginRequest(BaseModel):
    totp: str

class StockRequest(BaseModel):
    company: str

class CompareRequest(BaseModel):
    symbols: List[str]

class ScreenerRequest(BaseModel):
    symbols: List[str] = []
    min_score: Optional[float] = None
    min_roe: Optional[float] = None
    max_pe: Optional[float] = None
    min_revenue_growth: Optional[float] = None
    max_debt_to_equity: Optional[float] = None
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    min_fcf_margin: Optional[float] = None

class Holding(BaseModel):
    symbol: str
    quantity: float
    avg_price: float

class PortfolioRequest(BaseModel):
    holdings: List[Holding]

class AlertRequest(BaseModel):
    symbol: str
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    below_price: Optional[float] = None
    above_price: Optional[float] = None

class HistoryRequest(BaseModel):
    symbol: str
    score: float
    decision: str = ""
    ltp: Optional[float] = None

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
# TRADINGVIEW DATA ENGINE
# =========================================================

def fetch_tradingview_data(symbol):
    url = "https://scanner.tradingview.com/india/scan"
    
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
    if len(totp) != 6 or not totp.isdigit():
        raise HTTPException(status_code=400, detail="TOTP must be exactly 6 digits.")

    consumer_key = os.getenv("KOTAK_CONSUMER_KEY")
    mobile = os.getenv("KOTAK_MOBILE")
    ucc = os.getenv("KOTAK_UCC")
    mpin = os.getenv("KOTAK_MPIN")

    missing = [
        name for name, value in {
            "KOTAK_CONSUMER_KEY": consumer_key,
            "KOTAK_MOBILE": mobile,
            "KOTAK_UCC": ucc,
            "KOTAK_MPIN": mpin,
        }.items() if not value
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail="Missing Render environment variables: " + ", ".join(missing)
        )

    neo_client = NeoAPI(
        environment="prod",
        access_token=None,
        neo_fin_key=None,
        consumer_key=consumer_key
    )

    try:
        r = neo_client.totp_login(
            mobile_number=mobile,
            ucc=ucc,
            totp=totp
        )
        if isinstance(r, dict) and "error" in r:
            raise HTTPException(status_code=401, detail=str(r))

        r = neo_client.totp_validate(mpin=mpin)
        if isinstance(r, dict) and "error" in r:
            raise HTTPException(status_code=401, detail=str(r))

        session_id = secrets.token_urlsafe(32)
        sessions[session_id] = {
            "neo": neo_client,
            "created_at": time.time(),
            "last_used": time.time()
        }
        return {
            "ok": True,
            "message": "Kotak Neo login successful",
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Kotak Neo login failed: {str(e)}"
        )

def get_session(request: Request) -> Dict[str, Any]:
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Login to Kotak Neo first.")

    session = sessions[session_id]
    # Keep the in-memory session alive for 12 hours.
    if time.time() - session["created_at"] > 12 * 3600:
        sessions.pop(session_id, None)
        raise HTTPException(status_code=401, detail="Session expired. Login again.")

    session["last_used"] = time.time()
    return session

@app.post("/api/logout")
def api_logout(request: Request):
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    if session_id:
        sessions.pop(session_id, None)
    return {"ok": True}

@app.get("/api/session")
def api_session(request: Request):
    get_session(request)
    return {"logged_in": True}


# =========================================================
# FIND NSE STOCK (KOTAK NEO)
# =========================================================

def find_stock(symbol, neo_client):
    symbol = symbol.upper().strip()
    queries_to_try = [f"{symbol}-EQ", symbol]
    segments_to_try = ["nse_cm", "NSE", "bse_cm", "BSE"]
    
    valid_items = []

    for seg in segments_to_try:
        for q in queries_to_try:
            try:
                res = neo_client.search_scrip(exchange_segment=seg, symbol=q, expiry="", option_type="", strike_price="")
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
# INTELLIGENT 100-POINT SCORING ENGINE
# =========================================================

def analyze_fundamentals(tv_data):
    positives, warnings, pts, data_points_found = [], [], 0, 0
    total_metrics = 4

    pe = num(tv_data.get("price_earnings_ttm"))
    roe_pct = num(tv_data.get("return_on_equity"))
    rg_pct = num(tv_data.get("total_revenue_yoy_growth")) or num(tv_data.get("revenue_growth_yoy"))
    de = num(tv_data.get("debt_to_equity_fq")) or num(tv_data.get("debt_to_equity"))

    if pe is not None and pe > 0:
        data_points_found += 1
        if pe <= 15: pts += 10; positives.append(f"Very attractive PE ratio ({pe:.1f})")
        elif pe <= 20: pts += 8; positives.append(f"Attractive PE ratio ({pe:.1f})")
        elif pe <= 30: pts += 6; positives.append(f"Reasonable PE ratio ({pe:.1f})")
        elif pe <= 40: pts += 3; warnings.append(f"High PE valuation ({pe:.1f})")
        else: warnings.append(f"Very high PE valuation ({pe:.1f})")

    if rg_pct is not None:
        data_points_found += 1
        if rg_pct >= 20: pts += 10; positives.append(f"Excellent Revenue Growth (+{rg_pct:.1f}%)")
        elif rg_pct >= 12: pts += 8; positives.append(f"Strong Revenue Growth (+{rg_pct:.1f}%)")
        elif rg_pct >= 5: pts += 5
        elif rg_pct >= 0: pts += 2; warnings.append(f"Low Revenue Growth (+{rg_pct:.1f}%)")
        else: warnings.append(f"Declining Revenue ({rg_pct:.1f}%)")

    if roe_pct is not None:
        data_points_found += 1
        if roe_pct >= 25: pts += 10; positives.append(f"Excellent ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 18: pts += 8; positives.append(f"Strong ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 12: pts += 5
        elif roe_pct >= 8: pts += 2; warnings.append(f"Low ROE ({roe_pct:.1f}%)")
        else: warnings.append(f"Very low ROE ({roe_pct:.1f}%)")

    if de is not None and de >= 0:
        data_points_found += 1
        if de <= 0.30: pts += 10; positives.append(f"Very low Debt/Equity ({de:.2f})")
        elif de <= 0.60: pts += 8; positives.append(f"Low Debt/Equity ({de:.2f})")
        elif de <= 1.00: pts += 5
        elif de <= 1.50: pts += 2; warnings.append(f"Elevated Debt/Equity ({de:.2f})")
        else: warnings.append(f"High Debt/Equity ({de:.2f})")

    return {
        "score": min(pts, 40), "max": 40, "confidence": data_points_found / total_metrics,
        "positives": positives, "warnings": warnings,
        "data": {"pe": pe, "roe_pct": round(roe_pct, 2) if roe_pct is not None else None, 
                 "revenue_growth_pct": round(rg_pct, 2) if rg_pct is not None else None, "debt_to_equity": de}
    }

def analyze_technicals(tv_data):
    positives, warnings, pts, found = [], [], 0, 0
    total_metrics = 4

    last_price = num(tv_data.get("close"))
    sma50 = num(tv_data.get("SMA50"))
    sma200 = num(tv_data.get("SMA200"))
    rv = num(tv_data.get("RSI"))
    m20 = num(tv_data.get("Perf.1M"))
    m60 = num(tv_data.get("Perf.3M"))
    high_52 = num(tv_data.get("price_52_week_high")) or num(tv_data.get("High.52"))
    low_52 = num(tv_data.get("price_52_week_low")) or num(tv_data.get("Low.52"))

    if not last_price:
        return {"score": 0, "max": 25, "confidence": 0.0, "positives": [], "warnings": ["Live price unavailable."], "data": {}}

    if sma50 is not None and sma200 is not None:
        found += 1
        if last_price > sma50 > sma200: pts += 8; positives.append("Strong Bullish Trend")
        elif last_price > sma50 and last_price > sma200: pts += 6; positives.append("Bullish Trend")
        elif last_price > sma50: pts += 4
        elif last_price > sma200: pts += 2
        else: warnings.append("Price is below both SMA50 and SMA200.")

    if rv is not None:
        found += 1
        if 55 <= rv <= 68: pts += 5; positives.append(f"Healthy RSI ({rv:.1f})")
        elif 50 <= rv < 55: pts += 3
        elif 68 < rv <= 75: pts += 2; warnings.append(f"RSI elevated ({rv:.1f})")
        elif rv > 75: pts += 0; warnings.append(f"RSI Overbought ({rv:.1f})")
        elif 40 <= rv < 50: pts += 1
        else: warnings.append(f"Weak RSI ({rv:.1f})")

    if m20 is not None and m60 is not None:
        found += 1
        if m20 >= 5 and m60 >= 10: pts += 7; positives.append(f"Strong momentum")
        elif m20 >= 3 and m60 >= 5: pts += 5
        elif m20 >= 0 and m60 >= 0: pts += 3
        elif m20 < -8 or m60 < -12: warnings.append(f"Negative momentum")
        else: pts += 1

    if high_52 is not None and low_52 is not None and high_52 > low_52:
        found += 1
        dist_high = ((last_price - high_52) / high_52) * 100
        dist_low = ((last_price - low_52) / low_52) * 100
        if dist_high >= -5: pts += 1; warnings.append("Trading near 52-Week High.")
        elif dist_low <= 10: pts += 2; positives.append("Trading near 52-Week Low support.")
        elif dist_high >= -20: pts += 4
        else: pts += 5

    return {
        "score": min(pts, 25), "max": 25, "confidence": found / total_metrics,
        "positives": positives, "warnings": warnings,
        "data": {"rsi": round(rv, 2) if rv else None}
    }

def analyze_valuation(tv_data):
    positives, warnings, pts, found = [], [], 0, 0
    pe = num(tv_data.get("price_earnings_ttm"))
    eg_pct = num(tv_data.get("earnings_growth")) or num(tv_data.get("total_revenue_yoy_growth"))
    pb = num(tv_data.get("price_book_fq")) or num(tv_data.get("price_book_ratio"))
    fcf = num(tv_data.get("free_cash_flow_margin_ttm"))
    peg = round(pe / eg_pct, 2) if pe and eg_pct and eg_pct > 0 else None

    if peg is not None:
        found += 1
        if peg <= 0.8: pts += 7; positives.append(f"Very attractive PEG")
        elif peg <= 1.2: pts += 6; positives.append(f"Attractive PEG")
        elif peg <= 1.8: pts += 4
        elif peg <= 2.5: pts += 2; warnings.append(f"Elevated PEG")
        else: warnings.append(f"High PEG indicates overvaluation")

    if pb is not None:
        found += 1
        if pb <= 1.5: pts += 6; positives.append(f"Attractive Price-to-Book")
        elif pb <= 2.5: pts += 5
        elif pb <= 4.0: pts += 3
        elif pb <= 6.0: pts += 1; warnings.append(f"Elevated Price-to-Book")
        else: warnings.append(f"High Price-to-Book")

    if fcf is not None:
        found += 1
        if fcf >= 15: pts += 7; positives.append(f"Strong FCF Margin")
        elif fcf >= 8: pts += 5
        elif fcf >= 3: pts += 3
        elif fcf >= 0: pts += 1
        else: warnings.append(f"Negative FCF Margin")

    return {"score": min(pts, 20), "max": 20, "confidence": found / 3, "positives": positives, "warnings": warnings, "data": {"peg": peg, "price_to_book": pb, "fcf_margin_pct": fcf}}

def analyze_risk(tv_data):
    positives, warnings, pts, found = [], [], 0, 0
    beta = num(tv_data.get("beta_1_year"))
    cr = num(tv_data.get("current_ratio_fq")) or num(tv_data.get("current_ratio"))

    if beta is not None:
        found += 1
        if 0.5 <= beta <= 1.0: pts += 7; positives.append(f"Moderate Market Beta")
        elif beta < 0.5: pts += 5; positives.append(f"Low Beta")
        elif beta <= 1.3: pts += 4
        elif beta <= 1.7: pts += 2; warnings.append(f"Elevated Beta")
        else: warnings.append(f"High Beta")

    if cr is not None:
        found += 1
        if cr >= 2.0: pts += 8; positives.append(f"Strong Current Ratio")
        elif cr >= 1.5: pts += 6
        elif cr >= 1.0: pts += 3
        else: warnings.append(f"Low Current Ratio")

    return {"score": min(pts, 15), "max": 15, "confidence": found / 2, "positives": positives, "warnings": warnings, "data": {}}

# =========================================================
# CORE FUNCTIONS
# =========================================================

def calculate_fair_value(tv_data, ltp):
    eps = num(tv_data.get("earnings_per_share_basic_ttm"))
    pe = num(tv_data.get("price_earnings_ttm"))
    ind_pe = 25.0

    if eps and eps > 0:
        target_pe = min(pe if pe else ind_pe, ind_pe * 1.1) if pe else ind_pe
        fair_val = round(eps * target_pe, 2)
        mos = round(((fair_val - ltp) / fair_val) * 100, 2) if fair_val > 0 else 0
        return {
            "fair_value": fair_val, "margin_of_safety_pct": mos,
            "buy_zone": f"₹{round(fair_val * 0.75, 2)} - ₹{round(fair_val * 0.90, 2)}",
            "valuation_status": "Undervalued" if mos > 10 else ("Fairly Valued" if mos >= -10 else "Overvalued")
        }
    return {"fair_value": None, "margin_of_safety_pct": None, "buy_zone": "N/A", "valuation_status": "Neutral"}

def make_decision(total_score, f_score, t_score, v_score, r_score, warnings):
    cw = len([w for w in warnings if any(word in w.lower() for word in ("high", "negative", "weak", "overvaluation"))])
    if total_score >= 80 and f_score >= 28 and t_score >= 17 and v_score >= 12 and r_score >= 8 and cw == 0: return "STRONG BUY"
    if total_score >= 65 and f_score >= 22 and t_score >= 13 and cw <= 1: return "BUY"
    if total_score >= 50: return "HOLD"
    return "AVOID"

def score_from_tv(symbol, tv_data, ltp):
    f_res = analyze_fundamentals(tv_data)
    t_res = analyze_technicals(tv_data)
    v_res = analyze_valuation(tv_data)
    r_res = analyze_risk(tv_data)

    total = max(0, min(
        100,
        f_res["score"] + t_res["score"] + v_res["score"] + r_res["score"]
    ))
    warnings = (
        f_res["warnings"] + t_res["warnings"] +
        v_res["warnings"] + r_res["warnings"]
    )

    confidence = (
        f_res["confidence"] * 0.40 +
        t_res["confidence"] * 0.25 +
        v_res["confidence"] * 0.20 +
        r_res["confidence"] * 0.15
    )
    fair = calculate_fair_value(tv_data, ltp)

    if fair.get("fair_value") and ltp:
        fair["upside_pct"] = round(
            ((fair["fair_value"] - ltp) / ltp) * 100, 2
        )
    else:
        fair["upside_pct"] = None

    return {
        "company": symbol,
        "symbol": symbol,
        "ltp": ltp,
        "score": total,
        "decision": make_decision(
            total,
            f_res["score"],
            t_res["score"],
            v_res["score"],
            r_res["score"],
            warnings
        ),
        "data_confidence_pct": round(confidence * 100, 1),
        "breakdown": {
            "fundamental": {
                "score": f_res["score"], "max": 40,
                "confidence": f_res["confidence"],
                "data": f_res["data"],
                "positives": f_res["positives"],
                "warnings": f_res["warnings"]
            },
            "technical": {
                "score": t_res["score"], "max": 25,
                "confidence": t_res["confidence"],
                "data": t_res["data"],
                "positives": t_res["positives"],
                "warnings": t_res["warnings"]
            },
            "valuation": {
                "score": v_res["score"], "max": 20,
                "confidence": v_res["confidence"],
                "data": v_res["data"],
                "positives": v_res["positives"],
                "warnings": v_res["warnings"]
            },
            "risk": {
                "score": r_res["score"], "max": 15,
                "confidence": r_res["confidence"],
                "data": r_res["data"],
                "positives": r_res["positives"],
                "warnings": r_res["warnings"]
            }
        },
        "fair_value_analysis": fair,
        "warnings": warnings,
        "analyzed_at": datetime.now(timezone.utc).isoformat()
    }


# =========================================================
# ROUTES
# =========================================================

DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "LT", "SBIN", "BHARTIARTL", "ITC", "AXISBANK"
]

def extract_ltp(obj):
    preferred_keys = ("ltp", "lastPrice", "pLtp", "last_price")
    if isinstance(obj, dict):
        for k in preferred_keys:
            if k in obj and num(obj[k]) is not None:
                return num(obj[k])
        for v in obj.values():
            result = extract_ltp(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = extract_ltp(item)
            if result is not None:
                return result
    return None

def analyze_symbol(symbol: str, neo_client):
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol is required.")

    stock = find_stock(symbol, neo_client)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found on Kotak Neo.")

    try:
        quote = neo_client.quotes(
            instrument_tokens=[{
                "instrument_token": clean_token(stock.get("pSymbol")),
                "exchange_segment": stock.get("pExchSeg", "nse_cm").lower()
            }],
            quote_type="all"
        )
        ltp = extract_ltp(quote)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unable to retrieve live LTP: {str(e)}")

    if ltp is None:
        raise HTTPException(status_code=502, detail="Unable to retrieve live LTP.")

    try:
        tv_data = fetch_tradingview_data(symbol)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"TradingView data request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    result = score_from_tv(symbol, tv_data, ltp)
    result["live_ltp_source"] = "Kotak Neo"
    result["market_data_source"] = "TradingView Scanner"

    # Keep the last 1000 analyses in memory for the current Render instance.
    score_history.append({
        "time": result["analyzed_at"],
        "symbol": symbol,
        "score": result["score"],
        "decision": result["decision"],
        "ltp": result["ltp"]
    })
    if len(score_history) > 1000:
        del score_history[:-1000]

    return result

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "StockMeter",
        "version": "2.0",
        "active_sessions": len(sessions)
    }

@app.get("/")
def home():
    candidates = [
        Path(__file__).parent / "index.html",
        Path(__file__).parent / "frontend" / "index.html",
        Path(__file__).parent.parent / "frontend" / "index.html",
        Path.cwd() / "frontend" / "index.html",
        Path.cwd() / "index.html"
    ]
    for local_path in candidates:
        if local_path.exists():
            return FileResponse(str(local_path))
    return {
        "error": "index.html not found",
        "checked": [str(x) for x in candidates]
    }

@app.post("/api/login")
def api_login(x: LoginRequest):
    return login_neo(x.totp)

@app.post("/api/score")
def api_score(x: StockRequest, request: Request):
    session = get_session(request)
    return analyze_symbol(x.company, session["neo"])

@app.get("/api/news")
def api_news(q: str = "Indian stock market", limit: int = 5):
    try:
        limit = max(1, min(int(limit), 20))
        q = (q or "Indian stock market").strip()[:120]
        rss_url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(q + " when:2d")
            + "&hl=en-IN&gl=IN&ceid=IN%3Aen"
        )
        r = requests.get(
            rss_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)

        items = []
        for item in root.findall("./channel/item")[:limit]:
            items.append({
                "title": item.findtext("title") or "Market update",
                "link": item.findtext("link") or "",
                "published": item.findtext("pubDate") or "",
                "source": "Google News"
            })
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"News fetch failed: {str(e)}")

@app.post("/api/compare")
def api_compare(x: CompareRequest, request: Request):
    session = get_session(request)
    symbols = list(dict.fromkeys(
        s.strip().upper() for s in x.symbols if s and s.strip()
    ))[:12]
    if not symbols:
        raise HTTPException(status_code=400, detail="Provide at least one stock symbol.")

    results, errors = [], []
    for symbol in symbols:
        try:
            results.append(analyze_symbol(symbol, session["neo"]))
        except HTTPException as e:
            errors.append({"symbol": symbol, "error": str(e.detail)})

    return {
        "results": results,
        "errors": errors,
        "count": len(results)
    }

@app.post("/api/screener")
def api_screener(x: ScreenerRequest, request: Request):
    session = get_session(request)

    symbols = list(dict.fromkeys(
        s.strip().upper() for s in (x.symbols or DEFAULT_UNIVERSE)
        if s and s.strip()
    ))[:25]

    results, errors = [], []
    for symbol in symbols:
        try:
            result = analyze_symbol(symbol, session["neo"])
            b = result["breakdown"]
            fdata = b["fundamental"]["data"]
            tdata = b["technical"]["data"]
            vdata = b["valuation"]["data"]
            rdata = b["risk"]["data"]

            roe = num(fdata.get("roe_pct"))
            pe = num(fdata.get("pe"))
            rg = num(fdata.get("revenue_growth_pct"))
            de = num(fdata.get("debt_to_equity"))
            rsi = num(tdata.get("rsi"))
            fcf = num(vdata.get("fcf_margin_pct"))
            if fcf is None:
                fcf = num(vdata.get("fcf"))

            checks = [
                x.min_score is None or result["score"] >= x.min_score,
                x.min_roe is None or (roe is not None and roe >= x.min_roe),
                x.max_pe is None or (pe is not None and pe <= x.max_pe),
                x.min_revenue_growth is None or (rg is not None and rg >= x.min_revenue_growth),
                x.max_debt_to_equity is None or (de is not None and de <= x.max_debt_to_equity),
                x.min_rsi is None or (rsi is not None and rsi >= x.min_rsi),
                x.max_rsi is None or (rsi is not None and rsi <= x.max_rsi),
                x.min_fcf_margin is None or (fcf is not None and fcf >= x.min_fcf_margin),
            ]
            if all(checks):
                results.append(result)
        except HTTPException as e:
            errors.append({"symbol": symbol, "error": str(e.detail)})

    results.sort(key=lambda r: r["score"], reverse=True)
    return {
        "results": results,
        "errors": errors,
        "count": len(results)
    }

@app.post("/api/portfolio")
def api_portfolio(x: PortfolioRequest, request: Request):
    session = get_session(request)
    if not x.holdings:
        raise HTTPException(status_code=400, detail="Portfolio is empty.")

    rows, errors = [], []
    total_invested = 0.0
    total_value = 0.0
    weighted_score = 0.0

    for holding in x.holdings[:30]:
        if holding.quantity <= 0 or holding.avg_price < 0:
            errors.append({"symbol": holding.symbol, "error": "Invalid quantity/average price."})
            continue

        try:
            result = analyze_symbol(holding.symbol, session["neo"])
            invested = holding.quantity * holding.avg_price
            value = holding.quantity * result["ltp"]
            pnl = value - invested
            total_invested += invested
            total_value += value
            weighted_score += result["score"] * value

            rows.append({
                "symbol": result["symbol"],
                "quantity": holding.quantity,
                "avg_price": holding.avg_price,
                "ltp": result["ltp"],
                "invested": round(invested, 2),
                "value": round(value, 2),
                "pnl": round(pnl, 2),
                "score": result["score"],
                "decision": result["decision"]
            })
        except HTTPException as e:
            errors.append({"symbol": holding.symbol.upper(), "error": str(e.detail)})

    portfolio_score = (
        round(weighted_score / total_value, 2)
        if total_value > 0 else None
    )
    pnl = total_value - total_invested
    pnl_pct = (pnl / total_invested * 100) if total_invested else None

    for row in rows:
        row["allocation_pct"] = (
            round(row["value"] / total_value * 100, 2)
            if total_value else 0
        )

    return {
        "rows": rows,
        "errors": errors,
        "total_invested": round(total_invested, 2),
        "total_value": round(total_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "portfolio_score": portfolio_score
    }

@app.get("/api/history")
def api_history(symbol: Optional[str] = None, limit: int = 50, request: Request = None):
    if request is not None:
        get_session(request)
    limit = max(1, min(int(limit), 200))
    symbol = symbol.strip().upper() if symbol else None
    items = [
        x for x in reversed(score_history)
        if not symbol or x["symbol"] == symbol
    ][:limit]
    return {"items": items}

@app.post("/api/history")
def api_history_add(x: HistoryRequest, request: Request):
    get_session(request)
    item = {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": x.symbol.strip().upper(),
        "score": x.score,
        "decision": x.decision,
        "ltp": x.ltp
    }
    score_history.append(item)
    return {"ok": True, "item": item}

@app.get("/api/alerts")
def api_alerts(request: Request):
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    get_session(request)
    items = [
        {**v, "id": k}
        for k, v in alerts_store.items()
        if v.get("session_id") == session_id
    ]
    return {"items": items}

@app.post("/api/alerts")
def api_alert_create(x: AlertRequest, request: Request):
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    get_session(request)

    alert_id = secrets.token_urlsafe(12)
    alerts_store[alert_id] = {
        "session_id": session_id,
        "symbol": x.symbol.strip().upper(),
        "min_score": x.min_score,
        "max_score": x.max_score,
        "below_price": x.below_price,
        "above_price": x.above_price,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    return {"ok": True, "id": alert_id, "alert": alerts_store[alert_id]}

@app.delete("/api/alerts/{alert_id}")
def api_alert_delete(alert_id: str, request: Request):
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    get_session(request)

    item = alerts_store.get(alert_id)
    if not item or item.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alerts_store.pop(alert_id, None)
    return {"ok": True}

@app.post("/api/alerts/check")
def api_alert_check(x: AlertRequest, request: Request):
    session = get_session(request)
    result = analyze_symbol(x.symbol, session["neo"])

    triggered = False
    reasons = []

    if x.min_score is not None and result["score"] >= x.min_score:
        triggered = True
        reasons.append(f"score >= {x.min_score}")
    if x.max_score is not None and result["score"] <= x.max_score:
        triggered = True
        reasons.append(f"score <= {x.max_score}")
    if x.below_price is not None and result["ltp"] <= x.below_price:
        triggered = True
        reasons.append(f"price <= ₹{x.below_price}")
    if x.above_price is not None and result["ltp"] >= x.above_price:
        triggered = True
        reasons.append(f"price >= ₹{x.above_price}")

    return {
        "triggered": triggered,
        "reasons": reasons,
        "symbol": result["symbol"],
        "score": result["score"],
        "decision": result["decision"],
        "ltp": result["ltp"]
    }

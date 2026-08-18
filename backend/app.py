import os
import math
import secrets
import time
import asyncio
from pathlib import Path
import json
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote_plus
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
news_cache: Dict[str, Dict[str, Any]] = {}
news_cache_ttl = 90

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
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    volume_spike_pct: Optional[float] = None

class HistoryRequest(BaseModel):
    symbol: str
    score: float
    decision: str = ""
    ltp: Optional[float] = None

# Existing per-stock screener default. The NSE-wide market leaderboard
# below is not limited by this list.
DEFAULT_UNIVERSE = [
    "RELIANCE","HDFCBANK","ICICIBANK","BHARTIARTL","TCS","INFY","ITC",
    "SBIN","LT","AXISBANK","KOTAKBANK","M&M","HINDUNILVR","BAJFINANCE",
    "MARUTI","SUNPHARMA","HCLTECH","TITAN","ULTRACEMCO","ADANIENT",
    "NTPC","ONGC","POWERGRID","TATASTEEL","TATAMOTORS","WIPRO","TECHM",
    "ASIANPAINT","NESTLEIND","JSWSTEEL"
]

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
            "close", "change", "change_abs", "volume",
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
# MARKET PULSE DATA (NSE + TRADINGVIEW)
# =========================================================
_nse_session = requests.Session()
_nse_session.headers.update({
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36",
    "Accept":"application/json,text/plain,*/*","Accept-Language":"en-US,en;q=0.9",
    "Referer":"https://www.nseindia.com/"
})
_pulse_cache: Dict[str, Dict[str, Any]] = {}
_pulse_cache_ttl = 20

def _cached(key):
    item=_pulse_cache.get(key)
    return item["data"] if item and time.time()-item["time"]<_pulse_cache_ttl else None

def _put_cache(key,data):
    _pulse_cache[key]={"time":time.time(),"data":data}; return data

def nse_json(path, params=None):
    base="https://www.nseindia.com"
    _nse_session.get(base,timeout=6)
    r=_nse_session.get(base+path,params=params,timeout=8); r.raise_for_status(); return r.json()

def _nse_variation(kind,limit=10):
    candidates=[{"index":kind,"type":"FOSec","limit":limit},{"index":kind,"type":"AllSec","limit":limit}]
    if kind=="losers": candidates=[{"index":"loosers","type":"FOSec","limit":limit},{"index":"losers","type":"FOSec","limit":limit},{"index":"loosers","type":"AllSec","limit":limit}]
    last=None
    for params in candidates:
        try:
            raw=nse_json("/api/live-analysis-variations",params=params); rows=raw.get("data",[]) if isinstance(raw,dict) else []
            if rows:
                out=[]
                for row in rows[:limit]:
                    out.append({"symbol":str(row.get("symbol") or row.get("symbolName") or row.get("identifier") or "").replace("-EQ","").upper(),"price":num(row.get("ltp") or row.get("lastPrice") or row.get("closePrice")),"change_pct":num(row.get("perChange") or row.get("pChange") or row.get("percentChange")),"change_abs":num(row.get("netPrice") or row.get("change")),"volume":num(row.get("trdVol") or row.get("volume")),"source":"NSE"})
                return out
        except Exception as e: last=e
    raise RuntimeError(str(last) if last else "NSE variation data unavailable")

def _tv_scan_tickers(tickers,columns,sort_by=None,sort_order="desc",limit=20):
    payload={"symbols":{"tickers":tickers,"query":{"types":[]}},"columns":columns,"range":[0,limit]}
    if sort_by: payload["sort"]={"sortBy":sort_by,"sortOrder":sort_order}
    r=requests.post("https://scanner.tradingview.com/india/scan",json=payload,headers={"User-Agent":"Mozilla/5.0","Content-Type":"application/json"},timeout=15)
    r.raise_for_status(); return [dict(zip(columns,row.get("d",[]))) for row in r.json().get("data",[])]

def _tv_scan_all_nse(columns,sort_by=None,sort_order="desc",limit=25):
    """Scan the broad NSE primary-equity universe instead of NIFTY 50 only."""
    payload={
        "symbols":{"tickers":[],"query":{"types":["stock"]}},
        "columns":columns,
        "filter":[
            {"left":"is_primary","operation":"equal","right":True},
            {"left":"exchange","operation":"equal","right":"NSE"}
        ],
        "range":[0, max(1, min(int(limit), 5000))]
    }
    if sort_by:
        payload["sort"]={"sortBy":sort_by,"sortOrder":sort_order}

    r=requests.post(
        "https://scanner.tradingview.com/india/scan",
        json=payload,
        headers={"User-Agent":"Mozilla/5.0","Content-Type":"application/json","Accept":"application/json"},
        timeout=20
    )
    r.raise_for_status()
    body=r.json()
    rows=body.get("data",[]) if isinstance(body,dict) else []
    return [dict(zip(columns,row.get("d",[]))) for row in rows]

def _tv_rows_to_movers(rows, source="TradingView NSE scanner"):
    out=[]
    for row in rows:
        symbol=str(row.get("name") or row.get("description") or "").upper().replace("NSE:","")
        if not symbol:
            continue
        out.append({
            "symbol":symbol,
            "name":row.get("description") or symbol,
            "price":num(row.get("close")),
            "change_pct":num(row.get("change")),
            "change_abs":num(row.get("change_abs")),
            "return_1m_pct":num(row.get("Perf.1M")),
            "volume":num(row.get("volume")),
            "market_cap":num(row.get("market_cap_basic")),
            "source":source
        })
    return out

def _nifty50_fallback_pulse():
    symbols="RELIANCE HDFCBANK ICICIBANK BHARTIARTL TCS INFY ITC SBIN LT AXISBANK KOTAKBANK M&M HINDUNILVR BAJFINANCE MARUTI SUNPHARMA HCLTECH TITAN ULTRACEMCO ADANIENT NTPC ONGC POWERGRID TATASTEEL TATAMOTORS WIPRO TECHM ASIANPAINT NESTLEIND JSWSTEEL COALINDIA BAJAJFINSV ADANIPORTS HINDALCO GRASIM CIPLA EICHERMOT DRREDDY DIVISLAB APOLLOHOSP TATACONSUM BPCL HEROMOTOCO BRITANNIA SHRIRAMFIN TRENT BEL INDUSINDBK SBILIFE HDFCLIFE".split()
    cols=["name","description","close","change","change_abs","volume"]
    rows=_tv_scan_tickers([f"NSE:{x}" for x in symbols],cols,"change","desc",50); out=[]
    for x in rows:
        sym=str(x.get("name") or x.get("description") or "").upper().replace("NSE:","")
        out.append({"symbol":sym,"price":num(x.get("close")),"change_pct":num(x.get("change")),"change_abs":num(x.get("change_abs")),"volume":num(x.get("volume")),"source":"TradingView"})
    return out

def get_monthly_movers(limit=10):
    """Top and bottom 1-month performers across NSE primary equities."""
    limit=max(5,min(int(limit),25))
    key=f"monthly-movers:{limit}"
    cached=_cached(key)
    if cached is not None:
        return cached

    cols=["name","description","close","change","change_abs","volume","market_cap_basic","Perf.1M"]
    try:
        top_rows=_tv_scan_all_nse(cols,"Perf.1M","desc",max(limit,25))
        bottom_rows=_tv_scan_all_nse(cols,"Perf.1M","asc",max(limit,25))
        gainers=_tv_rows_to_movers(top_rows)[:limit]
        losers=_tv_rows_to_movers(bottom_rows)[:limit]
        gainers=[x for x in gainers if x.get("return_1m_pct") is not None]
        losers=[x for x in losers if x.get("return_1m_pct") is not None]

        return _put_cache(key,{
            "gainers_1m":gainers,
            "losers_1m":losers,
            "universe":"NSE primary equity stocks",
            "period":"1M",
            "source":"TradingView NSE-wide scanner",
            "updated_at":datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return _put_cache(key,{
            "gainers_1m":[],
            "losers_1m":[],
            "universe":"NSE primary equity stocks",
            "period":"1M",
            "source":"unavailable",
            "error":str(e),
            "updated_at":datetime.now(timezone.utc).isoformat()
        })

def get_top_movers(limit=10):
    key=f"movers:{limit}"; cached=_cached(key)
    if cached is not None:return cached
    try:
        g=_nse_variation("gainers",limit)
        l=_nse_variation("losers",limit)
        src="NSE"
    except Exception:
        # Fallback is NSE-wide, not NIFTY 50-only.
        cols=["name","description","close","change","change_abs","volume"]
        try:
            rows_up=_tv_scan_all_nse(cols,"change","desc",max(limit,25))
            rows_dn=_tv_scan_all_nse(cols,"change","asc",max(limit,25))
            g=_tv_rows_to_movers(rows_up)[:limit]
            l=_tv_rows_to_movers(rows_dn)[:limit]
            src="TradingView NSE-wide fallback"
        except Exception:
            rows=_nifty50_fallback_pulse()
            g=sorted(rows,key=lambda x:x.get("change_pct") if x.get("change_pct") is not None else -999,reverse=True)[:limit]
            l=sorted(rows,key=lambda x:x.get("change_pct") if x.get("change_pct") is not None else 999)[:limit]
            src="TradingView NIFTY 50 emergency fallback"

    monthly=get_monthly_movers(limit)
    return _put_cache(key,{
        "gainers":g,
        "losers":l,
        "monthly":monthly,
        "source":src,
        "updated_at":datetime.now(timezone.utc).isoformat()
    })

SECTOR_TICKERS={"NIFTY Auto":"NSE:CNXAUTO","NIFTY Bank":"NSE:BANKNIFTY","NIFTY IT":"NSE:CNXIT","NIFTY Pharma":"NSE:CNXPHARMA","NIFTY Metal":"NSE:CNXMETAL","NIFTY FMCG":"NSE:CNXFMCG","NIFTY Realty":"NSE:CNXREALTY","NIFTY Media":"NSE:CNXMEDIA","NIFTY PSU Bank":"NSE:CNXPSUBANK","NIFTY Private Bank":"NSE:NIFTYPVTBANK","NIFTY Financial Services":"NSE:NIFTYFINSERVICE","NIFTY Energy":"NSE:NIFTYENERGY","NIFTY Healthcare":"NSE:NIFTYHEALTHCARE","NIFTY Consumer Durables":"NSE:NIFTYCONSUMERDURABLES","NIFTY Oil & Gas":"NSE:NIFTYOILANDGAS"}

def get_sector_performance():
    key="sectors"; cached=_cached(key)
    if cached is not None:return cached
    cols=["name","description","close","change","change_abs","volume"]
    try:
        rows=_tv_scan_tickers(list(SECTOR_TICKERS.values()),cols,"change","desc",len(SECTOR_TICKERS)); items=[]
        for label,ticker in SECTOR_TICKERS.items():
            suffix=ticker.split(":")[-1].upper(); r=next((v for v in rows if suffix in str(v.get("name") or v.get("description") or "").upper()),None)
            if r: items.append({"sector":label,"symbol":ticker,"price":num(r.get("close")),"change_pct":num(r.get("change")),"change_abs":num(r.get("change_abs")),"volume":num(r.get("volume")),"source":"TradingView"})
        items.sort(key=lambda x:x.get("change_pct") if x.get("change_pct") is not None else -999,reverse=True)
        return _put_cache(key,{"items":items,"source":"TradingView sector indices","updated_at":datetime.now(timezone.utc).isoformat()})
    except Exception as e:return _put_cache(key,{"items":[],"source":"unavailable","error":str(e),"updated_at":datetime.now(timezone.utc).isoformat()})

def get_market_breadth():
    key="breadth"; cached=_cached(key)
    if cached is not None:return cached
    try:
        raw=nse_json("/api/equity-stockIndices",params={"index":"NIFTY 50"}); rows=raw.get("data",[]) if isinstance(raw,dict) else []
        adv=sum(1 for x in rows if num(x.get("pChange")) is not None and num(x.get("pChange"))>0); dec=sum(1 for x in rows if num(x.get("pChange")) is not None and num(x.get("pChange"))<0); unch=sum(1 for x in rows if num(x.get("pChange"))==0); total=adv+dec+unch
        return _put_cache(key,{"advances":adv,"declines":dec,"unchanged":unch,"total":total,"ad_ratio":round(adv/dec,2) if dec else None,"scope":"NIFTY 50","source":"NSE","updated_at":datetime.now(timezone.utc).isoformat()})
    except Exception as e:return _put_cache(key,{"advances":None,"declines":None,"unchanged":None,"total":None,"ad_ratio":None,"scope":"NIFTY 50","source":"unavailable","error":str(e),"updated_at":datetime.now(timezone.utc).isoformat()})

def get_fii_dii():
    key="fii_dii"; cached=_cached(key)
    if cached is not None:return cached
    try:
        raw=nse_json("/api/fiidiiTradeReact"); rows=raw if isinstance(raw,list) else raw.get("data",[]) if isinstance(raw,dict) else []; items=[]
        for row in rows:
            cat=str(row.get("category") or row.get("categoryName") or "").upper()
            buy=num(row.get("buyValue") or row.get("buyValueInCr") or row.get("buy")); sell=num(row.get("sellValue") or row.get("sellValueInCr") or row.get("sell")); net=num(row.get("netValue") or row.get("netValueInCr") or row.get("net"))
            if cat and (buy is not None or sell is not None or net is not None): items.append({"category":cat,"buy":buy,"sell":sell,"net":net,"date":row.get("date") or row.get("tradeDate")})
        return _put_cache(key,{"items":items,"source":"NSE","updated_at":datetime.now(timezone.utc).isoformat()})
    except Exception as e:return _put_cache(key,{"items":[],"source":"unavailable","error":str(e),"updated_at":datetime.now(timezone.utc).isoformat()})

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
            "last_used": time.time(),
            "ticks": {},
            "token_map": {},
            "subscribed": set()
        }

        # Use one callback per Kotak client and keep the latest tick in memory.
        # Browser WebSocket clients read from this cache, so one browser tab
        # cannot overwrite another tab's callback.
        def _neo_on_message(message):
            tick = extract_live_tick(message)
            if tick and tick.get("token") is not None and tick.get("ltp") is not None:
                sess = sessions.get(session_id, {})
                token = clean_token(tick["token"])
                if not tick.get("symbol"):
                    tick["symbol"] = sess.get("token_map", {}).get(token, "")
                sess.get("ticks", {})[token] = tick

        def _neo_on_error(message):
            sessions.get(session_id, {})["last_ws_error"] = str(message)

        neo_client.on_message = _neo_on_message
        neo_client.on_error = _neo_on_error
        neo_client.on_close = lambda message: None
        neo_client.on_open = lambda message: None
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
# STOCK ANALYSIS ORCHESTRATOR
# =========================================================

def analyze_symbol(symbol: str, neo_client=None):
    """Analyze a stock with TradingView scanner data and optional Kotak Neo live price.

    Public analysis does not require a Kotak session. If a logged-in Neo client is
    available, its quote is preferred for LTP; otherwise TradingView scanner close
    is used as the fallback price.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol is required.")

    # TradingView scanner is the no-login analysis source.
    try:
        tv_data = fetch_tradingview_data(symbol)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Market data request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    tv_ltp = num(tv_data.get("close"))
    ltp = tv_ltp
    ltp_source = "TradingView scanner"

    # Prefer the broker quote when the user has logged in.
    if neo_client is not None:
        try:
            stock = find_stock(symbol, neo_client)
            if stock:
                token = clean_token(stock.get("pSymbol"))
                segment = str(stock.get("pExchSeg", "nse_cm")).lower()
                quote = neo_client.quotes(
                    instrument_tokens=[{
                        "instrument_token": token,
                        "exchange_segment": segment
                    }],
                    quote_type="all"
                )
                neo_ltp = extract_ltp(quote)
                if neo_ltp is not None:
                    ltp = neo_ltp
                    ltp_source = "Kotak Neo"
        except Exception:
            # Never make stock analysis fail just because the broker quote failed.
            pass

    if ltp is None:
        raise HTTPException(status_code=502, detail=f"No price data available for {symbol}.")

    result = score_from_tv(symbol, tv_data, ltp)
    result["price_change_pct"] = num(tv_data.get("change"))
    result["price_change_abs"] = num(tv_data.get("change_abs"))
    result["volume"] = num(tv_data.get("volume"))
    result["live_ltp_source"] = ltp_source
    result["market_data_source"] = "TradingView Scanner"

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


# =========================================================
# REAL-TIME KOTAK NEO MARKET STREAM
# =========================================================

def extract_live_tick(message):
    """Normalize common Kotak Neo websocket tick shapes into a small payload."""
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except Exception:
            return None
    if isinstance(message, list):
        for item in message:
            hit = extract_live_tick(item)
            if hit:
                return hit
        return None
    if not isinstance(message, dict):
        return None

    # SDK payloads can expose data directly or under data/result.
    candidates = [message]
    for key in ("data", "result", "response", "message"):
        value = message.get(key)
        if isinstance(value, (dict, list)):
            candidates.append(value)

    for obj in candidates:
        if isinstance(obj, list):
            for item in obj:
                hit = extract_live_tick(item)
                if hit:
                    return hit
            continue
        if not isinstance(obj, dict):
            continue

        price = None
        for k in ("ltp", "lastPrice", "last_traded_price", "pLtp", "LTP", "last_price"):
            if k in obj and num(obj.get(k)) is not None:
                price = num(obj.get(k)); break
        if price is None:
            continue

        change = None
        for k in ("change", "changePercent", "change_percentage", "change_pct", "pChange"):
            if k in obj and num(obj.get(k)) is not None:
                change = num(obj.get(k)); break

        symbol = obj.get("trading_symbol") or obj.get("tradingSymbol") or obj.get("pTrdSymbol") or obj.get("symbol")
        token = obj.get("instrument_token") or obj.get("instrumentToken") or obj.get("pSymbol")
        return {
            "type": "tick",
            "symbol": str(symbol or "").replace("-EQ", "").upper(),
            "token": clean_token(token) if token is not None else None,
            "ltp": price,
            "change_pct": change,
            "raw_time": datetime.now(timezone.utc).isoformat()
        }
    return None

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Stream Kotak Neo ticks from the per-login client cache to the browser."""
    await websocket.accept()
    session_id = websocket.query_params.get("session", "").strip()
    symbol = websocket.query_params.get("symbol", "").strip().upper()

    if not session_id or session_id not in sessions:
        await websocket.send_json({"type": "error", "message": "LOGIN_REQUIRED"})
        await websocket.close(code=4401)
        return
    if not symbol:
        await websocket.send_json({"type": "error", "message": "SYMBOL_REQUIRED"})
        await websocket.close(code=4400)
        return

    session = sessions[session_id]
    neo_client = session["neo"]
    stock = find_stock(symbol, neo_client)
    if not stock:
        await websocket.send_json({"type": "error", "message": f"Stock '{symbol}' not found."})
        await websocket.close(code=4404)
        return

    token = clean_token(stock.get("pSymbol"))
    segment = str(stock.get("pExchSeg", "nse_cm")).lower()
    session["token_map"][token] = symbol

    try:
        if token not in session["subscribed"]:
            neo_client.subscribe(
                instrument_tokens=[{
                    "instrument_token": token,
                    "exchange_segment": segment
                }],
                isIndex=False,
                isDepth=False
            )
            session["subscribed"].add(token)

        await websocket.send_json({
            "type": "stream_status",
            "status": "connected",
            "symbol": symbol,
            "source": "Kotak Neo WebSocket"
        })

        # Immediate quote snapshot.
        try:
            snapshot = neo_client.quotes(
                instrument_tokens=[{
                    "instrument_token": token,
                    "exchange_segment": segment
                }],
                quote_type="all"
            )
            snap_ltp = extract_ltp(snapshot)
            if snap_ltp is not None:
                await websocket.send_json({
                    "type": "tick",
                    "symbol": symbol,
                    "token": token,
                    "ltp": snap_ltp,
                    "change_pct": None,
                    "raw_time": datetime.now(timezone.utc).isoformat()
                })
        except Exception:
            pass

        last_sent = None
        while True:
            await asyncio.sleep(0.20)
            if session_id not in sessions:
                break

            tick = session.get("ticks", {}).get(token)
            if not tick:
                continue

            payload = dict(tick)
            payload["symbol"] = symbol
            if payload != last_sent:
                await websocket.send_json(payload)
                last_sent = payload

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "StockMeter",
        "version": "5.1-nse-wide-movers",
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
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    neo = None
    if session_id:
        try:
            neo = get_session(request)["neo"]
        except HTTPException:
            neo = None
    return analyze_symbol(x.company, neo)

@app.get("/api/market-overview")
def api_market_overview(request: Request):
    """Market snapshot. Uses Kotak Neo when logged in, TradingView scanner otherwise."""
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    if session_id and session_id in sessions:
        try:
            neo = sessions[session_id]["neo"]
            names = [("NIFTY", "Nifty 50", "nse_cm"), ("BANKNIFTY", "Nifty Bank", "nse_cm"), ("SENSEX", "SENSEX", "bse_cm"), ("INDIA VIX", "INDIA VIX", "nse_cm")]
            items=[]
            for label, token, seg in names:
                try:
                    q=neo.quotes(instrument_tokens=[{"instrument_token":token,"exchange_segment":seg}], quote_type="all")
                    raw=q
                    if isinstance(raw,str):
                        try: raw=json.loads(raw)
                        except Exception: pass
                    # Search common nested shapes.
                    def walk(v):
                        if isinstance(v,dict):
                            yield v
                            for vv in v.values(): yield from walk(vv)
                        elif isinstance(v,list):
                            for vv in v: yield from walk(vv)
                    hit=None
                    for o in walk(raw):
                        p=next((num(o.get(k)) for k in ("ltp","lastPrice","pLtp","LTP","last_price") if o.get(k) is not None),None)
                        if p is not None:
                            hit=o; break
                    if hit:
                        ch=next((num(hit.get(k)) for k in ("changePercent","change_percentage","change_pct","pChange") if hit.get(k) is not None),None)
                        items.append({"symbol":label,"price":p,"change_pct":ch,"source":"Kotak Neo"})
                except Exception:
                    pass
            if items:
                return {"items":items,"source":"Kotak Neo","updated_at":datetime.now(timezone.utc).isoformat()}
        except Exception:
            pass

    symbols=["NSE:NIFTY","NSE:BANKNIFTY","BSE:SENSEX"]
    payload={"symbols":{"tickers":symbols},"columns":["close","change","change_abs","volume"]}
    try:
        r=requests.post("https://scanner.tradingview.com/india/scan",json=payload,headers={"User-Agent":"Mozilla/5.0","Content-Type":"application/json"},timeout=10)
        r.raise_for_status(); data=r.json().get("data",[]); out=[]
        for row in data:
            vals=dict(zip(payload["columns"],row.get("d",[])))
            out.append({"symbol":row.get("s","").split(":")[-1],"price":vals.get("close"),"change_pct":vals.get("change"),"change_abs":vals.get("change_abs"),"volume":vals.get("volume"),"source":"TradingView scanner"})
        return {"items":out,"source":"TradingView scanner","updated_at":datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market overview failed: {str(e)}")

@app.get("/api/market-pulse")
def api_market_pulse(limit: int = 10):
    limit=max(5,min(int(limit),25))
    return {"movers":get_top_movers(limit),"sectors":get_sector_performance(),"breadth":get_market_breadth(),"fii_dii":get_fii_dii(),"updated_at":datetime.now(timezone.utc).isoformat()}

@app.get("/api/movers")
def api_movers(limit: int = 10):
    return get_top_movers(max(5,min(int(limit),25)))

@app.get("/api/monthly-movers")
def api_monthly_movers(limit: int = 10):
    """NSE-wide top and bottom 1-month performers."""
    return get_monthly_movers(max(5,min(int(limit),25)))

@app.get("/api/sectors")
def api_sectors(): return get_sector_performance()

@app.get("/api/breadth")
def api_breadth(): return get_market_breadth()

@app.get("/api/fii-dii")
def api_fii_dii(): return get_fii_dii()


@app.get("/api/news")
def api_news(q: str = "Indian stock market", limit: int = 8):
    limit=max(1,min(int(limit),20)); q=(q or "Indian stock market").strip()[:120]
    cache_key=q.lower(); now=time.time()
    cached=news_cache.get(cache_key)
    if cached and now-cached["time"] < news_cache_ttl:
        return {**cached["data"],"cached":True}
    urls=[
        ("https://news.google.com/rss/search?q="+quote_plus(q+" when:2d")+"&hl=en-IN&gl=IN&ceid=IN%3Aen","Google News"),
        ("https://www.bing.com/news/search?q="+quote_plus(q)+"&format=rss","Bing News")
    ]
    errors=[]
    for rss_url,provider in urls:
        try:
            r=requests.get(rss_url,headers={"User-Agent":"Mozilla/5.0"},timeout=10); r.raise_for_status()
            root=ET.fromstring(r.content); items=[]; seen=set()
            for item in root.findall("./channel/item"):
                title=(item.findtext("title") or "Market update").strip()
                key=re.sub(r"\W+"," ",title.lower()).strip()
                if key in seen: continue
                seen.add(key); source_el=item.find("source"); source=(source_el.text.strip() if source_el is not None and source_el.text else provider)
                items.append({"title":title,"link":item.findtext("link") or "","published":item.findtext("pubDate") or "","source":source})
                if len(items)>=limit: break
            if items:
                data={"items":items,"provider":provider,"updated_at":datetime.now(timezone.utc).isoformat(),"cached":False}
                news_cache[cache_key]={"time":now,"data":data}; return data
        except Exception as e: errors.append(f"{provider}: {e}")
    raise HTTPException(status_code=502,detail="News fetch failed. "+" | ".join(errors))

@app.get("/api/live-price")
def api_live_price(symbol: str, request: Request):
    symbol=symbol.strip().upper()
    if not symbol: raise HTTPException(status_code=400,detail="Symbol is required")
    session_id=request.headers.get("X-StockMeter-Session","").strip()
    if session_id and session_id in sessions:
        sess=sessions[session_id]; neo=sess["neo"]; stock=find_stock(symbol,neo)
        if stock:
            token=clean_token(stock.get("pSymbol")); sess["token_map"][token]=symbol
            tick=sess.get("ticks",{}).get(token)
            if tick and tick.get("ltp") is not None:
                return {**tick,"source":"Kotak Neo WebSocket"}
            try:
                q=neo.quotes(instrument_tokens=[{"instrument_token":token,"exchange_segment":str(stock.get("pExchSeg","nse_cm")).lower()}],quote_type="all")
                raw=q
                if isinstance(raw,str):
                    try: raw=json.loads(raw)
                    except Exception: pass
                def walk(v):
                    if isinstance(v,dict):
                        yield v
                        for vv in v.values(): yield from walk(vv)
                    elif isinstance(v,list):
                        for vv in v: yield from walk(vv)
                for o in walk(raw):
                    p=next((num(o.get(k)) for k in ("ltp","lastPrice","pLtp","LTP","last_price") if o.get(k) is not None),None)
                    if p is not None:
                        ch=next((num(o.get(k)) for k in ("changePercent","change_percentage","change_pct","pChange") if o.get(k) is not None),None)
                        return {"type":"tick","symbol":symbol,"token":token,"ltp":p,"change_pct":ch,"source":"Kotak Neo quote","raw_time":datetime.now(timezone.utc).isoformat()}
            except Exception: pass
    try:
        tv=fetch_tradingview_data(symbol); return {"type":"tick","symbol":symbol,"ltp":num(tv.get("close")),"change_pct":num(tv.get("change")),"source":"TradingView scanner","raw_time":datetime.now(timezone.utc).isoformat()}
    except Exception as e: raise HTTPException(status_code=502,detail=f"Live price failed: {e}")

@app.post("/api/compare")
def api_compare(x: CompareRequest, request: Request):
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    neo = None
    if session_id:
        try: neo = get_session(request)["neo"]
        except HTTPException: neo = None
    symbols = list(dict.fromkeys(
        s.strip().upper() for s in x.symbols if s and s.strip()
    ))[:12]
    if not symbols:
        raise HTTPException(status_code=400, detail="Provide at least one stock symbol.")

    results, errors = [], []
    for symbol in symbols:
        try:
            results.append(analyze_symbol(symbol, neo))
        except HTTPException as e:
            errors.append({"symbol": symbol, "error": str(e.detail)})

    return {
        "results": results,
        "errors": errors,
        "count": len(results)
    }

@app.post("/api/screener")
def api_screener(x: ScreenerRequest, request: Request):
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    neo = None
    if session_id:
        try: neo = get_session(request)["neo"]
        except HTTPException: neo = None

    symbols = list(dict.fromkeys(
        s.strip().upper() for s in (x.symbols or DEFAULT_UNIVERSE)
        if s and s.strip()
    ))[:25]

    results, errors = [], []
    for symbol in symbols:
        try:
            result = analyze_symbol(symbol, neo)
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
    session_id = request.headers.get("X-StockMeter-Session", "").strip()
    neo = None
    if session_id:
        try: neo = get_session(request)["neo"]
        except HTTPException: neo = None
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
            result = analyze_symbol(holding.symbol, neo)
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


@app.get("/api/account/holdings")
def api_account_holdings(request: Request):
    session = get_session(request)
    try:
        data = session["neo"].holdings()
        if isinstance(data, str):
            data = json.loads(data)
        return {"items": data.get("data", []) if isinstance(data, dict) else data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Holdings fetch failed: {str(e)}")

@app.get("/api/account/positions")
def api_account_positions(request: Request):
    session = get_session(request)
    try:
        data = session["neo"].positions()
        if isinstance(data, str):
            data = json.loads(data)
        return {"items": data.get("data", []) if isinstance(data, dict) else data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Positions fetch failed: {str(e)}")

@app.get("/api/account/limits")
def api_account_limits(request: Request):
    session = get_session(request)
    try:
        data = session["neo"].limits(segment="ALL", exchange="ALL", product="ALL")
        if isinstance(data, str):
            data = json.loads(data)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Limits fetch failed: {str(e)}")


# =========================================================
# KOTAK NEO EXTENDED DATA ENDPOINTS
# =========================================================

@app.get("/api/neo/search-scrip")
def api_neo_search_scrip(symbol: str, request: Request):
    """Resolve an NSE cash symbol to Kotak's instrument token/trading symbol."""
    session = get_session(request)
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required.")
    stock = find_stock(symbol, session["neo"])
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found on Kotak Neo.")
    return {"item": stock}

@app.get("/api/neo/quote")
def api_neo_quote(symbol: str, quote_type: str = "all", request: Request = None):
    """Kotak Neo quote: LTP/OHLC/depth/52w/circuit/scrip details."""
    session = get_session(request)
    symbol = symbol.strip().upper()
    allowed = {"all", "depth", "ohlc", "ltp", "oi", "52w", "circuit_limits", "scrip_details"}
    if quote_type not in allowed:
        raise HTTPException(status_code=400, detail=f"quote_type must be one of: {', '.join(sorted(allowed))}")
    stock = find_stock(symbol, session["neo"])
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found on Kotak Neo.")
    try:
        data = session["neo"].quotes(
            instrument_tokens=[{
                "instrument_token": clean_token(stock.get("pSymbol")),
                "exchange_segment": str(stock.get("pExchSeg", "nse_cm")).lower()
            }],
            quote_type=quote_type
        )
        if isinstance(data, str):
            try: data = json.loads(data)
            except Exception: pass
        return {"symbol": symbol, "instrument": stock, "quote_type": quote_type, "data": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kotak quote failed: {str(e)}")

@app.get("/api/neo/indices")
def api_neo_indices(request: Request):
    """Live index quotes after Kotak login."""
    session = get_session(request)
    instruments = [
        {"instrument_token": "Nifty 50", "exchange_segment": "nse_cm"},
        {"instrument_token": "Nifty Bank", "exchange_segment": "nse_cm"},
        {"instrument_token": "SENSEX", "exchange_segment": "bse_cm"},
        {"instrument_token": "INDIA VIX", "exchange_segment": "nse_cm"},
    ]
    try:
        data = session["neo"].quotes(instrument_tokens=instruments, quote_type="all")
        if isinstance(data, str):
            try: data = json.loads(data)
            except Exception: pass
        return {"items": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kotak index quotes failed: {str(e)}")

@app.get("/api/neo/holdings")
def api_neo_holdings(request: Request):
    session = get_session(request)
    try:
        data = session["neo"].holdings()
        if isinstance(data, str):
            data = json.loads(data)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kotak holdings failed: {str(e)}")

@app.get("/api/neo/positions")
def api_neo_positions(request: Request):
    session = get_session(request)
    try:
        data = session["neo"].positions()
        if isinstance(data, str):
            data = json.loads(data)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kotak positions failed: {str(e)}")

@app.get("/api/neo/limits")
def api_neo_limits(request: Request):
    session = get_session(request)
    try:
        data = session["neo"].limits(segment="ALL", exchange="ALL", product="ALL")
        if isinstance(data, str):
            data = json.loads(data)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kotak limits failed: {str(e)}")

@app.get("/api/live-status")
def api_live_status(request: Request):
    sid=request.headers.get("X-StockMeter-Session","").strip()
    if not sid or sid not in sessions:
        return {"logged_in":False,"source":"TradingView scanner","websocket":"offline"}
    s=sessions[sid]
    return {"logged_in":True,"source":"Kotak Neo","websocket":"ready","tick_count":len(s.get("ticks",{})),"last_ws_error":s.get("last_ws_error") }

@app.get("/api/history")
def api_history(symbol: Optional[str] = None, limit: int = 50, request: Request = None):
    limit = max(1, min(int(limit), 200))
    symbol = symbol.strip().upper() if symbol else None
    items = [
        x for x in reversed(score_history)
        if not symbol or x["symbol"] == symbol
    ][:limit]
    return {"items": items}

@app.post("/api/history")
def api_history_add(x: HistoryRequest, request: Request):
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
        "min_rsi": x.min_rsi,
        "max_rsi": x.max_rsi,
        "min_change_pct": x.min_change_pct,
        "max_change_pct": x.max_change_pct,
        "volume_spike_pct": x.volume_spike_pct,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_triggered_at": None
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
    session_id=request.headers.get("X-StockMeter-Session","").strip()
    neo=None
    if session_id:
        try: neo=get_session(request)["neo"]
        except HTTPException: neo=None
    result=analyze_symbol(x.symbol,neo)
    triggered=False; reasons=[]
    if x.min_score is not None and result["score"]>=x.min_score: triggered=True; reasons.append(f"score >= {x.min_score}")
    if x.max_score is not None and result["score"]<=x.max_score: triggered=True; reasons.append(f"score <= {x.max_score}")
    if x.below_price is not None and result["ltp"]<=x.below_price: triggered=True; reasons.append(f"price <= ₹{x.below_price}")
    if x.above_price is not None and result["ltp"]>=x.above_price: triggered=True; reasons.append(f"price >= ₹{x.above_price}")
    raw=result.get("technical",{}).get("data",{}) if isinstance(result.get("technical"),dict) else {}
    rsi=num(raw.get("rsi") or raw.get("RSI")); change_pct=num(result.get("price_change_pct")); volume=num(result.get("volume")); avg_volume=num(raw.get("average_volume_30d_calc") or raw.get("average_volume_60d_calc") or raw.get("volume_sma")); volume_spike=(volume/avg_volume*100-100) if volume is not None and avg_volume and avg_volume>0 else None
    if x.min_rsi is not None and rsi is not None and rsi>=x.min_rsi: triggered=True; reasons.append(f"RSI >= {x.min_rsi}")
    if x.max_rsi is not None and rsi is not None and rsi<=x.max_rsi: triggered=True; reasons.append(f"RSI <= {x.max_rsi}")
    if x.min_change_pct is not None and change_pct is not None and change_pct>=x.min_change_pct: triggered=True; reasons.append(f"change % >= {x.min_change_pct}")
    if x.max_change_pct is not None and change_pct is not None and change_pct<=x.max_change_pct: triggered=True; reasons.append(f"change % <= {x.max_change_pct}")
    if x.volume_spike_pct is not None and volume_spike is not None and volume_spike>=x.volume_spike_pct: triggered=True; reasons.append(f"volume spike >= {x.volume_spike_pct}%")
    return {"triggered":triggered,"reasons":reasons,"symbol":result["symbol"],"score":result["score"],"decision":result["decision"],"ltp":result["ltp"],"rsi":rsi,"change_pct":change_pct,"volume_spike_pct":round(volume_spike,2) if volume_spike is not None else None}

@app.post("/api/alerts/evaluate")
def api_alerts_evaluate(request: Request):
    session_id=request.headers.get("X-StockMeter-Session","").strip(); get_session(request); results=[]
    for alert_id,alert in list(alerts_store.items()):
        if alert.get("session_id")!=session_id: continue
        try:
            payload=AlertRequest(**{k:v for k,v in alert.items() if k in AlertRequest.model_fields})
            check=api_alert_check(payload,request)
            if check.get("triggered"): alert["last_triggered_at"]=datetime.now(timezone.utc).isoformat()
            results.append({"id":alert_id,"alert":alert,"result":check})
        except Exception as e: results.append({"id":alert_id,"alert":alert,"error":str(e)})
    return {"items":results,"updated_at":datetime.now(timezone.utc).isoformat()}


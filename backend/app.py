import os
import math
from pathlib import Path

import pandas as pd
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

app = FastAPI(title="StockScore Live")

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
# HELPER
# =========================================================

def num(v):
    try:
        v = float(v)

        if math.isnan(v) or math.isinf(v):
            return None

        return v

    except Exception:
        return None


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
# FIND NSE STOCK
# =========================================================

def find_stock(symbol):

    r = neo.search_scrip(
        exchange_segment="nse_cm",
        symbol=symbol.upper(),
        expiry="",
        option_type="",
        strike_price=""
    )

    if not r:
        return None

    # Exact equity match
    for x in r:

        if (
            x.get("pExchSeg") == "nse_cm"
            and str(x.get("pTrdSymbol", "")).upper().endswith("-EQ")
            and str(x.get("pSymbolName", "")).upper() == symbol.upper()
        ):
            return x

    # Fallback equity
    for x in r:

        if (
            x.get("pExchSeg") == "nse_cm"
            and str(x.get("pTrdSymbol", "")).upper().endswith("-EQ")
        ):
            return x

    return None


# =========================================================
# RSI
# =========================================================

def rsi(close, n=14):

    d = close.diff()

    gain = d.clip(lower=0).rolling(n).mean()

    loss = (-d.clip(upper=0)).rolling(n).mean()

    rs = gain / loss.replace(0, pd.NA)

    result = 100 - (100 / (1 + rs))

    return num(result.iloc[-1])


# =========================================================
# TECHNICAL SCORE
#
# TOTAL = 30
#
# Price > SMA50       = 6
# Price > SMA200      = 6
# SMA50 > SMA200      = 6
# RSI                 = 6
# 20D Momentum        = 6
# =========================================================

def technical(hist):

    if hist is None or hist.empty:
        return 0, {}

    try:

        c = hist["Close"].dropna()

    except Exception:

        return 0, {}

    if len(c) < 210:

        return 0, {
            "error": "Not enough historical data"
        }

    last = float(c.iloc[-1])

    sma50 = float(
        c.rolling(50).mean().iloc[-1]
    )

    sma200 = float(
        c.rolling(200).mean().iloc[-1]
    )

    rv = rsi(c)

    # 20 trading day momentum
    momentum_20d = None

    if len(c) >= 21:

        old_price = float(c.iloc[-21])

        if old_price != 0:

            momentum_20d = (
                (last / old_price) - 1
            ) * 100

    pts = 0

    # -----------------------------------------------------
    # 1. PRICE ABOVE SMA50
    # -----------------------------------------------------

    if last > sma50:
        pts += 6

    # -----------------------------------------------------
    # 2. PRICE ABOVE SMA200
    # -----------------------------------------------------

    if last > sma200:
        pts += 6

    # -----------------------------------------------------
    # 3. SMA50 ABOVE SMA200
    # -----------------------------------------------------

    if sma50 > sma200:
        pts += 6

    # -----------------------------------------------------
    # 4. RSI
    # -----------------------------------------------------

    rsi_points = 0

    if rv is not None:

        if 55 <= rv <= 70:
            rsi_points = 6

        elif 50 <= rv < 55:
            rsi_points = 4

        elif 40 <= rv < 50:
            rsi_points = 2

        elif rv > 70:
            # Overbought
            rsi_points = 3

        else:
            rsi_points = 0

    pts += rsi_points

    # -----------------------------------------------------
    # 5. MOMENTUM
    # -----------------------------------------------------

    momentum_points = 0

    if momentum_20d is not None:

        if momentum_20d >= 5:
            momentum_points = 6

        elif momentum_20d >= 0:
            momentum_points = 4

        elif momentum_20d >= -5:
            momentum_points = 2

        else:
            momentum_points = 0

    pts += momentum_points

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if last > sma50 > sma200:

        trend = "Bullish"

    elif last < sma50 < sma200:

        trend = "Bearish"

    else:

        trend = "Mixed"

    return min(pts, 30), {

        "rsi": round(rv, 2)
        if rv is not None else None,

        "sma50": round(sma50, 2),

        "sma200": round(sma200, 2),

        "momentum_20d_pct": round(momentum_20d, 2)
        if momentum_20d is not None else None,

        "trend": trend

    }


# =========================================================
# FUNDAMENTAL SCORE
#
# TOTAL = 40
#
# PE              = 6
# ROE             = 7
# Operating Margin= 6
# Revenue Growth  = 6
# Earnings Growth = 7
# Debt/Equity     = 8
# =========================================================

def fundamentals(info):

    pts = 0

    pe = num(info.get("trailingPE"))

    roe = num(info.get("returnOnEquity"))

    margin = num(info.get("operatingMargins"))

    rg = num(info.get("revenueGrowth"))

    eg = num(info.get("earningsGrowth"))

    de = num(info.get("debtToEquity"))

    # -----------------------------------------------------
    # PE
    # -----------------------------------------------------

    if pe is not None:

        if pe <= 20:
            pts += 6

        elif pe <= 30:
            pts += 5

        elif pe <= 40:
            pts += 3

        elif pe <= 60:
            pts += 1

    # -----------------------------------------------------
    # ROE
    # -----------------------------------------------------

    if roe is not None:

        if roe >= 0.20:
            pts += 7

        elif roe >= 0.15:
            pts += 6

        elif roe >= 0.12:
            pts += 4

        elif roe > 0:
            pts += 2

    # -----------------------------------------------------
    # OPERATING MARGIN
    # -----------------------------------------------------

    if margin is not None:

        if margin >= 0.20:
            pts += 6

        elif margin >= 0.15:
            pts += 5

        elif margin >= 0.08:
            pts += 3

        elif margin > 0:
            pts += 1

    # -----------------------------------------------------
    # REVENUE GROWTH
    # -----------------------------------------------------

    if rg is not None:

        if rg >= 0.15:
            pts += 6

        elif rg >= 0.10:
            pts += 5

        elif rg >= 0.05:
            pts += 3

        elif rg > 0:
            pts += 1

    # -----------------------------------------------------
    # EARNINGS GROWTH
    # -----------------------------------------------------

    if eg is not None:

        if eg >= 0.20:
            pts += 7

        elif eg >= 0.15:
            pts += 6

        elif eg >= 0.05:
            pts += 3

        elif eg > 0:
            pts += 1

    # -----------------------------------------------------
    # DEBT TO EQUITY
    # -----------------------------------------------------

    if de is not None:

        if de <= 30:
            pts += 8

        elif de <= 60:
            pts += 6

        elif de <= 100:
            pts += 4

        elif de <= 200:
            pts += 2

    return min(pts, 40), {

        "pe": pe,

        "roe_pct": round(roe * 100, 2)
        if roe is not None else None,

        "operating_margin_pct": round(margin * 100, 2)
        if margin is not None else None,

        "revenue_growth_pct": round(rg * 100, 2)
        if rg is not None else None,

        "earnings_growth_pct": round(eg * 100, 2)
        if eg is not None else None,

        "debt_to_equity": de

    }


# =========================================================
# VALUATION + RISK SCORE
#
# TOTAL = 30
#
# PEG                  = 8
# Price / Book         = 6
# Free Cash Flow       = 6
# Profit Margin        = 5
# Current Ratio        = 5
# =========================================================

def valuation_risk(info):

    pts = 0

    peg = num(info.get("pegRatio"))

    pb = num(info.get("priceToBook"))

    fcf = num(info.get("freeCashflow"))

    profit_margin = num(info.get("profitMargins"))

    current_ratio = num(info.get("currentRatio"))

    # -----------------------------------------------------
    # PEG
    # -----------------------------------------------------

    if peg is not None and peg > 0:

        if peg <= 1:
            pts += 8

        elif peg <= 1.5:
            pts += 6

        elif peg <= 2:
            pts += 4

        elif peg <= 3:
            pts += 2

    # -----------------------------------------------------
    # PRICE / BOOK
    # -----------------------------------------------------

    if pb is not None and pb > 0:

        if pb <= 2:
            pts += 6

        elif pb <= 4:
            pts += 4

        elif pb <= 6:
            pts += 2

    # -----------------------------------------------------
    # FREE CASH FLOW
    # -----------------------------------------------------

    if fcf is not None:

        if fcf > 0:
            pts += 6

        else:
            pts += 0

    # -----------------------------------------------------
    # PROFIT MARGIN
    # -----------------------------------------------------

    if profit_margin is not None:

        if profit_margin >= 0.20:
            pts += 5

        elif profit_margin >= 0.10:
            pts += 4

        elif profit_margin > 0:
            pts += 2

    # -----------------------------------------------------
    # CURRENT RATIO
    # -----------------------------------------------------

    if current_ratio is not None:

        if current_ratio >= 1.5:
            pts += 5

        elif current_ratio >= 1:
            pts += 3

        elif current_ratio > 0.75:
            pts += 1

    return min(pts, 30), {

        "peg_ratio": peg,

        "price_to_book": pb,

        "free_cashflow": fcf,

        "profit_margin_pct": round(
            profit_margin * 100, 2
        ) if profit_margin is not None else None,

        "current_ratio": current_ratio

    }


# =========================================================
# DECISION
# =========================================================

def get_decision(total):

    if total >= 75:

        return "STRONG BUY"

    elif total >= 60:

        return "BUY"

    elif total >= 45:

        return "HOLD"

    else:

        return "AVOID"


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    frontend = (
        Path(__file__).parent
        / "../frontend/index.html"
    ).resolve()

    return FileResponse(str(frontend))


# =========================================================
# LOGIN API
# =========================================================

@app.post("/api/login")
def api_login(x: LoginRequest):

    return login_neo(x.totp)


# =========================================================
# STOCK SCORE API
# =========================================================

@app.post("/api/score")
def api_score(x: StockRequest):

    if not logged_in or neo is None:

        raise HTTPException(
            status_code=401,
            detail="Login to Kotak Neo first."
        )

    symbol = x.company.strip().upper()

    # -----------------------------------------------------
    # FIND STOCK
    # -----------------------------------------------------

    stock = find_stock(symbol)

    if not stock:

        raise HTTPException(
            status_code=404,
            detail=f"NSE equity not found for '{symbol}'."
        )

    # -----------------------------------------------------
    # LIVE LTP
    # -----------------------------------------------------

    try:

        q = neo.quotes(
            instrument_tokens=[
                {
                    "instrument_token":
                        str(stock["pSymbol"]),

                    "exchange_segment":
                        "nse_cm"
                }
            ],
            quote_type="ltp"
        )

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=f"Kotak quote error: {str(e)}"
        )

    if not q:

        raise HTTPException(
            status_code=502,
            detail="No live quote returned."
        )

    try:

        ltp = float(q[0]["ltp"])

    except Exception:

        raise HTTPException(
            status_code=502,
            detail="Invalid LTP returned by Kotak Neo."
        )

    # -----------------------------------------------------
    # YAHOO FINANCE
    # -----------------------------------------------------

    try:

        t = yf.Ticker(symbol + ".NS")

        info = t.info or {}

        hist = t.history(
            period="1y",
            auto_adjust=False
        )

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=f"Yahoo Finance data error: {str(e)}"
        )

    # -----------------------------------------------------
    # CALCULATE SCORES
    # -----------------------------------------------------

    tp, td = technical(hist)

    fp, fd = fundamentals(info)

    vp, vd = valuation_risk(info)

    # =====================================================
    # FINAL SCORE
    # =====================================================

    total = fp + tp + vp

    total = max(0, min(100, total))

    decision = get_decision(total)

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "company":
            stock.get("pSymbolName"),

        "symbol":
            stock.get("pTrdSymbol"),

        "token":
            str(stock.get("pSymbol")),

        "ltp":
            ltp,

        "score":
            total,

        "score_model":
            "100-point model: Fundamental 40 + Technical 30 + Valuation/Risk 30",

        "decision":
            decision,

        "fundamental": {

            "score":
                fp,

            "out_of":
                40,

            **fd
        },

        "technical": {

            "score":
                tp,

            "out_of":
                30,

            **td
        },

        "valuation_risk": {

            "score":
                vp,

            "out_of":
                30,

            **vd
        },

        "data_note":
            "Live LTP: Kotak Neo. "
            "Fundamentals/history: Yahoo Finance; "
            "may be delayed or incomplete.",

        "disclaimer":
            "Decision-support only; not a guaranteed "
            "prediction or personalized investment advice."

    }

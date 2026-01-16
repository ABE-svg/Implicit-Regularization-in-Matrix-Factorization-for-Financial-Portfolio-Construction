from dataclasses import dataclass

@dataclass
class Config:
    tickers = (
    # ===== Broad Market ETFs =====
    "SPY","QQQ","IWM","DIA","VTI","VOO","IVV","VT","VEA","VWO",
    "EFA","EWJ","EWG","EWU","EWC","EWH","EWT","EWZ","INDA","FXI",

    # ===== Bond & Defensive ETFs =====
    "TLT","IEF","SHY","BND","AGG","LQD","HYG","TIP","MUB",
    "VGIT","VCSH","VCIT","BSV","JPST","ICSH","GLDM","SLV","USO",

    # ===== Sector ETFs =====
    "XLF","XLK","XLE","XLY","XLP","XLV","XLI","XLB","XLU","XLRE",
    "SMH","SOXX","KRE","XBI","IBB","ITA","IYR","XRT","FDN","ARKK",

    # ===== Mega-cap Tech =====
    "MSFT","AMZN","GOOGL","GOOG","META","NVDA","AVGO","ORCL",
    "ADBE","CRM","INTC","AMD","CSCO","QCOM","TXN","IBM","NOW","SNOW",

    # ===== Financials =====
    "JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP",
    "PNC","TFC","COF","BK","STT","ALL","PRU","TRV",

    # ===== Healthcare =====
    "JNJ","UNH","PFE","MRK","ABBV","LLY","BMY","AMGN","GILD","CVS",
    "HUM","CI","MDT","SYK","BDX","ISRG","VRTX","REGN","ZTS","BIIB",

    # ===== Consumer & Retail =====
    "WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE","DIS","NFLX",
    "KO","PEP","MO","CL","PG","EL","KMB","GIS","KHC",

    # ===== Industrials =====
    "CAT","DE","BA","GE","HON","LMT","RTX","UPS","FDX",
    "CSX","NSC","MMM","EMR","ETN","ITW","PH","ROK","CMI","PCAR",

    # ===== Energy & Materials =====
    "XOM","CVX","COP","SLB","EOG","VLO","OXY","KMI",
    "BHP","RIO","FCX","NEM","AA","APD","SHW","DOW","DD"
    )


    start = "2019-01-01"
    end = "2024-01-01"
    test_size = 252
    T_train = 252
    T_test  = 63
    step    = 63         


    mask_frac = 0.30
    include_diag = True
    seed = 42

    lr = 1e-3
    n_steps = 300        
    log_every = 50

    init_scale_tiny = 1e-4
    init_scale_big = 1e-1

    m_portfolios = None   # set in code as 2*N or 10*N
    w_kind = "l2"         # or "simplex"

    ridge = 0.0   # main experiment: no explicit regularization on weights


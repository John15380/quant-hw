import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from numba import jit
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from datetime import timedelta
from joblib import Parallel, delayed
import gc
import shutil
from tqdm import tqdm

# 设置绘图风格
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
class Config:
    # --- 路径配置 ---
    ROOT_DIR = './data'
    PATH_PLOTS = './plots'
    
    # 基础数据路径
    PATH_HF_BAR = os.path.join(ROOT_DIR, 'bar1m_2016_2018') 
    PATH_DAILY_QUOTE = os.path.join(ROOT_DIR, '日行情') 
    PATH_TRADING_DAYS = os.path.join(ROOT_DIR, '交易日期', 'tradingdays.csv')
    PATH_INDUSTRY = os.path.join(ROOT_DIR, '股票行业分类', 'industry.csv')
    PATH_CAP = os.path.join(ROOT_DIR, '股票市值', 'stock_cap.parquet')
    
    # 指数成分
    PATH_INDEX_WT_300 = os.path.join(ROOT_DIR, '指数权重', 'index_wt_000300.parquet')
    PATH_INDEX_WT_500 = os.path.join(ROOT_DIR, '指数权重', 'index_wt_000905.parquet')
    
    # --- 运算配置 ---
    N_JOBS = 1            # 顺序处理更高效
    SMART_MONEY_WINDOW = 10 
    ROLLING_WINDOW = 12   # 训练窗口(月)
    START_PREDICT = '2017-01-01'
    
    # --- 交易参数 ---
    TOP_K = 50            # 持仓数量
    COST_RATE_TOTAL = 0.003  # 双边千三
    MAX_GAP_DAYS = 20     # 超过20天无数据视为断层，清空历史

    TEST_PERIODS = {
        'Weekly': 5,
        'BiWeekly': 10,
        'Monthly': 20
    }

# -----------------------------------------------------------------------------
# 2. 核心计算函数
# -----------------------------------------------------------------------------
@jit(nopython=True, cache=True)
def calc_daily_smart_money(open_arr, close_arr, vol_arr, amt_arr):
    n = len(close_arr)
    if n == 0: return np.nan
    
    # 1. 计算收益率和 S 指标
    s_metric = np.zeros(n)
    for i in range(n):

        # 在循环内部，vol_arr 为 0 的处理必须放在前面，否则会导致除以零的错误
        if vol_arr[i] < 1e-6:
            s_metric[i] = -1.0 # 确保排在最后
            continue

        # 收益率计算保持原样
        if i == 0:
            ret = close_arr[i] / open_arr[i] - 1.0 if open_arr[i] > 1e-6 else 0.0
        else:
            ret = close_arr[i] / close_arr[i-1] - 1.0 if close_arr[i-1] > 1e-6 else 0.0
            
        safe_vol = vol_arr[i] if vol_arr[i] > 1e-9 else 0.0
        s_metric[i] = np.abs(ret) / (np.power(safe_vol, 0.25) + 1e-9)
            
    # 2. 排序
    sorted_idx = np.argsort(s_metric)[::-1]
    total_vol = np.sum(vol_arr)
    if total_vol < 1e-6: return np.nan
    
    # 3. 截取前 20% (含插值)
    target_vol = total_vol * 0.20
    current_vol = 0.0
    smart_amt = 0.0
    smart_vol = 0.0 # 理论上应严格等于 target_vol
    
    for idx in sorted_idx:
        v = vol_arr[idx]
        if v <= 0: continue
        
        # 检查是否加上这个 bar 会超过 20%
        if current_vol + v > target_vol:
            remain_vol = target_vol - current_vol
            ratio = remain_vol / v
            
            smart_vol += remain_vol
            smart_amt += amt_arr[idx] * ratio
            current_vol += remain_vol # 达到 target，结束
            break
        else:
            smart_vol += v
            smart_amt += amt_arr[idx]
            current_vol += v
            
    if smart_vol < 1e-6: return np.nan
    
    # 4. 计算 VWAP 比例
    vwap_smart = smart_amt / smart_vol
    vwap_all = np.sum(amt_arr) / total_vol
    
    if vwap_all < 1e-6: return np.nan
    
    return vwap_smart / vwap_all

@jit(nopython=True, cache=True)
def calc_linear_slope(y_arr):
    """
    计算时间序列 y 对时间 t (0, 1, 2...) 的线性回归斜率
    用于 CPV Trend 计算
    """
    n = len(y_arr)
    if n < 3: return 0.0
    
    x_sum = 0.0
    y_sum = 0.0
    xy_sum = 0.0
    xx_sum = 0.0
    
    count = 0
    for i in range(n):
        v = y_arr[i]
        if np.isnan(v): continue # 跳过 NaN
        
        x = float(i)
        x_sum += x
        y_sum += v
        xy_sum += x * v
        xx_sum += x * x
        count += 1
        
    if count < 3: return 0.0
    
    x_mean = x_sum / count
    y_mean = y_sum / count
    
    numerator = xy_sum - count * x_mean * y_mean
    denominator = xx_sum - count * x_mean * x_mean
    
    if np.abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator

@jit(nopython=True, cache=True)
def calc_subset_correlation(x_arr, y_arr, mode):
    """
    计算特定象限子序列的相关系数
    mode 1: x>0, y>0 (Up-Up)
    mode 2: x<0, y<0 (Down-Down)
    """
    n = len(x_arr)
    if n < 2: return 0.0
    
    sum_x, sum_y = 0.0, 0.0
    sum_xx, sum_yy = 0.0, 0.0
    sum_xy = 0.0
    count = 0
    
    for i in range(n):
        vx = x_arr[i]
        vy = y_arr[i]
        
        # 筛选逻辑
        valid = False
        if mode == 1:
            if vx > 1e-9 and vy > 1e-9: valid = True
        elif mode == 2:
            if vx < -1e-9 and vy < -1e-9: valid = True
            
        if valid:
            sum_x += vx
            sum_y += vy
            sum_xx += vx * vx
            sum_yy += vy * vy
            sum_xy += vx * vy
            count += 1
            
    if count < 3: return 0.0 # 样本过少无法计算相关系数
    
    mean_x = sum_x / count
    mean_y = sum_y / count
    
    cov = sum_xy - count * mean_x * mean_y
    var_x = sum_xx - count * mean_x * mean_x
    var_y = sum_yy - count * mean_y * mean_y

    # 防止精度误差导致 var < 0
    if var_x < 0: var_x = 0
    if var_y < 0: var_y = 0
    
    if var_x > 1e-12 and var_y > 1e-12:
        return cov / np.sqrt(var_x * var_y)
    return 0.0

@jit(nopython=True, cache=True)
def calc_daily_metrics_intraday(open_arr, close_arr, high_arr, low_arr, vol_arr):
    """
    Level 1: 计算基于分钟数据的当日基础指标 (Updated: Skewness & Amplitude)
    Returns: [SM, CPV, CDPDP, High, Low, Close, Ret, Skew, TailRet, Amp] -> Size 10
    """
    n = len(close_arr)
    if n < 30: return np.array([np.nan] * 10)
    
    # --- 1. Smart Money ---
    s_metric = np.zeros(n)
    valid_sm_count = 0
    for i in range(1, n):
        v = vol_arr[i]
        if v < 1e-6:
            s_metric[i] = -1.0 
            continue
        if close_arr[i-1] > 1e-6:
            r = np.abs(close_arr[i] / close_arr[i-1] - 1.0)
            s_metric[i] = r / (np.power(v, 0.25))
            valid_sm_count += 1
        else:
            s_metric[i] = -1.0
            
    sm_factor = np.nan
    if valid_sm_count > 10:
        valid_indices = np.where(s_metric >= 0)[0]
        if len(valid_indices) > 0:
            s_valid = s_metric[valid_indices]
            sorted_args = np.argsort(s_valid)[::-1]
            sorted_real_idx = valid_indices[sorted_args]
            total_vol = np.sum(vol_arr)
            if total_vol > 1e-6:
                target_vol = total_vol * 0.20
                curr_vol = 0.0
                smart_amt = 0.0
                smart_vol = 0.0
                avg_price = close_arr 
                amt_arr = avg_price * vol_arr
                for idx in sorted_real_idx:
                    v = vol_arr[idx]
                    if curr_vol + v > target_vol:
                        remain = target_vol - curr_vol
                        ratio = remain / v
                        smart_vol += remain
                        smart_amt += amt_arr[idx] * ratio
                        break
                    else:
                        smart_vol += v
                        smart_amt += amt_arr[idx]
                        curr_vol += v
                if smart_vol > 1e-6:
                    vwap_smart = smart_amt / smart_vol
                    vwap_all = np.sum(amt_arr) / total_vol
                    if vwap_all > 1e-6:
                        sm_factor = vwap_smart / vwap_all

    # --- 2. CPV Component ---
    mean_c = np.mean(close_arr)
    mean_v = np.mean(vol_arr)
    num = 0.0
    den_c = 0.0
    den_v = 0.0
    for i in range(n):
        dc = close_arr[i] - mean_c
        dv = vol_arr[i] - mean_v
        num += dc * dv
        den_c += dc**2
        den_v += dv**2
    corr_pv = 0.0
    if den_c > 1e-9 and den_v > 1e-9:
        corr_pv = num / np.sqrt(den_c * den_v)
        
    # --- 3. CDPDP ---
    diffs = close_arr[1:] - close_arr[:-1]
    if len(diffs) > 10:
        x_seq = diffs[:-1]
        y_seq = diffs[1:]
        corr_up_up = calc_subset_correlation(x_seq, y_seq, 1)
        corr_down_down = calc_subset_correlation(x_seq, y_seq, 2)
        cdpdp_raw = corr_up_up + corr_down_down
    else:
        cdpdp_raw = np.nan
    
    # --- 4. Daily Basic Stats (保持不变) ---
    day_high = np.max(high_arr)
    day_low = np.min(low_arr)
    day_close = close_arr[-1]
    if open_arr[0] > 1e-6:
        day_ret_intraday = day_close / open_arr[0] - 1.0 
    else:
        day_ret_intraday = 0.0

    # ==========================================
    # --- 5. New High-Freq Factors ---
    # ==========================================

    # A. Skewness (偏度) - 替换了 Vol
    # 计算公式: E[(x-u)^3] / std^3
    val_skew = np.nan
    if n > 10:
        # 1. 计算分钟收益率
        rets = np.zeros(n-1)
        count = 0
        sum_r = 0.0
        for i in range(1, n):
            if close_arr[i-1] > 1e-6:
                r = close_arr[i] / close_arr[i-1] - 1.0
                rets[i-1] = r
                sum_r += r
                count += 1
        
        if count > 10:
            mean_r = sum_r / count
            sum_sq = 0.0
            sum_cu = 0.0
            
            for i in range(n-1):
                if rets[i] == 0 and i > count: continue # Skip padding if any
                diff = rets[i] - mean_r
                sum_sq += diff * diff
                sum_cu += diff * diff * diff
            
            var_r = sum_sq / count
            if var_r > 1e-12:
                std_r = np.sqrt(var_r)
                m3 = sum_cu / count
                val_skew = m3 / (std_r * std_r * std_r)
            else:
                val_skew = 0.0

    # B. Tail Return (尾盘收益) - 保留
    val_tail = np.nan
    lookback = 30
    if n >= lookback:
        start_price = close_arr[-lookback]
        end_price = close_arr[-1]
        if start_price > 1e-6:
            val_tail = end_price / start_price - 1.0
    else:
        start_idx = int(n * 0.75)
        start_price = close_arr[start_idx]
        end_price = close_arr[-1]
        if start_price > 1e-6:
            val_tail = end_price / start_price - 1.0

    # C. Amplitude (振幅) - 替换了 Efficiency
    # (High - Low) / Open
    val_amp = np.nan
    if open_arr[0] > 1e-6:
        val_amp = (day_high - day_low) / open_arr[0]

    # Return size: 10
    return np.array([
        sm_factor,      # 0
        corr_pv,        # 1
        cdpdp_raw,      # 2
        day_high,       # 3
        day_low,        # 4
        day_close,      # 5
        day_ret_intraday, # 6
        val_skew,       # 7 (New: Skew)
        val_tail,       # 8 (Old: Tail)
        val_amp         # 9 (New: Amp)
    ])

# -----------------------------------------------------------------------------
# 3. 数据加载器
# -----------------------------------------------------------------------------
class DataLoader:
    @staticmethod
    def load_universe(index_path):
        # 保持原样，无需修改
        print(f">>> [Loader] 加载股票池: {index_path} ...")
        df = pd.read_parquet(index_path)
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'symbol']].drop_duplicates()

    @staticmethod
    def load_daily_quote():
        # 保持原样，无需修改
        print(">>> [Loader] 加载日行情 ...")
        files = glob.glob(os.path.join(Config.PATH_DAILY_QUOTE, "*.csv"))
        dfs = []
        cols = [
            'SecuCode', 'TradingDay', 'OpenPrice', 'ClosePrice', 'PTFlag',
            'TradingVolumes', 'TurnoverValue', 'NonRestrictedShares', 'AFloats', 
        ] 
        for f in tqdm(files, desc="[Loader] Reading CSVs"):
            try:
                temp = pd.read_csv(f, usecols=lambda c: c in cols, dtype={'SecuCode': str})
                dfs.append(temp)
            except ValueError:
                print(f"Warning: File {f} format mismatch, skipping.")
        
        if not dfs:
            raise ValueError("No daily quote files found!")

        df = pd.concat(dfs)
        df['TradingDay'] = pd.to_datetime(df['TradingDay'])
        df['SecuCode'] = df['SecuCode'].str.zfill(6)
        
        df.rename(columns={
            'TradingVolumes': 'Volume', 
            'TurnoverValue': 'Amount'
        }, inplace=True)
        
        df = df.sort_values(['SecuCode', 'TradingDay'])
        return df

    @staticmethod
    def load_industry_cap():
        # 保持原样，无需修改
        print(">>> [Loader] 加载行业与市值 ...")
        df_ind = pd.read_csv(Config.PATH_INDUSTRY, dtype={'SecuCode': str})
        df_ind['InfoPublDate'] = pd.to_datetime(df_ind['InfoPublDate'])
        df_ind['InfoPublDate'] = df_ind['InfoPublDate'] + pd.Timedelta(days=1)
        df_ind['SecuCode'] = df_ind['SecuCode'].str.zfill(6)
        df_ind = df_ind.sort_values('InfoPublDate')
        
        df_cap = pd.read_parquet(Config.PATH_CAP)
        df_cap.rename(columns={'date': 'TradingDay', 'symbol': 'SecuCode', 'tot_cap': 'Cap'}, inplace=True)
        df_cap['TradingDay'] = pd.to_datetime(df_cap['TradingDay'])
        df_cap['SecuCode'] = df_cap['SecuCode'].astype(str).str.zfill(6)
        df_cap['Cap'] = df_cap['Cap'].astype(float)
        df_cap['LnCap'] = np.log(df_cap['Cap'] + 1)
        return df_ind, df_cap

    @staticmethod
    def construct_label(df_daily, is_for_backtest=False):
        """
        修正版：
        1. Reindex 修复停牌时间穿越。
        2. 严格剔除 T+1 一字涨停和停牌样本，防止模型作弊。
        """
        print(f">>> [Loader] 构造 Labels (Strict Limit Check: {'Backtest' if is_for_backtest else 'Train'})...")
        
        # --- 1. 加载全量交易日历 (防止时间断层) ---
        df_cal = pd.read_csv(Config.PATH_TRADING_DAYS)
        df_cal['TradingDay'] = pd.to_datetime(df_cal['TradingDay'])
        
        min_date = df_daily['TradingDay'].min()
        max_date = df_daily['TradingDay'].max()
        all_trading_days = df_cal[(df_cal['TradingDay'] >= min_date) & 
                                  (df_cal['TradingDay'] <= max_date)]['TradingDay'].sort_values().values
        
        # --- 2. Reindex 全市场索引 ---
        all_codes = df_daily['SecuCode'].unique()
        idx = pd.MultiIndex.from_product([all_codes, all_trading_days], names=['SecuCode', 'TradingDay'])
        
        # 停牌日或缺失日会自动填充 NaN
        df_full = df_daily.set_index(['SecuCode', 'TradingDay']).reindex(idx)
        
        # --- 3. 计算 Shift (Label) ---
        g = df_full.groupby('SecuCode')
        
        # T+1 的状态
        df_full['Next_Open'] = g['OpenPrice'].shift(-1)
        df_full['Next_PTFlag'] = g['PTFlag'].shift(-1) # T+1 的停牌状态
        
        # T+2 的状态 (用于计算持有收益)
        df_full['Next_Next_Open'] = g['OpenPrice'].shift(-2)
        
        # 真正的结算日期 (T+2 日期)
        df_full['Temp_Date'] = df_full.index.get_level_values('TradingDay')
        df_full['SettlementDate'] = df_full.groupby('SecuCode')['Temp_Date'].shift(-2)
        df_full.drop(columns=['Temp_Date'], inplace=True)

        # 重置索引，方便向量化计算
        df = df_full.reset_index()
        
        # --- 4. 严格的涨跌停阈值计算 ---
        # 目的：判断 T+1 开盘是否一字涨停 (无法买入)
        # 逻辑：
        #   - 科创板 (688): 20%
        #   - 北交所 (8xx/4xx): 30%
        #   - 创业板 (300): 2020-08-24 前 10%, 后 20%
        #   - 其他 (主板): 10% (ST暂统一按10%处理或需额外ST列表，此处偏向保守取 9.8% 左右作为阈值)
        
        # 提取 Code 前缀和日期
        codes_series = df['SecuCode'].astype(str)
        dates = df['TradingDay'].values
        
        # 构造条件掩码 (使用 Pandas 原生向量化操作，会自动处理底层类型)
        # .values 将 Series 转为 numpy boolean array，性能无损
        is_688 = codes_series.str.startswith('688').values
        is_bj = (codes_series.str.startswith('8') | codes_series.str.startswith('4')).values
        is_300 = codes_series.str.startswith('300').values
        
        date_threshold = np.datetime64('2020-08-24')
        is_after_reform = dates >= date_threshold
        
        # 向量化选择阈值 (使用稍微收紧的阈值以防精度误差)
        limit_thresholds = np.select(
            [
                is_bj,                            # 北交所 30%
                is_688,                           # 科创板 20%
                (is_300 & is_after_reform),       # 创业板注册制后 20%
                (is_300 & ~is_after_reform)       # 创业板注册制前 10%
            ],
            [0.29, 0.195, 0.195, 0.095],          # 对应的值
            default=0.095                         # 主板默认 10%
        )
        
        # 存储阈值供后续判断
        df['Limit_Threshold'] = limit_thresholds
        
        # --- 5. 计算收益与 Label 清洗 ---
        # T+1 开盘涨幅 (相对于 T 日收盘价)
        # 注意：这里是用 T+1 Open / T Close - 1 来判断是否在 T+1 开盘时就涨停
        df['Next_Open_Chg'] = df['Next_Open'] / df['ClosePrice'] - 1.0
        
        # 计算持有收益 (T+1 Open 到 T+2 Open)
        df['Ret_Hold'] = df['Next_Next_Open'] / df['Next_Open'] - 1.0
        
        if not is_for_backtest:
            # === [训练模式 Training Mode] ===
            # 必须剔除模型无法交易的样本，防止 Label Leakage
            
            # 条件A: T+1 开盘一字涨停 (买不进去)
            cond_limit_up = df['Next_Open_Chg'] >= df['Limit_Threshold']
            
            # 条件B: T+1 停牌 (PTFlag != 0, 买不进去)
            # 注意：Reindex 后，如果当天没数据，Next_PTFlag 也是 NaN，这也会被下面的 notna() 过滤掉
            # 这里假设 PTFlag=0 是正常，其他是异常
            cond_suspend = (df['Next_PTFlag'] != 0) | df['Next_PTFlag'].isna()
            
            # 将这些“不可交易”样本的收益设为 NaN，从而在 dropna 时剔除
            mask_invalid = cond_limit_up | cond_suspend
            df.loc[mask_invalid, 'Ret_Hold'] = np.nan
            
            # 截断极端值
            df['Ret_Hold'] = df['Ret_Hold'].clip(-0.2, 0.2)
            
            # 丢弃缺失值 (包含原始缺失、停牌、一字涨停)
            df = df.dropna(subset=['Ret_Hold', 'ClosePrice'])
            
        else:
            # === [回测模式 Backtest Mode] ===
            # 保留所有行（包括 NaN），以便回测引擎查询“当天是否停牌”
            pass
            
        return df

# -----------------------------------------------------------------------------
# 4. 因子引擎
# -----------------------------------------------------------------------------
class FactorEngine:
    def __init__(self, valid_universe_df, cache_dir='./temp_factors'):
        self.daily_files = sorted(glob.glob(os.path.join(Config.PATH_HF_BAR, "*.parquet")))
        self.univ_map = valid_universe_df.groupby('date')['symbol'].apply(set).to_dict()
        self.cache_dir = cache_dir
        self.batch_size = 50000 
        self.buffer = []       
        self.part_idx = 0      
        
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.MAX_HISTORY = 170
        self.history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self.MAX_GAP_DAYS = 30 

    def _flush_buffer(self):
        if not self.buffer: return
        df_chunk = pd.DataFrame(self.buffer)
        cols = df_chunk.select_dtypes(include=['float64']).columns
        df_chunk[cols] = df_chunk[cols].astype('float32')
        save_path = os.path.join(self.cache_dir, f'part_{self.part_idx:04d}.parquet')
        df_chunk.to_parquet(save_path, index=False)
        self.part_idx += 1
        self.buffer = []
        del df_chunk
        gc.collect()

    def run(self):
        print(f">>> [Factor] 开始计算高频与滚动因子 (Raw Value Version)...")
        file_iterator = tqdm(self.daily_files, desc="[Factor] Processing Days")

        for f_path in file_iterator:
            date_str = os.path.basename(f_path).split('.')[0]
            curr_date = pd.Timestamp(date_str)
            
            try:
                df_day = pd.read_parquet(f_path)
                df_day.columns = [c.lower() for c in df_day.columns]
            except: continue
                
            if curr_date not in self.univ_map: continue
            valid_symbols = self.univ_map[curr_date]
            df_day = df_day[df_day['symbol'].isin(valid_symbols)]
            if df_day.empty: continue
            
            g = df_day.groupby('symbol')
            
            for sym, sub_df in g:
                if 'time' in sub_df.columns: sub_vals = sub_df.sort_values('time')
                else: sub_vals = sub_df
                
                daily_mets = calc_daily_metrics_intraday(
                    sub_vals['open'].values, sub_vals['close'].values,
                    sub_vals['high'].values, sub_vals['low'].values,
                    sub_vals['volume'].values
                )
                
                queue = self.history[sym]
                
                if len(queue) > 0:
                    last_date, last_mets = queue[-1]
                    if (curr_date - last_date).days > self.MAX_GAP_DAYS:
                        queue.clear()
                    else:
                        prev_close = last_mets[5] 
                        curr_close = daily_mets[5]
                        if prev_close > 1e-6:
                            true_ret = curr_close / prev_close - 1.0
                            daily_mets[6] = true_ret
                        else:
                            daily_mets[6] = 0.0

                if not np.isnan(daily_mets[5]): 
                    queue.append((curr_date, daily_mets))
                
                n_hist = len(queue)
                if n_hist < 20: continue 
                
                hist_data = np.array([item[1] for item in queue]) 
                
                # --- Factor 1: Smart Money ---
                sm_vals = hist_data[-10:, 0]
                sm_vals = sm_vals[sm_vals > -0.5] 
                val_sm = np.mean(sm_vals) if len(sm_vals) > 0 else np.nan
                
                # --- Factor 2: CPV Mean ---
                val_cpv_mean = np.nan
                if n_hist >= 20: 
                    cpv_slice = hist_data[-20:, 1]
                    mask = ~np.isnan(cpv_slice)
                    if np.sum(mask) > 15:
                        val_cpv_mean = np.mean(cpv_slice[mask]) 

                # --- Factor 3: CDPDP ---
                cdpdp_slice = hist_data[-20:, 2] 
                cdpdp_slice = cdpdp_slice[~np.isnan(cdpdp_slice)]
                val_cdpdp = np.mean(cdpdp_slice) if len(cdpdp_slice) > 0 else np.nan
                
                # --- Factor 4: HF_Skewness ---
                skew_slice = hist_data[-20:, 7]
                skew_slice = skew_slice[~np.isnan(skew_slice)]
                val_skew = np.mean(skew_slice) if len(skew_slice) > 0 else np.nan

                # --- Factor 5: HF_Tail_Reversal ---
                tail_slice = hist_data[-20:, 8]
                tail_slice = tail_slice[~np.isnan(tail_slice)]
                val_tail_rev = np.mean(tail_slice) if len(tail_slice) > 0 else np.nan

                # --- Factor 6: HF_Amplitude  ---
                amp_slice = hist_data[-20:, 9]
                amp_slice = amp_slice[~np.isnan(amp_slice)]
                val_amp = np.mean(amp_slice) if len(amp_slice) > 0 else np.nan

                # 4. 写入 Buffer (注意：这里去掉了所有的 -1.0 乘法)
                self.buffer.append({
                    'TradingDay': curr_date,
                    'SecuCode': str(sym).zfill(6),
                    'HF_SmartMoney': val_sm,
                    'HF_Raw_CPV_Mean': val_cpv_mean,  # Raw
                    'HF_CDPDP': val_cdpdp,            # Raw
                    'HF_Skewness': val_skew,          # Raw
                    'HF_Tail_Reversal': val_tail_rev, # Raw
                    'HF_Amplitude': val_amp           # Raw
                })
            
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
                
            file_iterator.set_postfix(date=date_str)

        self._flush_buffer()
        print(">>> [Factor] 计算完成 (Raw Values)，正在合并...")
        try:
            df_final = pd.read_parquet(self.cache_dir)
            return df_final
        except Exception as e:
            print(f"Error merging parquet files: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# 5. 数据处理与特征工程
# -----------------------------------------------------------------------------
def process_features_and_label(df_full, temp_dir='./temp_processed_features'):
    """
    内存优化版数据预处理：
    1. 预计算控制变量 (Reversal, Volatility, Turnover) 用于 CPV 正交化。
    2. Label 行业中性化。
    3. 特征清洗与正交化 (分批处理 + 差异化回归)。
    """
    print(f">>> [Process] 执行特征工程 (Strict Neutralization Mode)...")
    
    # --- 0. 环境清理 ---
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 确保数据按时间排序，方便计算 Rolling
    df_full = df_full.sort_values(['SecuCode', 'TradingDay'])

    # --- 1. 预计算控制变量 (Control Factors) ---
    # 这些变量用于剔除 CPV 因子中的风格干扰
    print("    [1/4] 计算风格控制变量 (Reversal, Volatility, Turnover)...")
    
    # A. Reversal (Ret20): 过去20日收益
    # 使用 ClosePrice 计算
    df_full['Ctrl_Ret20'] = df_full.groupby('SecuCode')['ClosePrice'].pct_change(20)
    
    # B. Volatility (Vol20): 过去20日日收益率的标准差
    df_full['Ctrl_Vol20'] = df_full.groupby('SecuCode')['ClosePrice'].transform(
        lambda x: x.pct_change().rolling(20).std()
    )
    
    # C. Turnover (Turnover20): 过去20日平均换手率 (需要 Volume 和 Cap)
    # 如果数据中没有 Volume，则跳过此项
    has_turnover_data = False
    # 1. 映射列名（确保后续代码能找到）
    # 即使你的 CSV 叫 TradingVolumes，为了通用性，我们建议新建标准列名
    if 'TradingVolumes' in df_full.columns:
        df_full['Volume'] = df_full['TradingVolumes'] # 对应成交量(股)
    if 'TurnoverValue' in df_full.columns:
        df_full['Amount'] = df_full['TurnoverValue']  # 对应成交额

    # 2. 确定流通股本 (优先用非限售，缺失则用A股流通)
    # 注意：分母不能为 0
    if 'FloatShares' not in df_full.columns:
        if 'NonRestrictedShares' in df_full.columns:
            print("    [Info] Constructing 'FloatShares' from 'NonRestrictedShares'...")
            df_full['FloatShares'] = df_full['NonRestrictedShares']
            
            # 如果有 AFloats，用它填充非限售股本的空缺
            if 'AFloats' in df_full.columns:
                df_full['FloatShares'] = df_full['FloatShares'].fillna(df_full['AFloats'])
                
        elif 'AFloats' in df_full.columns:
            print("    [Info] Constructing 'FloatShares' from 'AFloats'...")
            df_full['FloatShares'] = df_full['AFloats']
        else:
            print("    [Error] Neither 'NonRestrictedShares' nor 'AFloats' found!")

    # 3. 计算换手率逻辑
    # 只要有 Volume 和 FloatShares 即可

    print("Debug Columns:", df_full.columns.tolist())
    
    if 'Volume' in df_full.columns and 'FloatShares' in df_full.columns:
        # 确保分母不为 0
        valid_cap = df_full['FloatShares'] > 1e-6
        
        # 初始化
        df_full['Turnover_Daily'] = np.nan
        
        # 计算：换手率 = 成交量 / 流通股本
        df_full.loc[valid_cap, 'Turnover_Daily'] = (
            df_full.loc[valid_cap, 'Volume'] / df_full.loc[valid_cap, 'FloatShares']
        )
        
        # 极值处理：换手率超过 200% 或小于 0 的通常是数据错误，可以清洗掉
        df_full.loc[df_full['Turnover_Daily'] < 0, 'Turnover_Daily'] = 0.0
        df_full.loc[df_full['Turnover_Daily'] > 2.0, 'Turnover_Daily'] = 2.0 
        
        # 4. 计算 20日平滑换手率 (用于中性化)
        # 必须按股票分组计算 Rolling
        df_full['Ctrl_Turnover20'] = df_full.groupby('SecuCode')['Turnover_Daily'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        
        # 填充一下初始的 NaN (用当天的替代，避免整行数据被丢弃)
        df_full['Ctrl_Turnover20'].fillna(df_full['Turnover_Daily'], inplace=True)
        
        print("    [Info] Turnover calculated using NonRestrictedShares/AFloats.")
        
        # 标记 flag 供后续中性化模块使用
        has_turnover_data = True 

    else:
        print("    Warning: Still missing Volume or Shares columns. Turnover Check Failed.")
        has_turnover_data = False

    # 填充控制变量的 NaN (无法计算的前20天填 0，避免整行被删)
    ctrl_cols = ['Ctrl_Ret20', 'Ctrl_Vol20']
    if has_turnover_data: ctrl_cols.append('Ctrl_Turnover20')
    df_full[ctrl_cols] = df_full[ctrl_cols].fillna(0)

    # --- 2. Label 行业中性化 ---
    if 'Industry' not in df_full.columns:
        df_full['Industry'] = 'Unknown'
    
    print("    [2/4] 计算 Label 行业相对收益...")
    ind_means = df_full.groupby(['TradingDay', 'Industry'])['Ret_Hold'].transform('mean')
    df_full['Ret_Excess'] = df_full['Ret_Hold'] - ind_means
    df_full['Ret_Excess'] = df_full['Ret_Excess'].fillna(0)
    df_full['Ret_Excess'] = df_full.groupby('TradingDay')['Ret_Excess'].transform(lambda x: x - x.mean())

    # --- 3. 特征清洗与差异化中性化 (分批处理) ---
    feature_cols = [c for c in df_full.columns if c.startswith('HF_')]
    cpv_cols = [c for c in feature_cols if 'CPV' in c] # 识别 CPV 因子
    normal_cols = [c for c in feature_cols if 'CPV' not in c]

    def _process_day_logic(sub_df):
        # 1. 剔除无效样本
        sub_df = sub_df.dropna(subset=feature_cols, how='all')
        if len(sub_df) < 20: return pd.DataFrame() 
        
        # 2. 缺失值填充 (中位数)
        for col in feature_cols:
            median_val = sub_df[col].median()
            sub_df[col] = sub_df[col].fillna(0 if pd.isna(median_val) else median_val)
        
        # 3. 构造回归矩阵
        # 基础 X: 行业 + 市值
        X_ind = pd.get_dummies(sub_df['Industry'], prefix='Ind')
        if X_ind.shape[1] > 0:
            X_base = pd.concat([X_ind, sub_df[['LnCap']]], axis=1).astype(float)
        else:
            X_base = sub_df[['LnCap']].astype(float)
            
        # 扩展 X (用于 CPV): 行业 + 市值 + 反转 + 波动 + 换手
        # 注意要对齐索引
        X_extended = pd.concat([X_base, sub_df[ctrl_cols]], axis=1).astype(float)
        
        X_base = X_base.fillna(0)
        X_extended = X_extended.fillna(0)
            
        model = LinearRegression()
        
        # 4. 逐个因子处理
        for col in feature_cols:
            y = sub_df[col].values
            
            # MAD 去极值
            median = np.median(y)
            diff = np.abs(y - median)
            mad = np.median(diff)
            mad = np.mean(diff) if mad == 0 else mad
            y = np.clip(y, median - 5 * mad, median + 5 * mad)
            
            # --- 差异化中性化 ---
            if col in cpv_cols:
                # CPV 因子：严格剔除所有风格
                if len(X_extended) > 0:
                    model.fit(X_extended, y)
                    y = y - model.predict(X_extended)
            else:
                # 普通因子：仅剔除行业和市值
                if len(X_base) > 0:
                    model.fit(X_base, y)
                    y = y - model.predict(X_base)
            
            # Z-Score 标准化
            std_val = np.std(y)
            sub_df[col] = 0 if std_val == 0 else (y - np.mean(y)) / (std_val + 1e-9)
                
        return sub_df

    print(f"    [3/4] 分批特征处理 (Batch -> Orthogonalization -> Disk)...")
    
    # 按月分块处理
    df_full['Month'] = df_full['TradingDay'].dt.to_period('M')
    unique_months = df_full['Month'].unique()
    
    for i, month in enumerate(tqdm(unique_months, desc="Processing Batches")):
        batch_df = df_full[df_full['Month'] == month].copy()
        
        daily_groups = [group for _, group in batch_df.groupby('TradingDay')]
        
        processed_list = Parallel(n_jobs=Config.N_JOBS)(
            delayed(_process_day_logic)(day_df) for day_df in daily_groups
        )
        
        processed_list = [res for res in processed_list if not res.empty]
        if processed_list:
            batch_result = pd.concat(processed_list)
            
            # ---【修复核心 1】筛选最终列 ---
            # 确保只保留需要的列，去除中间变量
            keep_cols = ['TradingDay', 'SecuCode', 'LnCap', 'Industry', 
                         'Ret_Hold', 'Ret_Excess', 'Next_PTFlag', 'Next_Open_Chg']
            
            # 动态添加因子列 (feature_cols) 和 控制变量列 (如果存在)
            potential_cols = feature_cols + ctrl_cols + ['Volume', 'Amount', 'Turnover_Daily']
            for c in potential_cols:
                if c in batch_result.columns:
                    keep_cols.append(c)
            
            # 确保列存在
            final_cols = [c for c in keep_cols if c in batch_result.columns]
            batch_result = batch_result[final_cols]
            
            # ---强制类型转换，防止 Int/Float 推断冲突 ---
            # 排除掉 非数值列
            exclude_cols = ['TradingDay', 'SecuCode', 'Industry']
            
            # 找出所有需要转 float 的列
            cols_to_convert = [c for c in batch_result.columns if c not in exclude_cols]
            
            # 强制转为 float32 (既省内存，又避免 0 被存为 Int)
            # 使用 errors='ignore' 防止万一有奇怪的非数值列
            for col in cols_to_convert:
                try:
                    batch_result[col] = batch_result[col].astype('float32')
                except:
                    pass # 如果转换失败(比如是字符串)，就跳过

            # 保存
            save_path = os.path.join(temp_dir, f"batch_{month}.parquet")
            batch_result.to_parquet(save_path, index=False)
        
        del batch_df, daily_groups, processed_list
        gc.collect()

    print("    [4/4] 合并最终数据集...")
    try:
        df_final = pd.read_parquet(temp_dir)
        df_final = df_final.sort_values(['TradingDay', 'SecuCode'])
        if 'Industry' in df_final.columns:
            df_final['Industry'] = df_final['Industry'].astype('category')
        return df_final
    except Exception as e:
        print(f"Error merging processed files: {e}")
        return pd.DataFrame()

def train_model(df_data):
    print(">>> [Model] 开始滚动训练 (Detailed Metrics: IC, RankIC, L/S Ret, Sharpe, Turnover)...")
    
    feature_cols = [c for c in df_data.columns if c.startswith('HF_')] + ['LnCap', 'Industry']
    target_col = 'Ret_Excess'
    
    all_dates = sorted(df_data['TradingDay'].unique())
    months = sorted(list(set([pd.Timestamp(d).to_period('M') for d in all_dates])))
    preds_list = []
    
    # 策略参数：Top K 持仓
    TOP_K = 50
    # 上一期的持仓集合 (用于计算换手率)
    prev_holdings = set()
    
    start_period = pd.Period(Config.START_PREDICT, 'M')
    try: start_idx = months.index(start_period)
    except: start_idx = Config.ROLLING_WINDOW

    iterator = tqdm(range(start_idx, len(months)), desc="[Model] Walk-Forward")
        
    for i in iterator:
        test_month = months[i]
        
        # 1. 切分训练/测试集
        test_mask = df_data['TradingDay'].dt.to_period('M') == test_month
        test_dates = df_data.loc[test_mask, 'TradingDay'].unique()
        if len(test_dates) == 0: continue
        
        test_start_date = test_dates.min()
        available_dates = [d for d in all_dates if d < test_start_date]
        train_end_idx = len(available_dates) - 3
        if train_end_idx < Config.ROLLING_WINDOW * 20: continue 
        train_end_date = available_dates[train_end_idx]
        train_start_month = months[i - Config.ROLLING_WINDOW]
        
        train_mask = (df_data['TradingDay'].dt.to_period('M') >= train_start_month) & \
                     (df_data['TradingDay'] <= train_end_date) & \
                     (df_data[target_col].notna())
        
        X_train = df_data.loc[train_mask, feature_cols]
        y_train = df_data.loc[train_mask, target_col]
        X_test = df_data.loc[test_mask, feature_cols]
        y_test = df_data.loc[test_mask, target_col]
        
        meta_cols = ['TradingDay', 'SecuCode', 'Ret_Hold', 'Ret_Excess']
        meta_test = df_data.loc[test_mask, meta_cols]
        
        if len(X_train) < 100 or len(X_test) == 0: continue
        
        # 2. 模型训练
        model = xgb.XGBRegressor(
            max_depth=5, learning_rate=0.05, n_estimators=100,
            n_jobs=4, random_state=42, objective='reg:squarederror',
            enable_categorical=True, tree_method='hist'
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # 构造临时 DataFrame 用于计算指标
        df_res = meta_test.copy()
        df_res['Pred'] = pred
        
        # =========================================================================
        # 3. 复杂指标计算 (按日计算后汇总)
        # =========================================================================
        
        daily_stats = []
        month_turnover_sum = 0.0
        valid_days = 0
        
        # 按天循环计算每日的截面指标
        for day, sub in df_res.groupby('TradingDay'):
            if len(sub) < 10: continue
            
            # A. IC & RankIC
            ic_pearson = sub['Pred'].corr(sub['Ret_Excess'], method='pearson')
            ic_spearman = sub['Pred'].corr(sub['Ret_Excess'], method='spearman')
            
            # B. 模拟持仓 (Top K vs Bottom K)
            # 降序排列，取 Top K
            top_k_df = sub.nlargest(TOP_K, 'Pred')
            bottom_k_df = sub.nsmallest(TOP_K, 'Pred')
            
            curr_holdings = set(top_k_df['SecuCode'].values)
            
            # C. 多空收益 (Long - Short)
            # Ret_Hold 是 T+1 到 T+2 的收益，代表基于 T 日预测进行交易的收益
            ret_long = top_k_df['Ret_Hold'].mean()
            ret_short = bottom_k_df['Ret_Hold'].mean()
            ret_ls = ret_long - ret_short
            
            # D. 换手率计算 (Strategy Turnover)
            # 换手率 = (买入金额 + 卖出金额) / (2 * 持仓市值)
            # 简化版：仅看持仓集合变化比例。Turnover = 1 - (Intersection / K)
            # 这是一个近似，假设等权
            if len(prev_holdings) > 0:
                overlap = len(curr_holdings.intersection(prev_holdings))
                turnover = 1.0 - (overlap / len(prev_holdings))
                month_turnover_sum += turnover
            else:
                month_turnover_sum += 1.0 # 第一天全买入
            
            prev_holdings = curr_holdings # 更新持仓
            valid_days += 1
            
            daily_stats.append({
                'IC': ic_pearson,
                'RankIC': ic_spearman,
                'LS_Ret': ret_ls
            })
            
        # 4. 汇总当月指标
        if not daily_stats: continue
        
        stats_df = pd.DataFrame(daily_stats)
        
        # 均值
        mean_ic = stats_df['IC'].mean()
        mean_rank_ic = stats_df['RankIC'].mean()
        
        # 年化多空收益 (简单年化: 日均 * 250)
        mean_daily_ls = stats_df['LS_Ret'].mean()
        ann_ls_ret = mean_daily_ls * 250
        
        # 收益波动比 (年化 Sharpe)
        std_daily_ls = stats_df['LS_Ret'].std()
        if std_daily_ls > 1e-9:
            sharpe = (mean_daily_ls * 250) / (std_daily_ls * np.sqrt(250))
        else:
            sharpe = 0.0
            
        # 月均换手率 (日均换手率)
        avg_daily_turnover = month_turnover_sum / valid_days if valid_days > 0 else 0.0

        # 5. 打印输出
        msg = (f"Pred: {test_month} | "
               f"IC: {mean_ic:.4f} | "
               f"RankIC: {mean_rank_ic:.4f} | "
               f"L/S AnnRet: {ann_ls_ret:.2%} | "
               f"Sharpe: {sharpe:.2f} | "
               f"Turnover: {avg_daily_turnover:.2%}")
        tqdm.write(msg)
        
        # 保存结果
        df_res['Pred_Score'] = pred
        df_res['Pred_Rank'] = df_res.groupby('TradingDay')['Pred_Score'].rank(ascending=False)
        preds_list.append(df_res)
        
    if not preds_list: return pd.DataFrame()
    return pd.concat(preds_list)

# -----------------------------------------------------------------------------
# 6. 因子分析模块
# -----------------------------------------------------------------------------
class FactorAnalyzer:
    @staticmethod
    def calc_forward_returns(df_daily, periods):
        print(">>> [Analyzer] Calculating Forward Returns for Multi-Period Testing...")
        # 优化 pivot 性能，防止内存溢出
        df_daily = df_daily.sort_values(['TradingDay', 'SecuCode'])
        df_pivot = df_daily.pivot(index='TradingDay', columns='SecuCode', values='ClosePrice')
        
        forward_rets = {}
        for name, days in periods.items():
            # 计算 T+1 到 T+1+N 的收益率 (Close_t to Close_{t+n})
            ret_n = df_pivot.pct_change(days).shift(-days)
            
            # 堆叠回长格式
            ret_stack = ret_n.stack().reset_index()
            ret_stack.columns = ['TradingDay', 'SecuCode', f'Ret_{name}']
            forward_rets[name] = ret_stack
            
        return forward_rets

    @staticmethod
    def plot_dual_axis(group_nav, ls_nav, title, metrics_text):
        """
        绘制双轴图并保存到 plots 文件夹
        """
        # 1. 创建画布
        fig, ax1 = plt.subplots(figsize=(10, 8))
        
        # 2. 左轴：绘制5组分层净值
        colors = sns.color_palette("RdYlGn", 5) 
        for i in range(5):
            nav_series = group_nav[i]
            if not nav_series.empty:
                # 补一个起始点 1.0
                start_date = nav_series.index[0] - pd.Timedelta(days=1)
                nav_series = pd.concat([pd.Series([1.0], index=[start_date]), nav_series])
            
            ax1.plot(nav_series.index, nav_series, color=colors[i], 
                     alpha=0.8, linewidth=1.5, label=f'Group {i+1}')
            
        ax1.set_ylabel('Group Net Value (Base=1)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', title='Groups')
        ax1.grid(True, alpha=0.3)

        # 3. 右轴：绘制多空净值
        ax2 = ax1.twinx()
        if not ls_nav.empty:
            start_date = ls_nav.index[0] - pd.Timedelta(days=1)
            ls_nav_plot = pd.concat([pd.Series([1.0], index=[start_date]), ls_nav])
        else:
            ls_nav_plot = ls_nav

        ax2.plot(ls_nav_plot.index, ls_nav_plot, color='black', linestyle='--', linewidth=2.5, label='Long-Short (Net)')
        
        # 填充颜色
        ax2.fill_between(ls_nav_plot.index, ls_nav_plot, 1.0, where=(ls_nav_plot>=1.0), facecolor='red', alpha=0.1)
        ax2.fill_between(ls_nav_plot.index, ls_nav_plot, 1.0, where=(ls_nav_plot<1.0), facecolor='green', alpha=0.1)
        
        ax2.set_ylabel('L/S Net Value (Base=1)', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='upper right')

        plt.title(title, fontsize=15)
        plt.tight_layout()

        # 4. 保存图片
        if not os.path.exists(Config.PATH_PLOTS):
            os.makedirs(Config.PATH_PLOTS, exist_ok=True)

        clean_title = title.replace(' ', '_').replace('|', '-').replace(':', '').replace('/', '-')
        filename = f"{clean_title}.png"
        save_path = os.path.join(Config.PATH_PLOTS, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # 关闭画布
        print(f">>> Plot saved to: {save_path}")

    @staticmethod
    def analyze_single_period(df_merged, factor_col, ret_col, period_name, period_days, is_ascending):
        """
        核心分析逻辑：计算 IC, RankIC, 多空收益, 换手率, 最大回撤
        """
        # 1. 数据准备
        data = df_merged[['TradingDay', 'SecuCode', factor_col, ret_col]].dropna()
        if data.empty: return None
        
        # 2. 每日分组 (Quintiles)
        try:
            # duplicates='drop' 处理大量相同值的情况
            data['Group'] = data.groupby('TradingDay')[factor_col].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
            )
        except:
            return None
        
        # 3. 计算每日截面收益
        daily_group_ret_N = data.groupby(['TradingDay', 'Group'])[ret_col].mean().unstack()
        
        # 处理方向:
        # 如果 is_ascending=True (值越大越好), Group 4 是多头
        # 如果 is_ascending=False (值越小越好), Group 0 是多头 -> 翻转列顺序让 Group 4 变多头
        if not is_ascending:
            daily_group_ret_N = daily_group_ret_N[daily_group_ret_N.columns[::-1]]
            daily_group_ret_N.columns = range(5)

        # 4. 计算 多空 N 日原始收益 (Long - Short)
        # 注意：这里的收益是 N 天的累计收益
        ls_ret_N_raw = daily_group_ret_N[4] - daily_group_ret_N[0]

        # ---------------------------------------------------------------------
        # Part A: 指标计算 (Resampling / RankIC / Turnover / DD)
        # ---------------------------------------------------------------------
        
        # === 1. 降采样 (Resampling) ===
        # 模拟真实持仓：每 period_days 换仓一次
        resampled_idx = ls_ret_N_raw.index[::period_days] 
        ls_ret_N_resampled = ls_ret_N_raw.loc[resampled_idx]
        
        # === 2. IC 计算 (Pearson & Spearman) ===
        def calc_daily_ic_stats(sub):
            if len(sub) < 10: return pd.Series({'IC': np.nan, 'RankIC': np.nan})
            p_ic = sub[factor_col].corr(sub[ret_col], method='pearson')
            s_ic = sub[factor_col].corr(sub[ret_col], method='spearman')
            return pd.Series({'IC': p_ic, 'RankIC': s_ic})

        daily_ic_df = data.groupby('TradingDay').apply(calc_daily_ic_stats)
        
        # 降采样统计 IC Mean (消除自相关性影响)
        resampled_ic_df = daily_ic_df.loc[resampled_idx]
        ic_mean = resampled_ic_df['IC'].mean()
        rank_ic_mean = resampled_ic_df['RankIC'].mean()
        
        # === 3. 换手率计算 (Turnover) ===
        # 计算逻辑：1 - (本期持仓与上期持仓的重叠率)
        # 目标组：调整后的 Group 4 (多头组)
        target_raw_group_idx = 4 if is_ascending else 0
        
        # 获取每一天属于多头组的股票 Set
        daily_long_stocks = data[data['Group'] == target_raw_group_idx].groupby('TradingDay')['SecuCode'].apply(set)
        
        turnover_vals = []
        # 只在调仓日计算
        valid_dates = sorted(daily_long_stocks.index)
        # 找到 resampled_idx 在 valid_dates 里的子集
        rebal_dates = [d for d in resampled_idx if d in valid_dates]
        
        for i in range(1, len(rebal_dates)):
            curr_date = rebal_dates[i]
            prev_date = rebal_dates[i-1]
            
            curr_set = daily_long_stocks[curr_date]
            prev_set = daily_long_stocks[prev_date]
            
            if len(curr_set) > 0:
                stayed = len(curr_set.intersection(prev_set))
                t_rate = 1.0 - (stayed / len(curr_set))
                turnover_vals.append(t_rate)
        
        avg_turnover = np.mean(turnover_vals) if turnover_vals else np.nan

        # === 4. 费后收益与风险指标 ===
        cost_per_period = Config.COST_RATE_TOTAL 
        ls_ret_N_net_resampled = ls_ret_N_resampled - cost_per_period
        
        # 年化系数
        count_per_year = 250.0 / period_days
        
        # 年化收益
        mean_ret_N = ls_ret_N_net_resampled.mean()
        ann_ret = (1 + mean_ret_N) ** count_per_year - 1
        
        # 年化波动率
        std_ret_N = ls_ret_N_net_resampled.std()
        ann_vol = std_ret_N * np.sqrt(count_per_year)
        
        # 收益波动比
        ret_vol_ratio = ann_ret / ann_vol if ann_vol > 1e-6 else 0
        
        # 最大回撤 (Max Drawdown)
        nav_resampled = (1 + ls_ret_N_net_resampled).cumprod()
        dd_series = nav_resampled / nav_resampled.cummax() - 1
        max_dd = dd_series.min()

        # ---------------------------------------------------------------------
        # Part B: 输出构建
        # ---------------------------------------------------------------------

        # 1. 绘图用的详细文本
        metrics_text = (
            f"IC Mean:      {ic_mean:.4f}\n"
            f"RankIC Mean:  {rank_ic_mean:.4f}\n"
            f"L/S Ann. Ret: {ann_ret:.2%}\n"
            f"L/S Ann. Vol: {ann_vol:.2%}\n"
            f"Ret/Vol:      {ret_vol_ratio:.2f}\n"
            f"Avg Turnover: {avg_turnover:.2%}\n"
            f"Max DD:       {max_dd:.2%}"
        )

        # 2. 命令行输出字符串 (增加 MaxDD)
        # 格式：[Freq] IC | RankIC | Ret | R/V | MaxDD | Turn
        cli_msg = (f"[{period_name: <8}] "
                   f"IC:{ic_mean:.3f} | "
                   f"RankIC:{rank_ic_mean:.3f} | "
                   f"Ret:{ann_ret:.1%} | "
                   f"R/V:{ret_vol_ratio:.2f} | "
                   f"MaxDD:{max_dd:.1%} | "   # <--- 新增 Max DD 输出
                   f"Turn:{avg_turnover:.2f}")

        # 3. 准备绘图曲线 (平滑处理，用于展示)
        group_ret_daily = daily_group_ret_N / period_days
        group_nav = (1 + group_ret_daily).cumprod()
        
        ls_ret_daily_net = (ls_ret_N_raw - cost_per_period) / period_days
        ls_nav = (1 + ls_ret_daily_net).cumprod()
        
        return {
            'group_nav': group_nav,
            'ls_nav': ls_nav,
            'metrics_str': metrics_text,
            'cli_msg': cli_msg  # 返回包含 MaxDD 的字符串
        }

    @staticmethod
    def analyze_factor(df, factor_name, return_col, ascending=True):
        # 兼容接口，实际调用在 run_full_analysis 中
        pass

def plot_factor_correlation(df, method='spearman', title='Factor Correlation Matrix'):
    """
    计算因子相关性并保存热力图到文件夹。
    """
    print(f"--- Plotting Correlation Matrix ({method}) ---")
    
    # 1. 筛选因子列
    factor_cols = [c for c in df.columns if c.startswith('HF_')]
    if not factor_cols:
        print("No columns starting with 'HF_' found.")
        return

    # 2. 计算相关性
    corr = df[factor_cols].corr(method=method)

    # 3. 绘图
    plt.figure(figsize=(12, 10))
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    
    sns.heatmap(
        corr, 
        mask=mask,            
        cmap='coolwarm',      
        vmax=1.0, vmin=-1.0,  
        center=0,             
        annot=True,           
        fmt='.2f',            
        square=True,          
        linewidths=.5,        
        cbar_kws={"shrink": .5} 
    )

    plt.title(f"{title} ({method.capitalize()})", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # --- 保存图片而不是显示 ---
    # 1. 确保文件夹存在
    if not os.path.exists(Config.PATH_PLOTS):
        os.makedirs(Config.PATH_PLOTS, exist_ok=True)
    
    # 2. 构造文件名 (清理非法字符)
    clean_title = title.replace(' ', '_').replace('|', '-').replace('/', '-')
    filename = f"{clean_title}_{method}.png"
    save_path = os.path.join(Config.PATH_PLOTS, filename)
    
    # 3. 保存并关闭
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画布，防止内存泄露
    print(f">>> Plot saved to: {save_path}")

# -----------------------------------------------------------------------------
# 集成到 Pipeline
# -----------------------------------------------------------------------------
def run_factor_analysis_step(df_final, df_pred, df_daily_quote):
    print(f"\n{'='*60}")
    print(f"Starting Factor Analysis (Corrected Direction & Extended Metrics)")
    print(f"{'='*60}")
    
    df_analysis = df_final.copy()
    if not df_pred.empty:
        temp_pred = df_pred[['TradingDay', 'SecuCode', 'Pred_Score']].copy()
        df_analysis = pd.merge(df_analysis, temp_pred, on=['TradingDay', 'SecuCode'], how='left')
    
    # 2. 计算各频率的 Forward Return
    forward_rets_dict = FactorAnalyzer.calc_forward_returns(df_daily_quote, Config.TEST_PERIODS)
    
    # 3. 定义因子与方向
    # 逻辑：False 表示 "Descending" (做多小值)。
    # 因为现在 FactorEngine 输出的是原始值：
    # - SmartMoney (S因子): 越小代表主力吸筹 -> False
    # - CPV (量价相关): 越高代表过热/追涨 -> 越小越好 -> False
    # - CDPDP (一致性): 越高代表一致性强(可能反转) -> 越小越好 -> False
    # - Skew (偏度): 越高(彩票股)越容易亏 -> 越小越好 -> False
    # - Tail (尾盘): 越高代表尾盘偷袭 -> 越小越好 -> False
    # - Amp (振幅): 越高代表分歧大/情绪化 -> 越小越好 -> False
    
    factor_directions = {
        'HF_SmartMoney': False,       
        'HF_Raw_CPV_Mean': False,     
        'HF_CDPDP': False,            
        'HF_Skewness': False,         
        'HF_Tail_Reversal': False,    
        'HF_Amplitude': False,
        'Pred_Score': True # 预测分越高越好
    }

    # 4. 循环分析
    for factor, is_ascending in factor_directions.items():
        if factor not in df_analysis.columns: continue
        
        direction_str = "Ascending" if is_ascending else "Descending"
        print(f"\n>>> Analyzing Factor: {factor} [{direction_str}]")
        
        for period_name, period_days in Config.TEST_PERIODS.items():
            ret_df = forward_rets_dict[period_name]
            
            # Merge factor and specific period return
            df_merged = pd.merge(
                df_analysis[['TradingDay', 'SecuCode', factor]], 
                ret_df, 
                on=['TradingDay', 'SecuCode'], 
                how='inner'
            )
            
            col_ret_name = f'Ret_{period_name}'
            
            res = FactorAnalyzer.analyze_single_period(
                df_merged, factor, col_ret_name, period_name, period_days, is_ascending
            )
            
            if res:
                title = f"Factor: {factor} | Freq: {period_name} | {direction_str}"
                FactorAnalyzer.plot_dual_axis(
                    res['group_nav'], res['ls_nav'], title, res['metrics_str']
                )
                # 打印增强后的命令行信息
                print(f"  {res['cli_msg']}")

    # 5. 相关性图表
    plot_factor_correlation(df_analysis, method='spearman')

# -----------------------------------------------------------------------------
# Main Pipeline 
# -----------------------------------------------------------------------------
def run_pipeline(index_path, universe_name):
    print(f"\n{'='*60}")
    print(f"Running Pipeline for: {universe_name} (Updated for Full Report Replication)")
    print(f"{'='*60}")
    
    # 1. 数据加载
    try:
        df_univ = DataLoader.load_universe(index_path)
        df_daily = DataLoader.load_daily_quote() # <--- 这里加载了原始日行情
    except Exception as e:
        print(f"Error Loading Data: {e}")
        return

    # 2. 因子计算
    fe = FactorEngine(df_univ, cache_dir=f'./temp_factors_{universe_name}')
    df_fac = fe.run()
    if df_fac.empty: return
    del fe; gc.collect()

    # 3. Label 构造
    df_tar = DataLoader.construct_label(df_daily)
    
    # 4. 合并数据
    print(">>> [Data] Merging Factors and Labels...")
    df_full = pd.merge(df_fac, df_univ.rename(columns={'date':'TradingDay', 'symbol':'SecuCode'}), on=['TradingDay', 'SecuCode'], how='inner')
    df_full = pd.merge(df_full, df_tar, on=['TradingDay', 'SecuCode'], how='inner')
    
    # 补全 Volume 等信息 (用于中性化)
    target_cols = ['Volume', 'Amount', 'NonRestrictedShares', 'AFloats']
    missing_cols = [c for c in target_cols if c not in df_full.columns and c in df_daily.columns]
    if missing_cols:
        df_supp = df_daily[['TradingDay', 'SecuCode'] + missing_cols]
        df_full = pd.merge(df_full, df_supp, on=['TradingDay', 'SecuCode'], how='left')

    del df_fac, df_univ, df_tar; gc.collect()
    
    # 合并行业市值
    df_ind, df_cap = DataLoader.load_industry_cap()
    df_full = df_full.sort_values('TradingDay')
    df_full = pd.merge_asof(df_full, df_ind[['SecuCode', 'InfoPublDate', 'tag']], left_on='TradingDay', right_on='InfoPublDate', by='SecuCode', direction='backward')
    df_full['Industry'] = df_full['tag'].fillna('Unknown')
    
    cols_cap = ['TradingDay', 'SecuCode', 'LnCap']
    df_full = pd.merge(df_full, df_cap[cols_cap], on=['TradingDay', 'SecuCode'], how='inner')
    
    # 5. 特征工程 & 训练
    df_final = process_features_and_label(df_full)
    df_pred = train_model(df_final)

    # 7. 全面的因子分析 (周/双周/月频 + 双轴图)
    plot_factor_correlation(df_final, method='spearman', title='Factor Correlation Matrix (Pre-Model)')

    if not df_final.empty:
        run_factor_analysis_step(df_final, df_pred, df_daily)

if __name__ == "__main__":
    if not os.path.exists(Config.ROOT_DIR):
        print(f"Error: Data directory '{Config.ROOT_DIR}' not found.")
    else:
        # run_pipeline(Config.PATH_INDEX_WT_300, "CSI300")
        run_pipeline(Config.PATH_INDEX_WT_500, "CSI500")
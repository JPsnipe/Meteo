import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, welch, csd, coherence, hilbert
import pywt
import io
import re
import unicodedata
from sklearn.cluster import KMeans

# ---------------------------------------------
# Utility functions
# ---------------------------------------------
COL_PATTERNS = {
    'time': r'^(fecha|fechahora|datetime|time|timestamp)$',
    'tws': r'^(velocidadmedia|tws|vel|speed)$',
    'twd': r'^(dir|twd|direction)$',
    'gust': r'^(velocidadmax|gust|racha)$'
}

def load_file(file, sep, decimal):
    if file.name.lower().endswith('.xlsx'):
        df = pd.read_excel(file, decimal=decimal)
    else:
        df = pd.read_csv(file, sep=sep, decimal=decimal)
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    return df


def _normalize(col):
    col = unicodedata.normalize('NFKD', col)
    col = ''.join(c for c in col if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]', '', col.lower())


def detect_columns(df):
    mapping = {}
    for col in df.columns:
        norm = _normalize(col)
        for key, pat in COL_PATTERNS.items():
            if re.match(pat, norm):
                mapping[key] = col
    return mapping

def standardize(df, mapping, tz_str, resample_min=1):
    time_col = mapping.get('time')
    if not time_col or time_col not in df.columns:
        raise KeyError("Missing 'time' column after mapping")
    rename_map = {time_col: 'time'}
    for key in ['tws', 'twd', 'gust']:
        col = mapping.get(key)
        if col and col in df.columns and col != time_col:
            rename_map[col] = key
    df = df.rename(columns=rename_map)
    dt = pd.to_datetime(df['time'], errors='coerce')
    if dt.isna().all():
        raise ValueError("Unable to parse 'time' column")
    dt = dt.dt.tz_localize(tz_str, nonexistent='shift_forward', ambiguous='NaT')
    df.index = dt
    cols = [c for c in ['tws', 'twd', 'gust'] if c in df.columns]
    df = df[cols]
    df = df.sort_index().resample(f'{resample_min}min').mean().interpolate()
    return df
def circular_mean_deg(series, window_min):
    rad = np.deg2rad(series)
    sin_m = np.sin(rad).rolling(window_min).mean()
    cos_m = np.cos(rad).rolling(window_min).mean()
    ang = np.arctan2(sin_m, cos_m)
    return (np.rad2deg(ang) + 360) % 360

def angular_residual(series, base_series):
    diff = (series - base_series + 180) % 360 - 180
    return diff

def detrend_rolling(x, win):
    return x - x.rolling(win, center=True).mean()

def butter_bandpass(x, fs, f1, f2, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
    return filtfilt(b, a, x)

def welch_psd(x, fs, nperseg, noverlap):
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    return f, Pxx

def cross_coherence(x, y, fs, nperseg, noverlap):
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    phase = np.angle(Pxy)
    return f, Cxy, phase

def find_char_periods(psd_periods, psd_vals, psd_periods_dir=None, psd_vals_dir=None,
                      coh_f=None, coh_vals=None, min_per=6, max_per=120, coh_thr=0.25, merge_tol=0.2):
    peaks = []
    mask = (psd_periods >= min_per) & (psd_periods <= max_per)
    per = psd_periods[mask]
    val = psd_vals[mask]
    med = np.median(val)
    for i in range(1, len(per) - 1):
        if val[i] > val[i-1] and val[i] > val[i+1] and val[i] > 3*med:
            peaks.append(per[i])
    merged = []
    for p in peaks:
        if not merged:
            merged.append(p)
        elif abs(p - merged[-1]) / merged[-1] <= merge_tol:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    if coh_f is not None and coh_vals is not None:
        keep = []
        for p in merged:
            f = 1 / p
            idx = (np.abs(coh_f - f)).argmin()
            if coh_vals[idx] >= coh_thr:
                keep.append(p)
        merged = keep
    return merged

def hilbert_envelope(x):
    return np.abs(hilbert(x))

def lag_correlation(a, b, max_lag_min):
    lags = np.arange(-max_lag_min, max_lag_min + 1)
    r = []
    for l in lags:
        if l < 0:
            r.append(np.corrcoef(a[-l:], b[:l or None])[0,1])
        elif l > 0:
            r.append(np.corrcoef(a[:-l or None], b[l:])[0,1])
        else:
            r.append(np.corrcoef(a, b)[0,1])
    r = np.array(r)
    idx = np.nanargmax(np.abs(r))
    return lags, r, lags[idx], r[idx]

def find_mature_phase(df, tws_thr=3, sector=(220,340)):
    tws_ok = df['tws'].rolling(5, min_periods=1).mean() >= tws_thr
    twd_ok = df['twd'].between(sector[0], sector[1])
    good = tws_ok & twd_ok
    groups = (good != good.shift()).cumsum()
    segments = df.groupby(groups).apply(lambda g: (g.index[0], g.index[-1]) if g['tws'].gt(0).all() and good.loc[g.index].all() else None)
    segments = [s for s in segments if s is not None]
    if not segments:
        return None
    lengths = [s[1]-s[0] for s in segments]
    idx = np.argmax(lengths)
    start, end = segments[idx]
    if (end - start).total_seconds()/60 < 20:
        return None
    return start, end

@st.cache_data
def compute_psd_cached(x, fs, nperseg, noverlap):
    return welch_psd(x, fs, nperseg, noverlap)

@st.cache_data
def compute_coh_cached(x, y, fs, nperseg, noverlap):
    return cross_coherence(x, y, fs, nperseg, noverlap)

# Example dataset
def generate_example_data():
    t = pd.date_range('2024-06-01', periods=24*60, freq='1min')
    tws = 5 + 2*np.sin(2*np.pi*t.minute/60) + np.random.randn(len(t))*0.5
    twd = 270 + 10*np.sin(2*np.pi*t.minute/30) + np.random.randn(len(t))*2
    gust = tws + np.random.rand(len(t))*2
    df = pd.DataFrame({'Fecha': t, 'Velocidad Media': tws, 'Dir': twd, 'Velocidad Max': gust})
    return df
# Streamlit App
st.set_page_config(page_title="Meteo Regata", layout="wide")

if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}
    st.session_state['meta'] = {}

sidebar = st.sidebar
sidebar.header("Carga de datos")
files = sidebar.file_uploader("Arrastra CSV/XLSX", type=['csv', 'xlsx'], accept_multiple_files=True)
sep = sidebar.text_input('Separador', value=';')
decimal = sidebar.text_input('Decimal', value=',')
tz_str = sidebar.text_input('Zona horaria', value='Europe/Madrid')
resample_min = sidebar.number_input('Resample (min)', 1, 60, 1)

if files:
    for f in files:
        df = load_file(f, sep, decimal)
        mapping = detect_columns(df)
        with sidebar.expander(f"Mapeo columnas {f.name}"):
            for key in ['time','tws','twd','gust']:
                cols = df.columns.tolist()
                default = mapping.get(key, cols[0]) if cols else None
                mapping[key] = st.selectbox(key, cols, index=cols.index(default) if default in cols else 0, key=f"{f.name}_{key}")
        df_std = standardize(df, mapping, tz_str, resample_min)
        st.session_state.datasets[f.name] = df_std
        st.session_state.meta[f.name] = {'mapping': mapping, 'tz': tz_str, 'resample': resample_min}
elif not st.session_state.datasets:
    df = generate_example_data()
    mapping = detect_columns(df)
    df_std = standardize(df, mapping, tz_str, resample_min)
    st.session_state.datasets['ejemplo'] = df_std
    st.session_state.meta['ejemplo'] = {'mapping': mapping, 'tz': tz_str, 'resample': resample_min}

st.sidebar.header("Parámetros")
detrend_win = st.sidebar.number_input('Ventana detrend (min)', 10, 180, 60)
sector = st.sidebar.slider('Sector on-shore', 0, 360, (220,340))
umb_tws = st.sidebar.number_input('Umbral TWS', 0.0, 20.0, 3.0)

resumen_tab, tiempo_tab, espectral_tab, estad_tab, comp_tab = st.tabs(["Resumen","Tiempo","Espectral","Estadística","Comparativa"])
# Resumen Tab
with resumen_tab:
    st.header("Resumen")
    ds_names = list(st.session_state.datasets.keys())
    ds_name = st.selectbox('Dataset', ds_names)
    df = st.session_state.datasets[ds_name]
    st.write(f"Intervalo: {df.index[0]} - {df.index[-1]} ({len(df)} muestras) TZ: {st.session_state.meta[ds_name]['tz']}")
    col1, col2, col3 = st.columns(3)
    if 'tws' in df:
        col1.metric('TWS̄', f"{df['tws'].mean():.2f} m/s")
    if 'twd' in df:
        col2.metric('TWD̄', f"{circular_mean_deg(df['twd'], len(df)).iloc[-1]:.1f}°")
    if 'gust' in df:
        col3.metric('GUST̄', f"{df['gust'].mean():.2f} m/s")
    if 'tws' in df and 'twd' in df:
        phase = find_mature_phase(df, umb_tws, sector)
        if phase:
            st.success(f"Fase madura: {phase[0]} a {phase[1]} ({(phase[1]-phase[0]).total_seconds()/60:.1f} min)")
        else:
            st.info("No se detectó fase madura")
    else:
        st.warning("Faltan columnas 'tws' o 'twd' para analizar la fase madura")
    if st.button('Exportar resumen'):
        buf = io.StringIO()
        df.describe().to_csv(buf)
        st.download_button('Descargar CSV', buf.getvalue(), file_name=f"{ds_name}_resumen.csv")

# Tiempo Tab
with tiempo_tab:
    st.header("Serie temporal")
    ds_name = st.selectbox('Dataset', list(st.session_state.datasets.keys()), key='time_ds')
    df = st.session_state.datasets[ds_name]
    detrend_win_min = st.number_input('Ventana detrend', 5, 180, detrend_win, key='dt_win')
    show_resid = st.checkbox('Mostrar residuo', True)
    tws_res = twd_res = None
    if show_resid:
        if 'tws' in df:
            tws_res = detrend_rolling(df['tws'], detrend_win_min)
        if 'twd' in df:
            twd_mean = circular_mean_deg(df['twd'], detrend_win_min)
            twd_res = angular_residual(df['twd'], twd_mean)
    fig = go.Figure()
    if 'tws' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['tws'], name='TWS'))
    if 'gust' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['gust'], name='GUST', line=dict(dash='dot')))
    if show_resid and tws_res is not None:
        fig.add_trace(go.Scatter(x=df.index, y=tws_res, name='TWS′'))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    if 'twd' in df:
        fig2.add_trace(go.Scatter(x=df.index, y=df['twd'], name='TWD'))
    if show_resid and twd_res is not None:
        fig2.add_trace(go.Scatter(x=df.index, y=twd_res, name='TWD′'))
    st.plotly_chart(fig2, use_container_width=True)
    if show_resid and tws_res is not None and twd_res is not None:
        lags, r, lag_max, rmax = lag_correlation(tws_res.dropna().values, twd_res.dropna().values, 60)
        fig3 = go.Figure(go.Scatter(x=lags, y=r, mode='lines'))
        fig3.add_vline(x=lag_max, line_dash='dash', annotation_text=f"lag={lag_max} r={rmax:.2f}")
        st.plotly_chart(fig3, use_container_width=True)

# Espectral Tab
with espectral_tab:
    st.header('Espectral')
    ds_name = st.selectbox('Dataset', list(st.session_state.datasets.keys()), key='spec_ds')
    df = st.session_state.datasets[ds_name]
    options = [c for c in ['tws', 'twd', 'gust'] if c in df]
    if not options:
        st.warning('Dataset sin variables de viento disponibles')
    else:
        var = st.selectbox('Variable', options)
        detrend_win_min = st.number_input('Ventana detrend', 5, 180, detrend_win, key='spec_win')
        if var == 'twd':
            data = angular_residual(df['twd'], circular_mean_deg(df['twd'], detrend_win_min))
        else:
            data = detrend_rolling(df[var], detrend_win_min)
        nperseg = st.number_input('nperseg', 32, 512, 160)
        fs = 1/60
        f, Pxx = compute_psd_cached(data.dropna().values, fs, nperseg, nperseg//2)
        mask = f > 0
        period = 1/f[mask]/60
        fig = go.Figure(go.Scatter(x=period, y=Pxx[mask]))
        fig.update_xaxes(title='Periodo (min)')
        fig.update_yaxes(type='log')
        st.plotly_chart(fig, use_container_width=True)
        if 'tws' in df and 'twd' in df:
            tws_res = detrend_rolling(df['tws'], detrend_win_min)
            twd_res = angular_residual(df['twd'], circular_mean_deg(df['twd'], detrend_win_min))
            f_c, Cxy, phase = compute_coh_cached(tws_res.dropna().values, twd_res.dropna().values, fs, nperseg, nperseg//2)
            mask_c = f_c > 0
            per_c = 1/f_c[mask_c]/60
            figc = go.Figure(go.Scatter(x=per_c, y=Cxy[mask_c]))
            figc.update_xaxes(title='Periodo (min)')
            figc.update_yaxes(title='Coherencia')
            st.plotly_chart(figc, use_container_width=True)
            peaks = find_char_periods(period, Pxx[mask], per_c, Cxy[mask_c])
            st.write('Periodos característicos:', peaks)

# Estadística Tab
with estad_tab:
    st.header('Wavelet')
    ds_name = st.selectbox('Dataset', list(st.session_state.datasets.keys()), key='wave_ds')
    df = st.session_state.datasets[ds_name]
    detrend_win_min = st.number_input('Ventana detrend', 5, 180, detrend_win, key='wave_win')
    if 'tws' in df:
        tws_res = detrend_rolling(df['tws'], detrend_win_min).fillna(0)
        widths = np.arange(6, 121)
        cwtmat, freqs = pywt.cwt(tws_res.values, widths, 'morl', sampling_period=60)
        power = np.abs(cwtmat)**2
        fig = px.imshow(power, aspect='auto', x=df.index, y=widths, origin='lower')
        fig.update_yaxes(title='Periodo (min)')
        st.plotly_chart(fig, use_container_width=True)
        band = st.slider('Banda (min)', 10, 120, (10,35))
        mask = (widths >= band[0]) & (widths <= band[1])
        amp = np.percentile(power[mask], [50,95])
        st.write({'p50': amp[0], 'p95': amp[1]})
    else:
        st.warning("El dataset no contiene 'tws' para análisis wavelet")

# Comparativa Tab
with comp_tab:
    st.header('Comparativa')
    sel = st.multiselect('Datasets', list(st.session_state.datasets.keys()), default=list(st.session_state.datasets.keys())[:2])
    rows = []
    for name in sel:
        df = st.session_state.datasets[name]
        if 'tws' not in df or 'twd' not in df:
            st.warning(f"{name} sin columnas 'tws' o 'twd'")
            continue
        tws_res = detrend_rolling(df['tws'], detrend_win)
        twd_res = angular_residual(df['twd'], circular_mean_deg(df['twd'], detrend_win))
        f, Pxx = welch_psd(tws_res.dropna().values, 1/60, 160, 80)
        mask = f > 0
        per = 1/f[mask]/60
        peaks = find_char_periods(per, Pxx[mask])
        lags, r, lag_max, rmax = lag_correlation(tws_res.dropna().values, twd_res.dropna().values, 60)
        rows.append({'dataset': name, 'tws_mean': df['tws'].mean(),
                     'twd_mean': circular_mean_deg(df['twd'], len(df)).iloc[-1],
                     'peaks': peaks, 'lag': lag_max, 'r': rmax})
    if rows:
        st.dataframe(pd.DataFrame(rows))
        if len(rows) >= 2:
            X = []
            for r in rows:
                peak = r['peaks'][0] if r['peaks'] else np.nan
                X.append([r['tws_mean'], peak, r['r']])
            km = KMeans(n_clusters=2, n_init='auto').fit(np.nan_to_num(X))
            st.write('Clusters:', km.labels_)
    else:
        st.info('Sin datasets válidos para comparar')

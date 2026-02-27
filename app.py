import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

# --- ページ設定 ---
st.set_page_config(page_title="Seismic OED Simulator", layout="wide")

# --- ディレクトリ設定 ---
BASE_DIR = Path(__file__).resolve().parent
# ※GitHub公開時は app.py と同じ階層に data, model フォルダを置く想定
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# --- セッションステートの初期化（状態管理用） ---
if 'inversion_done' not in st.session_state:
    st.session_state.inversion_done = False

def reset_state():
    """スライダーが動かされたらインバージョン結果をリセットする"""
    st.session_state.inversion_done = False

# --- 1. 物理シミュレータ (リアルタイム波形生成用) ---
class WaveFDTD2D(torch.nn.Module):
    def __init__(self, dx, dz, dt, nx, nz, device):
        super().__init__()
        self.dx, self.dz, self.dt, self.nx, self.nz = dx, dz, dt, nx, nz
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device) / (dx * dz)
        self.laplacian_kernel = kernel.view(1, 1, 3, 3)

    def forward(self, velocity_model, source, src_x, src_z, rec_x, rec_z):
        device = velocity_model.device
        n_steps, n_receivers = len(source), len(rec_x)
        p = torch.zeros((1, 1, self.nx, self.nz), device=device)
        p_old = torch.zeros_like(p)
        seismogram = torch.zeros((n_receivers, n_steps), device=device)
        v2_dt2 = (velocity_model ** 2) * (self.dt ** 2)
        v2_dt2 = v2_dt2.view(1, 1, self.nx, self.nz)

        for t in range(n_steps):
            laplacian = F.conv2d(p, self.laplacian_kernel, padding=1)
            p_new = 2.0 * p - p_old + v2_dt2 * laplacian
            p_new[0, 0, src_x, src_z] += source[t] * (self.dt**2)
            for i, (rx, rz) in enumerate(zip(rec_x, rec_z)):
                seismogram[i, t] = p_new[0, 0, rx, rz]
            p_old = p.clone()
            p = p_new.clone()
        return seismogram

def get_ricker_wavelet(f_dom, dt, t_max, device):
    n_steps = int(t_max / dt)
    t = torch.arange(n_steps, device=device) * dt
    t0 = 1.0 / f_dom
    return (1.0 - 2.0 * (math.pi * f_dom * (t - t0))**2) * torch.exp(-(math.pi * f_dom * (t - t0))**2)

# --- 2. AIモデル定義 ---
def create_multishot_heatmap(src_coords, size=64, sigma=3.0, device='cpu'):
    b, n_shots, _ = src_coords.shape
    heatmap = torch.zeros((b, n_shots, size, size), device=device)
    grid_x, grid_z = torch.meshgrid(torch.arange(size, dtype=torch.float32, device=device), torch.arange(size, dtype=torch.float32, device=device), indexing='ij')
    for i in range(b):
        for s in range(n_shots):
            sx, sz = src_coords[i, s, 0] * size, src_coords[i, s, 1] * size
            if sx > 0 and sz > 0:
                dist_sq = (grid_x - sx)**2 + (grid_z - sz)**2
                heatmap[i, s] = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmap

class WaveformToSpatial(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 7), stride=(2, 4), padding=(1, 3)), nn.BatchNorm2d(16), nn.LeakyReLU(0.2), 
            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), 
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(64), nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 25, 64 * 8 * 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        )
    def forward(self, x):
        b = x.size(0)
        x = self.encoder(x)
        x = x.view(b, -1) # 【バグ修正済み】 self.view ではなく x.view
        x = self.fc(x)
        x = x.view(b, 64, 8, 8)
        return self.decoder(x)

class SpatialUNet(nn.Module):
    def __init__(self, n_shots=3):
        super().__init__()
        in_channels = 16 + n_shots + 1
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1))
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(64, 32, 3, padding=1))
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        b = self.bottleneck(p2)
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        return self.final(d2)

class HighResInversionNet(nn.Module):
    def __init__(self, n_shots=3):
        super().__init__()
        self.w2s = WaveformToSpatial(in_channels=n_shots)
        self.unet = SpatialUNet(n_shots=n_shots)
    def forward(self, x, src_coords, smooth_bg):
        spatial_features = self.w2s(x)
        source_heatmap = create_multishot_heatmap(src_coords, size=64, sigma=3.0, device=x.device)
        combined = torch.cat([spatial_features, source_heatmap, smooth_bg], dim=1)
        return self.unet(combined)

# --- 3. アセットのロード (Web公開対応版) ---
@st.cache_resource
def load_assets():
    device = torch.device("cpu") # クラウド環境は通常CPUなので明示的に指定
    model = HighResInversionNet(n_shots=3).to(device)
    model_path = MODEL_DIR / "23_best_prior_model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 【変更点】クラウドでのメモリ不足を防ぐため、軽量版データセットを読み込む
    dataset_path = DATA_DIR / "demo_dataset.pt"
    if not dataset_path.exists():
        st.error(f"データセットが見つかりません: {dataset_path}")
        st.stop()
        
    data = torch.load(dataset_path, map_location=device)
    mean_val, std_val = -0.0001, 0.05
    return model, data, device, mean_val, std_val

model, data, device, mean_val, std_val = load_assets()
num_samples = len(data['velocity']) - 1 # 選択可能なデータの最大数

# --- 4. UI 構築 ---
st.title("Interactive Seismic Inversion Simulator 🌍")
st.markdown("観測網（センサー配置・震源位置）を設計し、リアルタイムにAIへインバージョンを実行させます。")

# --- サイドバー: 設定パネル (すべてon_changeでリセット発火) ---
st.sidebar.header("1. Target Geology Selection")
# 【変更点】上限を mini_dataset のサイズに合わせて動的に変更
sample_idx = st.sidebar.slider("Select Geological Model", 0, num_samples, 0, on_change=reset_state)

st.sidebar.header("2. Sensor Configuration")
num_surface = st.sidebar.slider("Surface Sensors (Max 16)", 0, 16, 16, on_change=reset_state)
num_borehole = st.sidebar.slider("Borehole Sensors (Max 16)", 0, 16, 16, on_change=reset_state)

st.sidebar.header("3. Source (Shot) Placement")
use_shot1 = st.sidebar.checkbox("Shot 1 Active", value=True, on_change=reset_state)
shot1_x = st.sidebar.slider("Shot 1 X Position", 5, 59, 15, on_change=reset_state)
shot1_z = st.sidebar.slider("Shot 1 Depth (Z)", 2, 59, 5, on_change=reset_state)

use_shot2 = st.sidebar.checkbox("Shot 2 Active", value=True, on_change=reset_state)
shot2_x = st.sidebar.slider("Shot 2 X Position", 5, 59, 32, on_change=reset_state)
shot2_z = st.sidebar.slider("Shot 2 Depth (Z)", 2, 59, 55, on_change=reset_state)

use_shot3 = st.sidebar.checkbox("Shot 3 Active", value=False, on_change=reset_state)
shot3_x = st.sidebar.slider("Shot 3 X Position", 5, 59, 50, on_change=reset_state)
shot3_z = st.sidebar.slider("Shot 3 Depth (Z)", 2, 59, 5, on_change=reset_state)

# --- メイン画面上部：リアルタイム・プレビュー窓 ---
st.subheader("📍 観測網のプレビュー (Survey Setup Preview)")
st.markdown("ここで設定した震源とセンサーの配置を確認してから、インバージョンを実行してください。")

v_min, v_max = 2000.0, 3000.0
target_vel = data['velocity'][sample_idx]
true_vel_np = target_vel.cpu().numpy()
rec_x_all = np.array(data['sensor_x'])
rec_z_all = np.array(data['sensor_z'])

shots_preview = []
if use_shot1: shots_preview.append((shot1_x, shot1_z))
if use_shot2: shots_preview.append((shot2_x, shot2_z))
if use_shot3: shots_preview.append((shot3_x, shot3_z))

# プレビュー画像の描画
fig_prev, ax_prev = plt.subplots(figsize=(8, 4))
im_prev = ax_prev.imshow(true_vel_np.T, cmap='turbo', vmin=v_min, vmax=v_max, alpha=0.3)

active_rec_x = list(rec_x_all[:num_surface]) + list(rec_x_all[16:16+num_borehole])
active_rec_z = list(rec_z_all[:num_surface]) + list(rec_z_all[16:16+num_borehole])
ax_prev.scatter(active_rec_x, active_rec_z, c='white', marker='v', edgecolors='black', label='Active Sensors')

for i, (sx, sz) in enumerate(shots_preview):
    ax_prev.scatter(sx, sz, c='red', marker='*', s=300, edgecolors='yellow', label=f'Shot {i+1}' if i==0 else "")

ax_prev.set_title("Current Survey Design")
ax_prev.legend(loc='lower left')
st.pyplot(fig_prev)

# --- 実行ボタン ---
if st.button("🚀 インバージョンを実行 (Run Live Inversion)", type="primary", use_container_width=True):
    reset_state()
    with st.spinner("波形データをシミュレーションし、AIで逆解析中... (数秒かかります)"):
        solver = WaveFDTD2D(10.0, 10.0, 0.001, 64, 64, device)
        source_wave = get_ricker_wavelet(15.0, 0.001, 0.4, device)
        
        shots = []
        shots.append((shot1_x, shot1_z) if use_shot1 else (0, 0))
        shots.append((shot2_x, shot2_z) if use_shot2 else (0, 0))
        shots.append((shot3_x, shot3_z) if use_shot3 else (0, 0))
        
        live_seismograms = []
        source_coords_norm = []
        
        for sx, sz in shots:
            if sx > 0:
                seismo = solver(target_vel, source_wave, sx, sz, rec_x_all, rec_z_all)
                if num_surface < 16: seismo[num_surface:16, :] = 0.0
                if num_borehole < 16: seismo[16 + num_borehole:32, :] = 0.0
                live_seismograms.append(seismo)
                source_coords_norm.append([sx / 64.0, sz / 64.0])
            else:
                live_seismograms.append(torch.zeros((32, 400), device=device))
                source_coords_norm.append([0.0, 0.0])
                
        x_input = torch.stack(live_seismograms).unsqueeze(0)
        x_input = (x_input - mean_val) / std_val
        src_input = torch.tensor([source_coords_norm], dtype=torch.float32, device=device)
        norm_target = (target_vel.unsqueeze(0).unsqueeze(0) - v_min) / (v_max - v_min)
        smooth_bg = F.avg_pool2d(norm_target, kernel_size=15, stride=1, padding=7)
        
        with torch.no_grad():
            pred_norm = model(x_input, src_input, smooth_bg)
            st.session_state.pred_vel = pred_norm[0, 0].cpu().numpy() * (v_max - v_min) + v_min
            
        st.session_state.smooth_bg_np = smooth_bg[0, 0].cpu().numpy() * (v_max - v_min) + v_min
        st.session_state.true_vel_np = true_vel_np
        st.session_state.inversion_done = True

# --- 実行結果の表示 ---
if st.session_state.inversion_done:
    st.markdown("---")
    st.subheader("📊 インバージョン結果 (AI Prediction)")
    
    fig_res, axes_res = plt.subplots(1, 2, figsize=(14, 5))
    
    # AIへの事前情報
    im0 = axes_res[0].imshow(st.session_state.smooth_bg_np.T, cmap='turbo', vmin=v_min, vmax=v_max)
    axes_res[0].set_title("Input: Macro-Model (Prior Knowledge)")
    fig_res.colorbar(im0, ax=axes_res[0], label="Velocity (m/s)")
    
    # AIの予測結果
    im1 = axes_res[1].imshow(st.session_state.pred_vel.T, cmap='turbo', vmin=v_min, vmax=v_max)
    axes_res[1].set_title("Output: AI Predicted High-Res Geology")
    fig_res.colorbar(im1, ax=axes_res[1], label="Velocity (m/s)")
    
    st.pyplot(fig_res)

    # --- 答え合わせボタン (Expander) ---
    st.markdown("### ↓ 予測は合っていたでしょうか？ ↓")
    with st.expander("🎯 答え合わせ (Ground Truth / 正解の地質モデルを開く)", expanded=False):
        fig_gt, ax_gt = plt.subplots(figsize=(7, 5))
        im_gt = ax_gt.imshow(st.session_state.true_vel_np.T, cmap='turbo', vmin=v_min, vmax=v_max)
        ax_gt.set_title("Ground Truth: True Complex Geology")
        fig_gt.colorbar(im_gt, ax=ax_gt, label="Velocity (m/s)")
        st.pyplot(fig_gt)
        
        st.info("上のAI予測結果と見比べてみてください。センサーが不足している部分（影）では、予測がぼやけたりズレたりしているのが観察できるはずです。")
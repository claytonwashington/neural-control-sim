#!/usr/bin/env python
"""Generate prediction plots and plant visualization for the dashboard.

Creates:
1. Prediction trace comparison for Causal Latent CA-NODE (z=64, best causal)
2. Prediction trace comparison for CA-NODE baseline (R²=0.90)
3. 3D plant setup visualization using Cleo's viz.plot()
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root on path
sys.path.insert(0, '/snel/home/cbwash2/cleo')

from modeling.data import load_trials_h5
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed

set_seed(DEFAULT_SEED)


def generate_causal_prediction_plot():
    """Generate prediction traces for the best causal latent CA-NODE model."""
    from modeling.models.latent_canode import LatentControlAffineODE
    
    model_path = 'results/causal_z64_pw200_h128_lr3e4/model.pt'
    output_path = 'results/causal_z64_pw200_h128_lr3e4/causal_prediction.png'
    
    if not os.path.exists(model_path):
        print(f'  [SKIP] Model not found: {model_path}')
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = LatentControlAffineODE(
        n_x=ckpt['n_x'], n_u=ckpt['n_u'], z_dim=ckpt['z_dim'],
        hidden_dim=ckpt['hidden'], n_layers=ckpt['n_layers']
    )
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    
    x_mean, x_std = ckpt['x_mean'], ckpt['x_std']
    u_mean, u_std = ckpt['u_mean'], ckpt['u_std']
    
    # Load data
    data = load_trials_h5('data/training_trials.h5')
    x_all, u_all, dt = data['x'], data['u'], float(data['dt'])
    n_trials = x_all.shape[0]
    n_test = DEFAULT_TEST_TRIALS
    n_train = n_trials - n_test
    
    # Use first test trial
    x_test = x_all[n_train]  # (n_ch, T)
    u_test = u_all[n_train]  # (n_u, T)
    x_test_n = (x_test - x_mean) / x_std
    u_test_n = (u_test - u_mean) / u_std
    
    past_window = 200  # 200ms
    future_window = 200  # 200ms
    
    # Generate predictions over first 2 seconds
    T_plot = 2000  # 2s
    n_windows = (T_plot - past_window) // future_window
    
    x_pred_full = np.full_like(x_test_n[:, :T_plot], np.nan)
    t_future = torch.arange(future_window, dtype=torch.float32).to(device) * dt
    
    for w in range(n_windows):
        t_split = past_window + w * future_window
        t0 = t_split - past_window
        t1 = t_split + future_window
        if t1 > T_plot:
            break
        
        x_past = torch.tensor(x_test_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
        u_past = torch.tensor(u_test_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
        u_future = torch.tensor(u_test_n[:, t_split:t1].T, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            x_pred = model.predict(x_past, u_past, u_future, t_future, method='dopri5')
            x_pred_np = x_pred[0].cpu().numpy().T  # (n_ch, future_window)
        
        x_pred_full[:, t_split:t1] = x_pred_np
    
    # Un-normalize for plotting
    x_true_raw = x_test[:, :T_plot]
    x_pred_raw = x_pred_full * x_std + x_mean
    
    # Pick 4 channels with high variance
    ch_vars = np.nanvar(x_true_raw, axis=1)
    top_ch = np.argsort(ch_vars)[::-1][:4]
    
    t_sec = np.arange(T_plot) * dt
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#0a0e1a')
    
    for idx, ch in enumerate(top_ch):
        ax = axes[idx]
        ax.set_facecolor('#111827')
        ax.plot(t_sec, x_true_raw[ch], color='#e5e7eb', alpha=0.9, lw=1.0, label='Ground Truth')
        
        valid = ~np.isnan(x_pred_raw[ch])
        ax.plot(t_sec[valid], x_pred_raw[ch, valid], color='#6366f1', alpha=0.9, lw=1.2, label='Causal Pred')
        
        # Window boundaries
        for w in range(n_windows + 1):
            t_line = (past_window + w * future_window) * dt
            if t_line <= T_plot * dt:
                ax.axvline(t_line, color='#4f46e5', alpha=0.2, ls='--', lw=0.6)
        
        ax.set_ylabel(f'Ch {ch}', color='#9ca3af', fontsize=9)
        ax.tick_params(colors='#6b7280', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a3055')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, facecolor='#1a1f35', edgecolor='#2a3055', labelcolor='#e5e7eb')
    
    axes[-1].set_xlabel('Time (s)', color='#9ca3af', fontsize=10)
    fig.suptitle('Causal Latent CA-NODE (z=64, R²=0.83) — Prediction Traces',
                 color='#e5e7eb', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')
    return output_path


def generate_canode_prediction_plot():
    """Generate prediction traces for the CA-NODE baseline model."""
    from modeling.models.canode import ControlAffineODE
    
    # Try to find a model.pt for a CA-NODE baseline with the 40/10 split
    # Fall back to the original baseline
    model_candidates = [
        'results/sweep_skip_40_10/run_00_h128_l2_lr0.0001_dopri5_noskip_compiled/',
        'results/canode/',
    ]
    
    model_path = None
    model_dir = None
    for d in model_candidates:
        p = os.path.join(d, 'model.pt')
        if os.path.exists(p):
            model_path = p
            model_dir = d
            break
    
    # If no model.pt, we'll use the existing prediction image
    if model_path is None:
        # Use existing prediction from the skip sweep baseline
        existing = 'results/sweep_skip_40_10/run_00_h128_l2_lr0.0001_dopri5_noskip_compiled/canode_prediction.png'
        if os.path.exists(existing):
            print(f'  Using existing plot: {existing}')
            return existing
        print('  [SKIP] No CA-NODE model or existing prediction found')
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Clean state dict keys (may have _orig_mod. prefix from compile)
    state_dict = {}
    for k, v in ckpt['model_state'].items():
        state_dict[k.replace('_orig_mod.', '')] = v
    
    model = ControlAffineODE(
        n_x=ckpt['n_x'], n_u=ckpt['n_u'],
        hidden=ckpt['hidden'], n_layers=ckpt['n_layers'],
        use_skip=ckpt.get('use_skip', False),
        skip_type=ckpt.get('skip_type', 'linear')
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    x_mean, x_std = ckpt['x_mean'], ckpt['x_std']
    u_mean, u_std = ckpt['u_mean'], ckpt['u_std']
    
    data = load_trials_h5('data/training_trials.h5')
    x_all, u_all, dt = data['x'], data['u'], float(data['dt'])
    n_trials = x_all.shape[0]
    n_test = DEFAULT_TEST_TRIALS
    n_train = n_trials - n_test
    
    x_test = x_all[n_train]
    u_test = u_all[n_train]
    x_test_n = (x_test - x_mean) / x_std
    u_test_n = (u_test - u_mean) / u_std
    
    horizon = 200
    T_plot = 2000
    n_windows = T_plot // horizon
    
    x_pred_n = np.zeros_like(x_test_n[:, :T_plot])
    t_win = torch.arange(horizon, dtype=torch.float32).to(device) * dt
    
    for w in range(n_windows):
        t0 = w * horizon
        t1 = t0 + horizon
        x0 = torch.tensor(x_test_n[:, t0], dtype=torch.float32).to(device)
        u_win = torch.tensor(u_test_n[:, t0:t1].T, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            model.set_input(t_win, u_win)
            x_pred_win = model.integrate(x0, t_win)
            x_pred_n[:, t0:t1] = x_pred_win.cpu().numpy().T
    
    x_true_raw = x_test[:, :T_plot]
    x_pred_raw = x_pred_n * x_std + x_mean
    
    ch_vars = np.var(x_true_raw, axis=1)
    top_ch = np.argsort(ch_vars)[::-1][:4]
    
    t_sec = np.arange(T_plot) * dt
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#0a0e1a')
    
    for idx, ch in enumerate(top_ch):
        ax = axes[idx]
        ax.set_facecolor('#111827')
        ax.plot(t_sec, x_true_raw[ch], color='#e5e7eb', alpha=0.9, lw=1.0, label='Ground Truth')
        ax.plot(t_sec, x_pred_raw[ch], color='#10b981', alpha=0.9, lw=1.2, label='CA-NODE Pred')
        
        for ww in range(1, n_windows):
            ax.axvline(ww * horizon * dt, color='#065f46', alpha=0.3, ls='--', lw=0.6)
        
        ax.set_ylabel(f'Ch {ch}', color='#9ca3af', fontsize=9)
        ax.tick_params(colors='#6b7280', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a3055')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, facecolor='#1a1f35', edgecolor='#2a3055', labelcolor='#e5e7eb')
    
    axes[-1].set_xlabel('Time (s)', color='#9ca3af', fontsize=10)
    fig.suptitle('CA-NODE Baseline (R²=0.90) — Prediction Traces (200ms windows)',
                 color='#e5e7eb', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(model_dir, 'canode_prediction_dashboard.png')
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')
    return output_path


def generate_plant_visualization():
    """Generate 3D plant setup visualization."""
    output_path = 'results/plant_setup.png'
    
    try:
        # Try Cleo's built-in viz
        import cleo.viz
        from modeling.plant import build_plant
        import brian2.only as b2
        from brian2 import um
        
        print('  Building plant for visualization...')
        sim, devices = build_plant()
        
        fig, ax = cleo.viz.plot(
            sim=sim,
            axis_scale_unit=um,
            figsize=(10, 8),
            scatterargs={'s': 2, 'alpha': 0.4},
        )
        
        ax.set_title('Cleo E/I Plant: 800E + 200I LIF neurons\n2 OpticFibers + 50-ch MUA Probe',
                      color='#e5e7eb', fontsize=11, fontweight='bold', pad=15)
        
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#111827')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#2a3055')
        ax.yaxis.pane.set_edgecolor('#2a3055')
        ax.zaxis.pane.set_edgecolor('#2a3055')
        ax.tick_params(colors='#6b7280', labelsize=7)
        ax.xaxis.label.set_color('#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        ax.zaxis.label.set_color('#9ca3af')
        
        # Style the legend
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor('#1a1f35')
            leg.get_frame().set_edgecolor('#2a3055')
            for text in leg.get_texts():
                text.set_color('#e5e7eb')
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {output_path} (using Cleo viz)')
        return output_path
        
    except Exception as e:
        print(f'  Cleo viz failed ({e}), generating matplotlib fallback...')
        
        # Fallback: simple 3D scatter plot
        np.random.seed(42)
        half = 250  # 500um / 2
        
        # Neuron positions
        n_exc, n_inh = 800, 200
        exc_pos = np.random.uniform(-half, half, (n_exc, 3))
        inh_pos = np.random.uniform(-half, half, (n_inh, 3))
        
        # Fiber positions
        quarter = half / 2
        fiber_pos = np.array([[-quarter, -quarter, -half], [quarter, quarter, -half]])
        
        # Probe positions (grid in XZ plane at Y=0)
        n_ch = 50
        n_side = int(np.ceil(np.sqrt(n_ch)))
        xs = np.linspace(-half * 0.8, half * 0.8, n_side)
        zs = np.linspace(-half * 0.8, half * 0.8, n_side)
        xx, zz = np.meshgrid(xs, zs)
        probe_pos = np.column_stack([xx.ravel()[:n_ch], np.zeros(n_ch), zz.ravel()[:n_ch]])
        
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('#0a0e1a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#111827')
        
        ax.scatter(*exc_pos.T, c='#3b82f6', s=2, alpha=0.3, label=f'Excitatory ({n_exc})')
        ax.scatter(*inh_pos.T, c='#ef4444', s=3, alpha=0.4, label=f'Inhibitory ({n_inh})')
        ax.scatter(*fiber_pos.T, c='#10b981', s=200, marker='^', alpha=0.9, edgecolors='#065f46', linewidths=1.5, label=f'OpticFibers (2)', zorder=5)
        ax.scatter(*probe_pos.T, c='#f59e0b', s=25, marker='s', alpha=0.7, edgecolors='#78350f', linewidths=0.5, label=f'MUA Probe ({n_ch}ch)', zorder=4)
        
        ax.set_xlabel('X (μm)', color='#9ca3af', fontsize=9, labelpad=-5)
        ax.set_ylabel('Y (μm)', color='#9ca3af', fontsize=9, labelpad=-5)
        ax.set_zlabel('Z (μm)', color='#9ca3af', fontsize=9, labelpad=-3)
        ax.set_title('Cleo E/I Plant: 800E + 200I LIF neurons\n2 OpticFibers + 50-ch MUA Probe',
                      color='#e5e7eb', fontsize=11, fontweight='bold', pad=15)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#2a3055')
        ax.yaxis.pane.set_edgecolor('#2a3055')
        ax.zaxis.pane.set_edgecolor('#2a3055')
        ax.tick_params(colors='#6b7280', labelsize=7)
        
        leg = ax.legend(loc='upper left', fontsize=8, facecolor='#1a1f35', edgecolor='#2a3055')
        for text in leg.get_texts():
            text.set_color('#e5e7eb')
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {output_path} (matplotlib fallback)')
        return output_path


def generate_latent_node_prediction_plot():
    """Generate prediction traces for the best acausal Latent NODE model.
    
    Since no model.pt is saved for the sweep runs, we look for
    any latent node model checkpoint or generate from existing assets.
    """
    from modeling.models.latent_node import LatentNeuralODE
    
    # Check if any latent node model.pt exists
    latent_dirs = [
        'results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5/',
    ]
    
    for d in latent_dirs:
        p = os.path.join(d, 'model.pt')
        if os.path.exists(p):
            print(f'  Found model: {p}')
            # Generate prediction plot
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt = torch.load(p, map_location=device, weights_only=False)
            model = LatentNeuralODE(
                n_x=ckpt['n_x'], n_u=ckpt['n_u'], z_dim=ckpt['z_dim'],
                hidden_dim=ckpt['hidden'], n_layers=ckpt['n_layers']
            )
            model.load_state_dict(ckpt['model_state'])
            model = model.to(device)
            model.eval()
            
            x_mean, x_std = ckpt['x_mean'], ckpt['x_std']
            u_mean, u_std = ckpt['u_mean'], ckpt['u_std']
            
            data = load_trials_h5('data/training_trials.h5')
            x_all, u_all, dt = data['x'], data['u'], float(data['dt'])
            n_trials = x_all.shape[0]
            n_test = DEFAULT_TEST_TRIALS
            n_train = n_trials - n_test
            
            x_test = x_all[n_train]
            u_test = u_all[n_train]
            x_test_n = (x_test - x_mean) / x_std
            u_test_n = (u_test - u_mean) / u_std
            
            horizon = 200
            T_plot = 2000
            n_windows = T_plot // horizon
            
            x_pred_n = np.zeros_like(x_test_n[:, :T_plot])
            t_win = torch.arange(horizon, dtype=torch.float32).to(device) * dt
            
            for w in range(n_windows):
                t0 = w * horizon
                t1 = t0 + horizon
                x_win = torch.tensor(x_test_n[:, t0:t1].T, dtype=torch.float32).unsqueeze(0).to(device)
                u_win = torch.tensor(u_test_n[:, t0:t1].T, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    x_pred_win = model.predict(x_win, u_win, t_win, method='dopri5')
                    x_pred_n[:, t0:t1] = x_pred_win[0].cpu().numpy().T
            
            x_true_raw = x_test[:, :T_plot]
            x_pred_raw = x_pred_n * x_std + x_mean
            
            ch_vars = np.var(x_true_raw, axis=1)
            top_ch = np.argsort(ch_vars)[::-1][:4]
            
            t_sec = np.arange(T_plot) * dt
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
            fig.patch.set_facecolor('#0a0e1a')
            
            for idx, ch in enumerate(top_ch):
                ax = axes[idx]
                ax.set_facecolor('#111827')
                ax.plot(t_sec, x_true_raw[ch], color='#e5e7eb', alpha=0.9, lw=1.0, label='Ground Truth')
                ax.plot(t_sec, x_pred_raw[ch], color='#f59e0b', alpha=0.9, lw=1.2, label='Latent NODE Pred')
                
                for ww in range(1, n_windows):
                    ax.axvline(ww * horizon * dt, color='#78350f', alpha=0.3, ls='--', lw=0.6)
                
                ax.set_ylabel(f'Ch {ch}', color='#9ca3af', fontsize=9)
                ax.tick_params(colors='#6b7280', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color('#2a3055')
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=8, facecolor='#1a1f35', edgecolor='#2a3055', labelcolor='#e5e7eb')
            
            axes[-1].set_xlabel('Time (s)', color='#9ca3af', fontsize=10)
            fig.suptitle('Latent NODE (z=64, R²=0.94) — Prediction Traces (200ms windows)',
                         color='#e5e7eb', fontsize=12, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            
            output_path = os.path.join(d, 'latent_node_prediction.png')
            fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {output_path}')
            return output_path
    
    print('  [SKIP] No Latent NODE model.pt found')
    return None


if __name__ == '__main__':
    os.chdir('/snel/home/cbwash2/cleo')
    
    print('=== Generating Dashboard Assets ===')
    
    print('\n1. Plant Setup Visualization:')
    plant_path = generate_plant_visualization()
    
    print('\n2. Causal Latent CA-NODE Prediction:')
    causal_path = generate_causal_prediction_plot()
    
    print('\n3. CA-NODE Baseline Prediction:')
    canode_path = generate_canode_prediction_plot()
    
    print('\n4. Latent NODE Prediction:')
    latent_path = generate_latent_node_prediction_plot()
    
    print('\n=== Summary ===')
    for name, path in [('Plant', plant_path), ('Causal', causal_path), ('CA-NODE', canode_path), ('Latent NODE', latent_path)]:
        status = '✅' if path and os.path.exists(path) else '❌'
        print(f'  {status} {name}: {path}')
    
    print('\nDone!')

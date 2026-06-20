import os
import json
import numpy as np
import sys

sys.path.append('/mnt/cbwash2/cleo')
from modeling.scripts.sweep_canode import parse_run_log

def regenerate():
    configs = []
    for lr in [1e-4, 5e-4]:
        # 1. No skip baseline
        configs.append({
            'hidden': 128,
            'n_layers': 2,
            'lr': lr,
            'method': 'dopri5',
            'compile': True,
            'skip': False,
        })
        # 2. MLP skip baseline (Phase 2.5 style)
        configs.append({
            'hidden': 128,
            'n_layers': 2,
            'lr': lr,
            'method': 'dopri5',
            'compile': True,
            'skip': True,
            'skip_type': 'mlp',
            'skip_weight_decay': None,
        })
        # 3. Linear skip with varying weight decay
        for swd in [None, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            configs.append({
                'hidden': 128,
                'n_layers': 2,
                'lr': lr,
                'method': 'dopri5',
                'compile': True,
                'skip': True,
                'skip_type': 'linear',
                'skip_weight_decay': swd,
            })

    output_dir = '/mnt/cbwash2/cleo/results/sweep_skip_40_10'
    completed_runs = []

    for idx, config in enumerate(configs):
        run_id = f'run_{idx:02d}_h{config["hidden"]}_l{config["n_layers"]}_lr{config["lr"]}_{config["method"]}'
        if config.get('skip', True):
            stype = config.get('skip_type', 'mlp')
            swd = config.get('skip_weight_decay', None)
            run_id += f'_skip_{stype}'
            if swd is not None:
                run_id += f'_swd{swd}'
        else:
            run_id += '_noskip'
        if config['compile']:
            run_id += '_compiled'

        log_path = os.path.join(output_dir, f'{run_id}.log')
        metrics = parse_run_log(log_path)
        metrics['idx'] = idx
        metrics['config'] = config
        metrics['run_id'] = run_id
        metrics['elapsed_s'] = metrics.get('elapsed_s', 0.0)
        completed_runs.append(metrics)

    completed_runs.sort(key=lambda r: -r.get('mean_windowed_r2', -999))

    print(f"\n{'Idx':<4} {'Hidden':<6} {'Layers':<5} {'LR':<7} {'Skip':<8} {'SWD':<6} "
          f"{'Train L':<8} {'Val L':<8} "
          f"{'Win R2':<8} {'Win MSE':<8} {'OL R2':<8}")
    print('-' * 90)
    for run in completed_runs:
        cfg = run['config']
        train_l_val = run.get('best_train_loss')
        train_l = f'{train_l_val:.4f}' if train_l_val is not None else 'nan'
        val_l_val = run.get('best_val_loss')
        val_l = f'{val_l_val:.4f}' if val_l_val is not None else 'nan'
        win_r2_val = run.get('mean_windowed_r2')
        win_r2 = f'{win_r2_val:.4f}' if win_r2_val is not None else 'nan'
        win_mse_val = run.get('mean_windowed_mse')
        win_mse = f'{win_mse_val:.4f}' if win_mse_val is not None else 'nan'
        ol_r2_val = run.get('mean_test_r2')
        ol_r2 = f'{ol_r2_val:.4f}' if ol_r2_val is not None else 'nan'
        skip_str = cfg.get('skip_type', 'mlp') if cfg.get('skip', True) else 'none'
        swd_val = cfg.get('skip_weight_decay', None)
        swd_str = 'none' if swd_val is None else f'{swd_val:.0e}'
        print(f"{run['idx']+1:<4} {cfg['hidden']:<6} {cfg['n_layers']:<5} "
              f"{cfg['lr']:<7} {skip_str:<8} {swd_str:<6} "
              f"{train_l:<8} {val_l:<8} {win_r2:<8} {win_mse:<8} {ol_r2:<8}")

    summary_path = os.path.join(output_dir, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(completed_runs, f, indent=2)
    print(f'\nSaved sweep summary to {summary_path}')

if __name__ == '__main__':
    regenerate()

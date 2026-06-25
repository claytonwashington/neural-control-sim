"""Launch LFADS PBT for Cleo spiking data.

Usage:
    python scripts/run_lfads_pbt.py --model-config cleo_spiking_ext --tag ext_input
    python scripts/run_lfads_pbt.py --model-config cleo_spiking_noext --datamodule-config cleo_spiking_noext --tag no_ext_input
"""
import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-config", required=True, help="Model config name")
parser.add_argument("--datamodule-config", default="cleo_spiking", help="Datamodule config name")
parser.add_argument("--tag", default="pbt", help="Run tag")
parser.add_argument("--num-workers", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=2000)
args = parser.parse_args()

# ---------- OPTIONS ----------
MODEL_STR = args.model_config
DATAMODULE_STR = args.datamodule_config
RUN_TAG = datetime.now().strftime("%y%m%d") + f"_{args.tag}"
RUN_DIR = Path("/mnt/cbwash2/cleo-worktrees/bidir-v2-plant/results") / f"lfads_{args.tag}" / RUN_TAG
HYPERPARAM_SPACE = {
    "model.lr_init": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=4e-3
    ),
    "model.dropout_rate": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "model.train_aug_stack.transforms.0.cd_rate": HyperParam(
        0.01, 0.7, explore_wt=0.3, enforce_limits=True, init=0.5, sample_fn="uniform"
    ),
    "model.kl_co_scale": HyperParam(1e-6, 1e-4, explore_wt=0.8),
    "model.kl_ic_scale": HyperParam(1e-6, 1e-3, explore_wt=0.8),
    "model.l2_gen_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
    "model.l2_con_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
}
# ------------------------------

def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}

init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}
mandatory_overrides = {
    "datamodule": DATAMODULE_STR,
    "model": MODEL_STR,
}
RUN_DIR.mkdir(parents=True, exist_ok=True)
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

metric = "valid/recon_smth"
num_trials = args.num_workers
perturbation_interval = 25
burn_in_period = 80 + 10

analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/pbt_cleo.yaml",
        do_posterior_sample=False,
    ),
    metric=metric,
    mode="min",
    name=RUN_DIR.name,
    stop=ImprovementRatioStopper(
        num_trials=num_trials,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=2, gpu=0.5),
    num_samples=num_trials,
    local_dir=str(RUN_DIR.parent),
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)

print(f"\nBest trial: {analysis.best_trial.trial_id}")
print(f"Best {metric}: {analysis.best_result[metric]:.4f}")

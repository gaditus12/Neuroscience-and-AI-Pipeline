#!/usr/bin/env python3
import argparse
import subprocess
import json
import sys
import os
import time

# your two dicts
FEATURES = {
    # 'combat': [
    #     # 'Fz_norm-ComBat',
    #     # 'O1_norm-ComBat',
    #     'O2_norm-ComBat',
    #     # 'Oz_norm-ComBat',
    #     # 'PO3_norm-ComBat',
    #     'PO4_norm-ComBat',
    #     # 'TP7_norm-ComBat',
    #     # 'TP8_norm-ComBat',
    # ],
    'z':[
        # 'Fz_spi_norm-z',
        # 'O1_spi_norm-z',
        # 'Oz_spi_norm-z',
        'O2_spi_norm-z',
        # 'po3_spi_norm-z',
        'po4_spi_norm-z',
        # 'tp7_spi_norm-z',
        # 'tp8_spi_norm-z',
    ],
    'raw': [
        # 'Fz_spi',
        # 'O1_spi',
        'O2_SPI',
        # 'Oz_spi',
        # 'PO3_spi',
        'PO4_SPI',
        # 'TP7_spi',
        # 'TP8_spi',
    ],
    'combat': [
        # 'Fz_spi_norm-ComBat',
        # 'O1_spi_norm-ComBat',
        'O2_spi_norm-ComBat',
        # 'Oz_spi_norm-ComBat',
        # 'PO3_spi_norm-ComBat',
        'PO4_spi_norm-ComBat',
        # 'TP7_spi_norm-ComBat',
        # 'TP8_spi_norm-ComBat',
    ]
}

def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--klist', nargs='+', type=int,
                     help="List of frozen_features values")
    grp.add_argument('--frozen_features', type=int,
                     help="Single frozen_features value")
    p.add_argument('--features_type', choices=['combat', 'z', ],#'raw'],
                   help="If given, only run that set")
    p.add_argument('--cv_method', choices=['loso','kfold'],
                   help="If given, only run that CV")
    p.add_argument('--kfold_splits', type=int, default=10)
    p.add_argument('--permu_count', type=int, default=0)
    p.add_argument('--eval', action='store_true')
    p.add_argument('--script_path', default='ml_model_playground.py')
    p.add_argument('--output', default='runs_summary.json')
    return p.parse_args()

def to_k(n):
    if n>=1000:
        return f"{n/1000:.1f}k".rstrip('0').rstrip('.')
    return str(n)

def main():
    args = parse_args()

    # which frozen values?
    if args.klist:
        freeze_vals = args.klist
    elif args.frozen_features is not None:
        freeze_vals = [args.frozen_features]
    else:
        freeze_vals = [None]   # unfrozen

    types = [args.features_type] if args.features_type else list(FEATURES)
    cvs   = [args.cv_method]  if args.cv_method   else ['loso','kfold']

    results = []

    for fv in freeze_vals:
        # build a top‐level run directory for this fv + perm count
        stamp = int(time.time())
        fv_str = f"{fv}k" if fv is not None else "unfrozen"
        perm_str = to_k(args.permu_count) + "perm"
        base_run = os.path.join(
            "ml_model_outputs",
            f"frozen_run_{fv_str}_{perm_str}_{stamp}"
        )
        # we don't mkdir here; EEGAnalyzer will mkdir its own run_directory

        for ft in types:
            for cv in cvs:
                for feat in FEATURES[ft]:
                    # nested path: base_run / features_type / cv_method
                    run_dir = os.path.join(base_run, ft, cv)

                    cmd = [
                        sys.executable, args.script_path,
                        '--features_file', feat,
                        '--permu_count', str(args.permu_count),
                        '--cv_method',   cv,
                        '--kfold_splits', str(args.kfold_splits),
                        '--run_directory', run_dir
                    ]
                    if args.eval:
                        cmd.append('--eval')
                    if fv is not None:
                        cmd += ['--frozen_features', str(fv)]

                    print("→", ' '.join(cmd))
                    proc = subprocess.run(cmd)
                    results.append({
                        'fv': fv, 'features_type': ft, 'cv_method': cv,
                        'feat_file': feat, 'cmd': ' '.join(cmd),
                        'returncode': proc.returncode
                    })

    with open(args.output,'w') as f:
        json.dump(results,f,indent=2)
    print("Done; summary in", args.output)

if __name__=='__main__':
    main()

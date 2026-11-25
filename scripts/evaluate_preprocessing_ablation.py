"""Exploratory within-trial EEG preprocessing ablation (S02)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def features(x, lag):
    """Lagged signal and observation mask, without crossing block boundaries."""
    mask = np.isfinite(x)
    values = np.where(mask, x, 0.0)
    n = len(x)
    return np.concatenate([
        part[lag + offset:n - lag + offset]
        for offset in range(-lag, lag + 1)
        for part in (values, mask.astype(float))
    ], axis=1)


def transform(x, method, keep, center=None, pc=None):
    z = x.copy()
    if method == 'P0':
        return z
    z[:, ~keep] = np.nan
    if method in ('P3', 'P4'):
        valid = np.isfinite(z)
        rows = valid.sum(axis=1) >= 2
        ref = (np.nanmean if method == 'P3' else np.nanmedian)(z[rows], axis=1)
        z[rows] -= ref[:, None]
        z[~rows] = np.nan
    elif method == 'P5':
        # Project out a training-fitted direction using observed channels only.
        v = z[:, keep] - center
        observed = np.isfinite(v)
        denom = observed @ (pc ** 2)
        score = np.divide(np.where(observed, v, 0) @ pc, denom,
                          out=np.zeros(len(v)), where=denom > 1e-12)
        v -= score[:, None] * pc
        z[:, keep] = v + center
    return z


def metrics(y, p):
    valid = np.isfinite(y)
    mse = np.mean((y[valid] - p[valid]) ** 2)
    zero = np.mean(y[valid] ** 2)
    corr, ratios, r2 = [], [], []
    for j in range(y.shape[1]):
        a, b = y[valid[:, j], j], p[valid[:, j], j]
        if len(a) < 2 or a.std() <= 1e-12:
            continue
        ratios.append(b.std() / a.std())
        r2.append(1 - np.sum((a-b)**2) / np.sum((a-a.mean())**2))
        corr.append(np.corrcoef(a,b)[0,1] if b.std() > 1e-12 else 0.0)
    return dict(mse=float(mse), rmse=float(np.sqrt(mse)),
                mean_pearson=float(np.mean(corr)), median_std_ratio=float(np.median(ratios)),
                mean_r2=float(np.mean(r2)), zero_mse=float(zero),
                improvement_vs_zero=float(1-mse/zero) if zero > 0 else np.nan,
                valid_targets=int(valid.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed/numpy')
    parser.add_argument('--output', default='diagnostics/preprocessing/ridge_ablation_s02.csv')
    parser.add_argument('--lag', type=int, default=10)
    parser.add_argument('--gap-seconds', type=float, default=5)
    args = parser.parse_args()
    root = Path(args.data_dir)
    meta = json.loads((root/'subject_02.json').read_text())
    data = np.load(root/'subject_02.npy', mmap_mode='r')
    ear = np.r_[30:36,45:51]
    scalp = np.arange(29)
    for name, expected in [('in_ear', ear), ('scalp', scalp)]:
        if not np.array_equal(meta['indices'][name], expected):
            raise ValueError(f'Unexpected {name} mapping: {meta["indices"][name]}')
    gap = round(args.gap_seconds * meta['fs'])
    rows = []
    for trial, (start, stop) in enumerate(zip(meta['trial_offsets'][:-1], meta['trial_offsets'][1:]), 1):
        x = np.asarray(data[start:stop, ear], dtype=float)
        y = np.asarray(data[start:stop, scalp], dtype=float)
        n = len(x)
        a,b = int(.6*n), int(.8*n)
        blocks = [slice(0,a-gap), slice(a+gap,b-gap), slice(b+gap,n)]
        train = x[blocks[0]]
        if min(len(x[s]) for s in blocks) <= 2*args.lag:
            raise ValueError('Block too short')
        keep1 = np.ones(12,dtype=bool)
        keep1[1] = False  # ELB, exploratory subject-specific choice
        keep2 = keep1.copy()
        for j in range(12):
            v = train[np.isfinite(train[:,j]),j]
            keep2[j] &= len(v) >= 100 and 1.4826*np.median(np.abs(v-np.median(v))) > .01 if len(v) else False
        if keep2.sum() < 2:
            raise ValueError('Too few usable training channels')
        complete = train[:,keep2]
        complete = complete[np.isfinite(complete).all(axis=1)]
        center = pc = None
        if len(complete) >= 100:
            center = complete.mean(axis=0)
            pc = np.linalg.svd(complete-center, full_matrices=False)[2][0]
        ys = [y[s][args.lag:len(y[s])-args.lag] for s in blocks]
        fit_mask = np.isfinite(ys[0]).all(axis=1)
        if not fit_mask.any():
            raise ValueError('No complete training targets')
        for method in ('P0','P1','P2','P3','P4','P5'):
            if method == 'P5' and pc is None:
                print(f'Trial {trial}: P5 skipped, insufficient complete training rows')
                continue
            keep = keep1 if method == 'P1' else keep2
            zs = [features(transform(x[s],method,keep,center,pc),args.lag) for s in blocks]
            for scaled in (False,True):
                scaler = StandardScaler().fit(zs[0][fit_mask]) if scaled else None
                zz = [scaler.transform(z) if scaled else z for z in zs]
                best = None
                for alpha in (.1,1,10,100,1000):
                    model = Ridge(alpha=alpha, solver='cholesky').fit(zz[0][fit_mask],ys[0][fit_mask])
                    score = metrics(ys[1],model.predict(zz[1]))['mse']
                    if best is None or score < best[0]:
                        best = (score,alpha,model)
                score,alpha,model = best
                row = dict(trial=trial,method=method,scaled=scaled,alpha=alpha,val_mse=score,
                           train_rows=int(fit_mask.sum()), **metrics(ys[2],model.predict(zz[2])))
                rows.append(row)
                print(f'T{trial} {method} scaled={scaled} alpha={alpha:g} MSE={row["mse"]:.4f} Pearson={row["mean_pearson"]:.4f}',flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()

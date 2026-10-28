"""Reproducible multi-scenario American put pricing benchmark.
Run: python benchmark_pricing.py
The fine-grid binomial price is a numerical reference, not ground truth.
"""
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from pricing_american_option import (
    binomial_tree_american, trinomial_tree_american,
    fdm_theta_american, longstaff_schwartz_american,
    bjerksund_stensland_2002,
)

K = 100.0
# 3 moneyness levels x 2 maturities x 2 volatility levels = 12 scenarios.
SCENARIOS = [(s, t, vol) for s in (80., 100., 120.)
             for t in (0.5, 2.) for vol in (0.15, 0.35)]
R, Q = 0.05, 0.02

METHODS = {
    'binomial_250': lambda s,t,v,i: binomial_tree_american(s,K,t,R,Q,v,N=250,option_type='put'),
    'trinomial_250': lambda s,t,v,i: trinomial_tree_american(s,K,t,R,Q,v,N=250,option_type='put'),
    'pde_CN_200': lambda s,t,v,i: fdm_theta_american(s,K,t,R,Q,v,M=200,N=250,theta=0.5,option_type='put'),
    'bjerksund_2002': lambda s,t,v,i: bjerksund_stensland_2002(s,K,t,R,Q,v,option_type='put'),
    'LSM_20000': lambda s,t,v,i: longstaff_schwartz_american(s,K,t,R,Q,v,N=60,simulations=20000,seed=2026+i,option_type='put'),
}

rows = []
for i, (s,t,v) in enumerate(SCENARIOS):
    reference = float(binomial_tree_american(s,K,t,R,Q,v,N=1500,option_type='put'))
    for name, func in METHODS.items():
        start = perf_counter()
        try:
            price = float(func(s,t,v,i))
            elapsed = perf_counter() - start
            # Avoid unstable percentage errors on low-priced OTM options.
            abs_error = abs(price - reference)
            rel_error = 100*abs_error/reference if reference > 1.0 else float('nan')
            status = 'OK' if np.isfinite(price) and price >= max(K-s, 0)-1e-8 else 'CHECK'
            rows.append(dict(method=name,S=s,K=K,T=t,sigma=v,reference=reference,
                             price=price,abs_error=abs_error,rel_error_pct=rel_error,
                             runtime_s=elapsed,status=status))
        except Exception as exc:
            rows.append(dict(method=name,S=s,K=K,T=t,sigma=v,reference=reference,
                             price='',abs_error='',rel_error_pct='',runtime_s='',
                             status=f'ERROR: {type(exc).__name__}: {exc}'))

output = Path(__file__).with_name('benchmark_pricing_results.csv')
with output.open('w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

print(f'Benchmark: {len(SCENARIOS)} scenarios x {len(METHODS)} methods = {len(rows)} evaluations')
print('Reference: 1,500-step binomial tree; American puts only; deterministic LSM seeds.')
print('Relative error only on scenarios with reference price > 1 currency unit.')
print(f"{'Method':<19} {'Valid':>7} {'Med abs err':>12} {'P90 abs err':>12} {'P90 rel %':>11} {'Avg ms':>10}")
for name in METHODS:
    entries=[x for x in rows if x['method']==name]
    valid=[x for x in entries if x['status']=='OK']
    if not valid:
        print(f'{name:<19} 0/{len(entries)}  (no valid evaluations)'); continue
    a=np.array([x['abs_error'] for x in valid],dtype=float)
    rel=np.array([x['rel_error_pct'] for x in valid if np.isfinite(x['rel_error_pct'])],dtype=float)
    ms=np.mean([x['runtime_s'] for x in valid])*1000
    p90rel=f'{np.percentile(rel,90):.3f}' if len(rel) else 'n/a'
    print(f'{name:<19} {len(valid):>3}/{len(entries):<3} {np.median(a):>12.4f} {np.percentile(a,90):>12.4f} {p90rel:>11} {ms:>10.1f}')
print('Full results:',output)

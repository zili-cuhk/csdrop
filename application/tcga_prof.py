import os as _os
import sys, json, numpy as np
_H=_os.path.dirname(_os.path.abspath(__file__)); sys.path.insert(0,_os.path.join(_H,'..','python')); sys.path.insert(0,_H)
import csm
from prep import prepare
d = prepare('exclude_unknown'); st = d['e1']*1 + d['e2']*2 + d['dr']*3
P = csm.prep(d['Y'], st, d['Z'], d['cut'], 2)
opt = dict(csm.OPT); opt['refit_max_iter'] = 20000; opt['refit_tol'] = 1e-6
S = np.zeros((3, 6), bool); S[0, 2] = True; S[1, 3] = True
grid = np.array([12., 24., 36.])
out = []
taus = [round(x, 2) for x in np.arange(-0.5, 0.81, 0.1)]
for lab, mask in (('full', None), ('selected', S)):
    X = np.zeros((3, 6)); xi = csm.breslow_init(P, X)
    for fam in ('frank', 'clayton', 'gumbel'):
        for t in taus:
            if fam != 'frank' and t < 0: continue
            if fam == 'gumbel' and t == 0: pass
            cop = (fam, csm.tau2theta(fam, t) if not (fam == 'gumbel' and t == 0) else 1.0)
            f = csm.fit(P, cop, X if mask is None else np.where(mask, X, 0), xi, dict(kind='none'), mask=mask, opt=opt, unpen=True)
            F = csm.cif(P, f['X'], f['xi'], d['Z'], grid).mean(1)
            B, _ = csm.to_raw(P, f['X'], f['xi'])
            out.append(dict(model=lab, fam=fam, tau=t, status=f['status'], ell=float(f['ell']), cif=F.tolist(), B=B.tolist()))
            print(lab, fam, t, f['status'], round(f['ell'], 3), np.round(F, 4).tolist(), flush=True)
json.dump(out, open(_os.path.join(_H, 'out', 'tcga_prof.json'), 'w'))

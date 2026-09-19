import argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common_ch_multidirectional_mc_projected import (
    KAPPA, LAMBDA, A0, A1, A2, ALPHA, BETA, OMEGA0,
    population_geometry, simulate_coefficients, solve_nnqp,
    branch_score_accuracy
)


def profile_objective_components(coef, W):
    d = coef['d']
    wd = W @ d
    denom = float(d @ wd)
    H = W - np.outer(wd, wd) / denom
    return d, H


def estimate_x_alpha(coef, W, starts=None, bound=1.5):
    d, H = profile_objective_components(coef, W)
    g0, b1, b2 = coef['g0'], coef['b1'], coef['b2']
    q11, q22, q12 = coef['q11'], coef['q22'], coef['q12']

    def v_of(x):
        x1, x2 = x
        return g0 + b1*x1 + b2*x2 + q11*x1*x1 + q22*x2*x2 + q12*x1*x2

    def fun(x):
        v = v_of(x)
        return float(v @ H @ v)

    def jac(x):
        x1, x2 = x
        v = v_of(x)
        Hv = H @ v
        dv1 = b1 + 2*q11*x1 + q12*x2
        dv2 = b2 + 2*q22*x2 + q12*x1
        return 2*np.array([dv1 @ Hv, dv2 @ Hv])

    if starts is None:
        starts = [np.zeros(2)]
        starts += [np.array([sx*a, sy*a]) for a in (0.10,0.25,0.50) for sx in (-1,1) for sy in (-1,1)]

    best = None
    for st in starts:
        res = minimize(fun, st, jac=jac, method='L-BFGS-B',
                       bounds=[(-bound,bound),(-bound,bound)],
                       options={'ftol':1e-14,'gtol':1e-9,'maxiter':250})
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    v = v_of(x)
    alpha_hat = float((d @ W @ v) / (d @ W @ d))
    return x, alpha_hat, float(best.fun)


def a_of_x(x):
    x1, x2 = x
    return np.array([x1-x2, x1+x2-1.0, 2.0-2.0*x1])


def estimate_prelim(outs, checkpoints, R):
    prelim = {}
    I = np.eye(6)
    for n in checkpoints:
        xs = np.zeros((R,2)); alphas = np.zeros(R); qvals = np.zeros(R)
        for i in range(R):
            coef = {k:v[i] for k,v in outs[n].items()}
            xs[i], alphas[i], qvals[i] = estimate_x_alpha(coef, I)
        prelim[n] = {'x':xs,'alpha':alphas,'Q':qvals}
    return prelim


def accumulate_omega_for_checkpoints(R, checkpoints, burn, seed, prelim):
    """Re-simulate identical paths and accumulate g_t(theta_tilde_n)g_t' for each n.
    Since each checkpoint has its own preliminary theta, keep one 6x6 accumulator per checkpoint.
    """
    cps = sorted(checkpoints)
    maxn = max(cps)
    rng = np.random.default_rng(seed)
    h = np.ones((R,2))
    f = np.sqrt(h) * rng.standard_normal((R,2))
    e = math.sqrt(KAPPA) * rng.standard_normal((R,3))
    y = f @ LAMBDA[:,:2].T + e
    h = OMEGA0 + ALPHA*f*f + BETA*h
    for _ in range(burn):
        f = np.sqrt(h) * rng.standard_normal((R,2))
        e = math.sqrt(KAPPA) * rng.standard_normal((R,3))
        y = f @ LAMBDA[:,:2].T + e
        h = OMEGA0 + ALPHA*f*f + BETA*h

    # Precompute portfolio weights/c for each checkpoint and replication.
    A = {}; c = {}
    for n in cps:
        xs = prelim[n]['x']; al = prelim[n]['alpha']
        aa = np.column_stack([xs[:,0]-xs[:,1], xs[:,0]+xs[:,1]-1.0, 2.0-2.0*xs[:,0]])
        cc = al + KAPPA*np.sum(aa*aa,axis=1)
        A[n]=aa; c[n]=cc

    acc = {n:np.zeros((R,6,6)) for n in cps}
    for t in range(1,maxn+1):
        z = np.concatenate([y,y*y],axis=1)
        f_next = np.sqrt(h)*rng.standard_normal((R,2))
        e_next = math.sqrt(KAPPA)*rng.standard_normal((R,3))
        y_next = f_next @ LAMBDA[:,:2].T + e_next
        h_next = OMEGA0 + ALPHA*f_next*f_next + BETA*h
        for n in cps:
            if t <= n:
                r = np.sum(A[n]*y_next,axis=1)
                u = r*r - c[n]
                g = z*u[:,None]
                acc[n] += g[:,:,None]*g[:,None,:]
        y,h = y_next,h_next
    return {n:acc[n]/n for n in cps}


def regularized_inverse(S, rel_floor=1e-8):
    S = 0.5*(S+S.T)
    vals, vecs = np.linalg.eigh(S)
    scale = max(float(np.max(vals)), 1e-12)
    floor = rel_floor*scale
    vals2 = np.maximum(vals,floor)
    return (vecs/vals2) @ vecs.T, float(np.min(vals)), float(np.max(vals)), float(np.max(vals2)/np.min(vals2))


def feasible_estimates(outs, checkpoints, prelim, Omega):
    R = len(prelim[checkpoints[0]]['x'])
    result={}
    for n in checkpoints:
        xf=np.zeros((R,2)); af=np.zeros(R); qf=np.zeros(R)
        min_eig=np.zeros(R); max_eig=np.zeros(R); cond=np.zeros(R)
        for i in range(R):
            W, mn, mx, cn = regularized_inverse(Omega[n][i])
            min_eig[i],max_eig[i],cond[i]=mn,mx,cn
            coef={k:v[i] for k,v in outs[n].items()}
            x0=prelim[n]['x'][i]
            starts=[x0,np.zeros(2)]
            starts += [np.array([sx*a,sy*a]) for a in (0.10,0.25,0.50) for sx in (-1,1) for sy in (-1,1)]
            xf[i],af[i],qf[i]=estimate_x_alpha(coef,W,starts=starts)
        result[n]={'x':xf,'alpha':af,'Q':qf,'min_eig':min_eig,'max_eig':max_eig,'cond':cond}
    return result


def oracle_x_for_comparison(outs,n,geom,R):
    # Re-implement compactly to avoid importing optimizer helper internals.
    xs=np.zeros((R,2)); ys=np.zeros((R,2))
    W=geom['W']; C=geom['C']
    for i in range(R):
        coef={k:v[i] for k,v in outs[n].items()}
        d=coef['d']; wd=W@d; H=W-np.outer(wd,wd)/(d@wd)
        g0,b1,b2=coef['g0'],coef['b1'],coef['b2']; q11,q22,q12=coef['q11'],coef['q22'],coef['q12']
        def fun(x):
            x1,x2=x; v=g0+b1*x1+b2*x2+q11*x1*x1+q22*x2*x2+q12*x1*x2
            return float(v@H@v)
        def jac(x):
            x1,x2=x; v=g0+b1*x1+b2*x2+q11*x1*x1+q22*x2*x2+q12*x1*x2; Hv=H@v
            return 2*np.array([(b1+2*q11*x1+q12*x2)@Hv,(b2+2*q22*x2+q12*x1)@Hv])
        hlead=math.sqrt(n)*(C.T@(W@g0)); ylead=solve_nnqp(hlead,geom['G']); ys[i]=ylead
        mag=np.sqrt(np.maximum(ylead,0))*n**(-0.25)
        starts=[np.zeros(2)]
        for s1 in (-1,1):
            for s2 in (-1,1): starts.append(np.array([s1*mag[0],s2*mag[1]]))
        starts += [np.array([sx*a,sy*a]) for a in (0.25,) for sx in (-1,1) for sy in (-1,1)]
        best=None
        for st in starts:
            rr=minimize(fun,st,jac=jac,method='L-BFGS-B',bounds=[(-1.5,1.5)]*2,
                        options={'ftol':1e-14,'gtol':1e-9,'maxiter':250})
            if best is None or rr.fun<best.fun: best=rr
        xs[i]=best.x
    return xs,ys


def energy_distance_sample(X,Y):
    # unbiased-ish empirical energy distance; O(n^2), for <=1000 fine.
    from scipy.spatial.distance import cdist
    return float(2*cdist(X,Y).mean() - cdist(X,X).mean() - cdist(Y,Y).mean())


def run(R=300, checkpoints=(2000,5000,10000), burn=2000, seed=20260917, out='/mnt/data/common_ch_feasible_twostep_R300.csv'):
    geom=population_geometry(); t0=time.time()
    print('simulate coefficient paths...')
    outs=simulate_coefficients(R,checkpoints,burn,seed)
    print('first-step identity GMM...')
    prelim=estimate_prelim(outs,checkpoints,R)
    print('re-simulate paths for Omega_hat...')
    omega=accumulate_omega_for_checkpoints(R,checkpoints,burn,seed,prelim)
    print('second-step feasible GMM...')
    feas=feasible_estimates(outs,checkpoints,prelim,omega)

    rows=[]; raw={}
    for n in checkpoints:
        print('oracle comparison n=',n)
        xo,ylead=oracle_x_for_comparison(outs,n,geom,R)
        xf=feas[n]['x']; xp=prelim[n]['x']
        bd_oracle=branch_score_accuracy(outs[n],n,xo,ylead,geom)
        bd_feas=branch_score_accuracy(outs[n],n,xf,ylead,geom)
        scale=n**0.25
        ed_fo=energy_distance_sample(scale*xf,scale*xo)
        rows.append({
            'n':n,'R':R,
            'MAD_prelim_x1':np.median(np.abs(xp[:,0])),'MAD_prelim_x2':np.median(np.abs(xp[:,1])),
            'MAD_feasible_x1':np.median(np.abs(xf[:,0])),'MAD_feasible_x2':np.median(np.abs(xf[:,1])),
            'MAD_scaled_feasible_x1':np.median(np.abs(scale*xf[:,0])),'MAD_scaled_feasible_x2':np.median(np.abs(scale*xf[:,1])),
            'MAD_scaled_oracle_x1':np.median(np.abs(scale*xo[:,0])),'MAD_scaled_oracle_x2':np.median(np.abs(scale*xo[:,1])),
            'median_abs_feasible_minus_oracle_x1':np.median(np.abs(scale*(xf[:,0]-xo[:,0]))),
            'median_abs_feasible_minus_oracle_x2':np.median(np.abs(scale*(xf[:,1]-xo[:,1]))),
            'energy_feasible_vs_oracle_scaled':ed_fo,
            'same_sign_feasible_oracle_x1':np.mean(np.sign(xf[:,0])==np.sign(xo[:,0])),
            'same_sign_feasible_oracle_x2':np.mean(np.sign(xf[:,1])==np.sign(xo[:,1])),
            'BSA1_feasible':bd_feas['BSA1_projected'],'BSA2_feasible':bd_feas['BSA2_projected'],
            'BSA_joint_feasible':bd_feas['BSA_joint_both_projected'],
            'BSA1_oracle':bd_oracle['BSA1_projected'],'BSA2_oracle':bd_oracle['BSA2_projected'],
            'median_Omega_min_eig':np.median(feas[n]['min_eig']),
            'median_Omega_cond':np.median(feas[n]['cond']),
            'p99_Omega_cond':np.quantile(feas[n]['cond'],0.99),
        })
        raw[n]=(xp,xf,xo,ylead)
    df=pd.DataFrame(rows); df.to_csv(out,index=False)
    np.savez(str(Path(out).with_suffix('.npz')), **{f'prelim_{n}':raw[n][0] for n in checkpoints},
             **{f'feasible_{n}':raw[n][1] for n in checkpoints},
             **{f'oracle_{n}':raw[n][2] for n in checkpoints},
             **{f'ylead_{n}':raw[n][3] for n in checkpoints})
    print(df.to_string(index=False)); print('elapsed',time.time()-t0); print('saved',out)
    return df

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--replications',type=int,default=300); ap.add_argument('--burn',type=int,default=2000); ap.add_argument('--seed',type=int,default=20260917); ap.add_argument('--out',type=str,default='/mnt/data/common_ch_feasible_twostep_R300.csv')
    args=ap.parse_args(); run(args.replications,(2000,5000,10000),args.burn,args.seed,args.out)

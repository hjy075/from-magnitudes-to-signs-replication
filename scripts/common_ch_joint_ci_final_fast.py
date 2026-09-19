import math, time, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common_ch_multidirectional_mc_projected import simulate_coefficients, solve_nnqp
from common_ch_feasible_twostep import accumulate_omega_for_checkpoints, regularized_inverse, profile_objective_components


def estimate_fast(coef,W,n,bound=1.5,extra=None):
    d,H=profile_objective_components(coef,W)
    g0,b1,b2=coef['g0'],coef['b1'],coef['b2']; q11,q22,q12=coef['q11'],coef['q22'],coef['q12']
    def v(x):
        x1,x2=x; return g0+b1*x1+b2*x2+q11*x1*x1+q22*x2*x2+q12*x1*x2
    def fun(x):
        vv=v(x); return float(vv@H@vv)
    def jac(x):
        x1,x2=x; vv=v(x); Hv=H@vv
        return 2*np.array([(b1+2*q11*x1+q12*x2)@Hv,(b2+2*q22*x2+q12*x1)@Hv])
    C=np.column_stack([q11,q22]); G=C.T@H@C; h=math.sqrt(n)*(C.T@H@g0)
    starts=[np.zeros(2)]
    try:
        if np.linalg.eigvalsh(G).min()>1e-10:
            M=solve_nnqp(h,G); mag=np.sqrt(np.maximum(M,0))*n**(-0.25)
            s1=(-1,1) if mag[0]>1e-10 else (0,); s2=(-1,1) if mag[1]>1e-10 else (0,)
            starts += [np.array([a*mag[0],b*mag[1]]) for a in s1 for b in s2]
    except Exception:
        pass
    if extra is not None: starts.append(np.asarray(extra,float))
    starts += [np.array([sx*.50,sy*.50]) for sx in (-1,1) for sy in (-1,1)]
    uniq=[]
    for st in starts:
        if not any(np.linalg.norm(st-u)<1e-10 for u in uniq): uniq.append(st)
    best=None
    for st in uniq:
        rr=minimize(fun,st,jac=jac,method='L-BFGS-B',bounds=[(-bound,bound)]*2,
                    options={'ftol':1e-13,'gtol':1e-8,'maxiter':150})
        if best is None or rr.fun<best.fun: best=rr
    xx=best.x; vv=v(xx); alpha=float((d@W@vv)/(d@W@d))
    return xx,alpha,float(best.fun)


def estimate_prelim_fast(outs,checkpoints,R):
    I=np.eye(6); out={}
    for n in checkpoints:
        xs=np.zeros((R,2)); al=np.zeros(R); qs=np.zeros(R)
        for i in range(R):
            coef={k:v[i] for k,v in outs[n].items()}
            xs[i],al[i],qs[i]=estimate_fast(coef,I,n)
        out[n]={'x':xs,'alpha':al,'Q':qs}
    return out


def feasible_fast(outs,checkpoints,prelim,omega):
    R=len(prelim[checkpoints[0]]['x']); out={}
    for n in checkpoints:
        xs=np.zeros((R,2)); al=np.zeros(R); qs=np.zeros(R); mn=np.zeros(R); cond=np.zeros(R)
        for i in range(R):
            W,mineig,_,cn=regularized_inverse(omega[n][i]); coef={k:v[i] for k,v in outs[n].items()}
            xs[i],al[i],qs[i]=estimate_fast(coef,W,n,extra=prelim[n]['x'][i]); mn[i]=mineig; cond[i]=cn
        out[n]={'x':xs,'alpha':al,'Q':qs,'min_eig':mn,'cond':cond}
    return out


def solve_nnqp_batch(h,G):
    B=h.shape[0]; cand=np.zeros((B,4,2)); valid=np.ones((B,4),dtype=bool)
    cand[:,1,0]=np.maximum(-h[:,0]/G[0,0],0); cand[:,2,1]=np.maximum(-h[:,1]/G[1,1],0)
    yb=-(np.linalg.solve(G,h.T)).T; cand[:,3,:]=yb; valid[:,3]=np.all(yb>=0,axis=1)
    vals=np.einsum('bki,ij,bkj->bk',cand,G,cand,optimize=True)+2*np.einsum('bi,bki->bk',h,cand,optimize=True); vals[~valid]=np.inf
    return cand[np.arange(B),np.argmin(vals,axis=1)]


def gaussian_draws_psd(S,B,rng):
    vals,vecs=np.linalg.eigh(.5*(S+S.T)); vals=np.maximum(vals,0)
    return rng.standard_normal((B,len(vals)))@(vecs*np.sqrt(vals)).T


def critical_radius(coef,Omega,Bsim,rng,q=.95):
    W,_,_,_=regularized_inverse(Omega); d=coef['d']; C=np.column_stack([coef['q11'],coef['q22']])
    wd=W@d; H0=W-np.outer(wd,wd)/(d@wd); G=C.T@H0@C
    Z=gaussian_draws_psd(Omega,Bsim,rng); h=Z@H0@C; M=solve_nnqp_batch(h,G)
    a=((Z+M@C.T)@W@d)/(d@W@d); T=np.sqrt(a*a+np.sum(M,axis=1))
    return float(np.quantile(T,q))


def run(R=1000,checkpoints=(2000,5000,10000),burn=2000,seed=20260917,Bsim=1500,bound=1.5,boundary_tol=.01,out='/mnt/data/gmm_paper_final/simulation/common_ch_joint_ci_R1000_raw.csv'):
    t=time.time(); print('coefficients',flush=True); outs=simulate_coefficients(R,checkpoints,burn,seed)
    print('prelim fast',flush=True); prelim=estimate_prelim_fast(outs,checkpoints,R)
    print('omega',flush=True); omega=accumulate_omega_for_checkpoints(R,checkpoints,burn,seed,prelim)
    print('feasible fast',flush=True); feas=feasible_fast(outs,checkpoints,prelim,omega)
    print('inference',flush=True); rng=np.random.default_rng(seed+777); rows=[]; thresh=bound-boundary_tol
    for n in checkpoints:
        for i in range(R):
            coef={k:v[i] for k,v in outs[n].items()}; c=critical_radius(coef,omega[n][i],Bsim,rng)
            x=feas[n]['x'][i]; a=feas[n]['alpha'][i]; U=np.r_[math.sqrt(n)*a,n**.25*x]; ma=float(np.max(np.abs(x)))
            rows.append({'rep':i,'n':n,'Bsim':Bsim,'covered_95':int(np.linalg.norm(U)<=c),'crit_radius_scaled':c,
                         'full_width_alpha':2*c/math.sqrt(n),'full_width_x':2*c/n**.25,'alpha_hat':a,'x1_hat':x[0],'x2_hat':x[1],
                         'max_abs_x':ma,'boundary_hit':int(ma>=thresh),'omega_min_eig':feas[n]['min_eig'][i],'omega_cond':feas[n]['cond'][i]})
        print('done n',n,flush=True)
    df=pd.DataFrame(rows); df.to_csv(out,index=False); print('elapsed',time.time()-t,flush=True); return df

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--replications',type=int,default=1000); ap.add_argument('--burn',type=int,default=2000); ap.add_argument('--seed',type=int,default=20260917); ap.add_argument('--bsim',type=int,default=1500); ap.add_argument('--out',required=True)
    a=ap.parse_args(); run(a.replications,burn=a.burn,seed=a.seed,Bsim=a.bsim,out=a.out)

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

HP = {
    "SEED": 20260810,
    "TOTAL_TRAINING_ITERATIONS": 3000,
    "LEARNING_RATE": 1.0e-4,
    "NETWORK_WIDTH": 64,
    "NETWORK_DEPTH": 3,
    "COLLOCATION_BATCH": 64,
    "EVALUATION_BATCH": 256,
    "GRADIENT_CLIP_NORM": 100.0,
    "DISPLACEMENT_SCALE": 2.0e-4,
    "BOUNDARY_BATCH": 64,
    "INITIAL_BATCH": 64,
    "ATTENTION_DECAY": 0.999,
    "ATTENTION_RATE": 0.01,
    "ATTENTION_EPSILON": 1.0e-12,
    "PRINT_EVERY": 100,
}

torch.set_default_dtype(torch.float64)

@dataclass(frozen=True)
class Physics:
    a: float = 1.0e-2
    b: float = 2.0e-2
    rho0: float = 1050.0
    mu: float = 2.0e5
    kappa: float = 8.0e5
    ell_g: float = 6.0e-4
    eta: float = 35.0
    zeta: float = 80.0
    gamma_g: float = 1.4
    p0: float = 101325.0
    DeltaP: float = 2.4e4
    Omega: float = 9.0e3
    nRamp: float = 4.0
    nPeriod: float = 30.0
    N: int = 121

    @property
    def g(self): return self.mu*self.ell_g**2
    @property
    def Tforce(self): return 2.0*math.pi/self.Omega
    @property
    def tauRamp(self): return self.nRamp*self.Tforce
    @property
    def tEnd(self): return self.nPeriod*self.Tforce

FIELD_NAMES = [
    "u", "v", "acceleration", "current_radius", "lambda_r", "lambda_theta",
    "green_strain_r", "green_strain_theta", "hencky_strain_r", "hencky_strain_theta",
    "J", "logJ", "logJ_t", "logJ_r", "q_gradient", "f_gradient",
    "P_r_elastic", "P_theta_elastic", "P_r_viscous", "P_theta_viscous",
    "P_r_local", "P_theta_local", "P_r_gradient", "P_theta_gradient",
    "S_r_generalized", "S_theta_generalized", "sigma_rr_local", "sigma_tt_local",
    "delta_sigma_local", "sigma_mean_local", "sigma_von_mises_local",
    "kinetic_energy_density", "elastic_energy_density", "gradient_energy_density",
    "stored_energy_density", "dissipation_density",
]

CURVE_NAMES = [
    "inner_radius","outer_radius","cavity_volume","cavity_pressure_absolute",
    "cavity_pressure_gauge","outer_pressure_gauge","u_inner","u_outer","v_inner","v_outer",
    "a_inner","a_outer","kinetic_energy","elastic_energy","gradient_energy","stored_energy",
    "mechanical_energy","dissipation_rate","cumulative_dissipation","min_J","max_J","max_abs_u",
    "max_abs_v","max_abs_q","max_abs_f_gradient","max_abs_sigma_rr","max_abs_sigma_tt",
]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def process_peak_memory_bytes():
    class PMC(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),
          ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
          ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
          ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
          ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
    c=PMC(); c.cb=ctypes.sizeof(c)
    getp=ctypes.windll.kernel32.GetCurrentProcess; getp.restype=ctypes.c_void_p
    getm=ctypes.windll.psapi.GetProcessMemoryInfo
    getm.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]; getm.restype=ctypes.c_int
    ok=getm(getp(),ctypes.byref(c),c.cb)
    return int(c.PeakWorkingSetSize) if ok else 0

def grad(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]

class MLP(nn.Module):
    def __init__(self,nin,nout,width,depth):
        super().__init__(); layers=[nn.Linear(nin,width),nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,nout)]; self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)

class RPINN(nn.Module):
    def __init__(self,p):
        super().__init__(); self.p=p
        self.U0=HP["DISPLACEMENT_SCALE"]; self.V0=self.U0*p.Omega
        self.net=MLP(2,2,HP["NETWORK_WIDTH"],HP["NETWORK_DEPTH"])
    def forward(self,r,t):
        x=2*(r-self.p.a)/(self.p.b-self.p.a)-1
        s=2*t/self.p.tEnd-1
        y=self.net(torch.cat((x,s),1))
        return self.U0*y[:,0:1],self.V0*y[:,1:2]

def pressure_torch(t,u,p):
    Rc=p.a+u[:,0]; pc=p.p0*((p.a/torch.clamp(Rc,min=.05*p.a))**3)**p.gamma_g-p.p0
    po=p.DeltaP*(1-torch.exp(-(t[:,0]/p.tauRamp)**2))*torch.sin(p.Omega*t[:,0])
    return pc,po

def pde_state(cell,p,r,t):
    u,v=cell(r,t); ur=grad(u,r); vr=grad(v,r); ut=grad(u,t); vt=grad(v,t)
    lr=1+ur; lt=1+u/r; J=torch.clamp(lr*lt**2,min=.2); e=torch.log(J)
    edot=vr/lr+2*v/(r+u); er=grad(e,r); q=p.g*er; M=grad(r*r*q,r)
    Pre=p.mu*(lr-1/lr)+p.kappa*e/lr; Pte=p.mu*(lt-1/lt)+p.kappa*e/lt
    Pr=Pre+p.eta*vr+p.zeta*edot/lr; Pt=Pte+p.eta*v/r+p.zeta*edot/lt
    Sr=Pr-M/(lr*r*r); St=Pt-M/(r*(r+u))
    kin=(ut-v)/cell.V0
    mom=(p.rho0*vt-(grad(Sr,r)+2*(Sr-St)/r))/(p.rho0*cell.V0*p.Omega)
    return {"u":u,"v":v,"kin":kin,"mom":mom,"er":er,"Sr":Sr,"lt":lt}

def attention_pde_loss(model,p,r0,t0,attention_kinematic,attention_momentum):
    r=r0.detach().clone().requires_grad_(True); t=t0.detach().clone().requires_grad_(True)
    f=pde_state(model,p,r,t)
    with torch.no_grad():
        nk=torch.abs(f["kin"].detach()); nm=torch.abs(f["mom"].detach())
        attention_kinematic.mul_(HP["ATTENTION_DECAY"]).add_(HP["ATTENTION_RATE"]*nk/(torch.max(nk)+HP["ATTENTION_EPSILON"]))
        attention_momentum.mul_(HP["ATTENTION_DECAY"]).add_(HP["ATTENTION_RATE"]*nm/(torch.max(nm)+HP["ATTENTION_EPSILON"]))
    lk=torch.mean((1+attention_kinematic)*f["kin"]**2)
    lm=torch.mean((1+attention_momentum)*f["mom"]**2)
    return lk+lm,{"pde_kinematic":lk,"pde_momentum":lm,
      "attention_kinematic_max":torch.max(attention_kinematic),"attention_momentum_max":torch.max(attention_momentum)}

def boundary_loss(model,p,nb,device):
    ni=nb//2; rb=torch.cat((torch.full((ni,1),p.a,device=device),torch.full((nb-ni,1),p.b,device=device))).requires_grad_(True)
    tb=(p.tEnd*torch.rand(nb,1,device=device)).requires_grad_(True)
    b=pde_state(model,p,rb,tb); pc,po=pressure_torch(tb,b["u"],p)
    target=torch.cat((-pc[:ni],-po[ni:]))[:,None]*b["lt"]**2
    traction=torch.mean(((b["Sr"]-target)/p.mu)**2); micro=torch.mean((b["er"]*(p.b-p.a))**2)
    return traction+micro,{"boundary_traction":traction,"boundary_micro":micro}

def initial_loss(model,p,device):
    r=(p.a+(p.b-p.a)*torch.rand(HP["INITIAL_BATCH"],1,device=device)).requires_grad_(True)
    t=torch.zeros_like(r).requires_grad_(True); u,v=model(r,t)
    lu=torch.mean((u/model.U0)**2); lv=torch.mean((v/model.V0)**2)
    return lu+lv,{"initial_u":lu,"initial_v":lv}

def plot_losses(history,out):
    ep=np.array([x["epoch"] for x in history]); keys=[k for k in history[-1] if k not in ("epoch","seconds")]
    colors=plt.cm.jet(np.linspace(.05,.95,len(keys)))
    for logy,name in ((False,"loss_normal.png"),(True,"loss_log.png")):
        fig,ax=plt.subplots(figsize=(8,5.5))
        for c,k in zip(colors,keys): ax.plot(ep,[max(x.get(k,np.nan),1e-300) for x in history],color=c,label=k,lw=1.1)
        if logy: ax.set_yscale("log")
        ax.set(xlabel="Epoch",ylabel="Loss",title="Training loss"); ax.grid(True,alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(out/name,dpi=300); plt.close(fig)

def train(root,device):
    p=Physics(); seed_all(HP["SEED"]); model=RPINN(p).to(device); out=root/"results"; out.mkdir(exist_ok=True)
    opt=torch.optim.Adam(model.parameters(),lr=HP["LEARNING_RATE"]); history=[]; times=[]; tic_all=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    r0=p.a+(p.b-p.a)*torch.rand(HP["COLLOCATION_BATCH"],1,device=device)
    t0=p.tEnd*torch.rand(HP["COLLOCATION_BATCH"],1,device=device)
    attention_kinematic=torch.zeros_like(r0); attention_momentum=torch.zeros_like(r0)
    for ep in range(1,HP["TOTAL_TRAINING_ITERATIONS"]+1):
        tic=time.perf_counter(); opt.zero_grad(set_to_none=True)
        lp,pde_parts=attention_pde_loss(model,p,r0,t0,attention_kinematic,attention_momentum)
        lb,boundary_parts=boundary_loss(model,p,HP["BOUNDARY_BATCH"],device)
        li,initial_parts=initial_loss(model,p,device); total=lp+lb+li
        parts={**pde_parts,**boundary_parts,**initial_parts}
        if not torch.isfinite(total): raise RuntimeError(f"non-finite loss at iteration {ep}")
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),HP["GRADIENT_CLIP_NORM"]); opt.step()
        if device.type=="cuda": torch.cuda.synchronize(device)
        dt=time.perf_counter()-tic; times.append(dt)
        row={"epoch":ep,"total":float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()},"seconds":dt}; history.append(row)
        if ep==1 or ep%HP["PRINT_EVERY"]==0: print(ep,row["total"],f"{dt:.6f}s",flush=True)
    total_seconds=time.perf_counter()-tic_all
    assert len(history)==HP["TOTAL_TRAINING_ITERATIONS"]
    torch.save({"state_dict":model.state_dict(),"physics":asdict(p),"hp":HP},out/"model.pt")
    np.savetxt(out/"final_residual_attention.txt",np.column_stack((r0.detach().cpu().numpy()[:,0],
      t0.detach().cpu().numpy()[:,0],attention_kinematic.detach().cpu().numpy()[:,0],
      attention_momentum.detach().cpu().numpy()[:,0])),fmt="%.16e",delimiter="\t")
    keys=sorted(set().union(*(x.keys() for x in history)))
    with open(out/"loss_history.txt","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter="\t"); w.writeheader(); w.writerows(history)
    allocated=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else process_peak_memory_bytes()
    reserved=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else allocated
    timing={"TotalTrainingIterations":len(history),"TotalTrainingSeconds":total_seconds,
      "MeanEpochSeconds":statistics.mean(times),"StdEpochSeconds":statistics.pstdev(times),
      "MedianEpochSeconds":statistics.median(times),"MinEpochSeconds":min(times),"MaxEpochSeconds":max(times),
      "PeakTrainMemoryAllocatedBytes":allocated,"PeakTrainMemoryAllocatedMiB":allocated/2**20,
      "PeakTrainMemoryReservedBytes":reserved,"PeakTrainMemoryReservedMiB":reserved/2**20,
      "FinalTotalLoss":history[-1]["total"],"MemoryMetricSource":"torch_cuda" if device.type=="cuda" else "process_peak_working_set_cpu"}
    (out/"training_metadata.json").write_text(json.dumps({**timing,"HP":HP},indent=2),encoding="utf-8")
    plot_losses(history,out); return model,timing

def d1_np(f,dr):
    z=np.empty_like(f); z[:,1:-1]=(f[:,2:]-f[:,:-2])/(2*dr)
    z[:,0]=(-3*f[:,0]+4*f[:,1]-f[:,2])/(2*dr); z[:,-1]=(3*f[:,-1]-4*f[:,-2]+f[:,-3])/(2*dr)
    return z

def fields_numpy(U,V,t,p):
    r=np.linspace(p.a,p.b,p.N); dr=r[1]-r[0]; ur=d1_np(U,dr); vr=d1_np(V,dr)
    lr=1+ur; lt=1+U/r; J=lr*lt**2; e=np.log(J); edot=vr/lr+2*V/(r+U)
    er=d1_np(e,dr); er[:,0]=0; er[:,-1]=0; q=p.g*er; M=d1_np(r*r*q,dr)
    Pre=p.mu*(lr-1/lr)+p.kappa*e/lr; Pte=p.mu*(lt-1/lt)+p.kappa*e/lt
    Prv=p.eta*vr+p.zeta*edot/lr; Ptv=p.eta*V/r+p.zeta*edot/lt; Pr=Pre+Prv; Pt=Pte+Ptv
    Prg=-M/(lr*r*r); Ptg=-M/(r*(r+U)); Sr=Pr+Prg; St=Pt+Ptg
    fg=d1_np(Prg,dr)+2*(Prg-Ptg)/r; srr=lr/J*Pr; stt=lt/J*Pt
    We=.5*p.mu*(lr**2+2*lt**2-3-2*e)+.5*p.kappa*e**2; Wg=.5*p.g*er**2
    K=.5*p.rho0*V**2; D=p.eta*(vr**2+2*(V/r)**2)+p.zeta*edot**2
    rf=np.empty(p.N+1); rf[0]=p.a; rf[-1]=p.b; rf[1:-1]=.5*(r[:-1]+r[1:])
    Rc=p.a+U[:,0]; pc=p.p0*(p.a/np.maximum(Rc,.05*p.a))**(3*p.gamma_g)-p.p0
    po=p.DeltaP*(1-np.exp(-(t/p.tauRamp)**2))*np.sin(p.Omega*t)
    sf=np.empty((len(t),p.N+1)); sf[:,0]=-pc*lt[:,0]**2; sf[:,-1]=-po*lt[:,-1]**2; sf[:,1:-1]=.5*(Sr[:,:-1]+Sr[:,1:])
    rm,rp=rf[:-1],rf[1:]; acc=(rp**2*sf[:,1:]-rm**2*sf[:,:-1]-2*r*St*(rp-rm))/(p.rho0*((rp**3-rm**3)/3))
    vals=[U,V,acc,r+U,lr,lt,.5*(lr**2-1),.5*(lt**2-1),np.log(lr),np.log(lt),J,e,edot,er,q,fg,
          Pre,Pte,Prv,Ptv,Pr,Pt,Prg,Ptg,Sr,St,srr,stt,stt-srr,(srr+2*stt)/3,np.abs(srr-stt),K,We,Wg,We+Wg,D]
    return dict(zip(FIELD_NAMES,vals))

def curves_numpy(F,t,p):
    r=np.linspace(p.a,p.b,p.N); U,V,A=F["u"],F["v"],F["acceleration"]
    Rc=p.a+U[:,0]; Ro=p.b+U[:,-1]; Vc=4*math.pi*Rc**3/3
    pc_abs=p.p0*(p.a/Rc)**(3*p.gamma_g); pc=pc_abs-p.p0
    po=p.DeltaP*(1-np.exp(-(t/p.tauRamp)**2))*np.sin(p.Omega*t)
    integ=lambda z: 4*math.pi*np.trapezoid(z*r*r,r,axis=1)
    Ek=integ(F["kinetic_energy_density"]); Ee=integ(F["elastic_energy_density"])
    Eg=integ(F["gradient_energy_density"]); Es=Ee+Eg; Em=Ek+Es
    Ddot=integ(F["dissipation_density"]); Ec=np.r_[0,np.cumsum(.5*(Ddot[1:]+Ddot[:-1])*np.diff(t))]
    vals=[Rc,Ro,Vc,pc_abs,pc,po,U[:,0],U[:,-1],V[:,0],V[:,-1],A[:,0],A[:,-1],Ek,Ee,Eg,Es,Em,Ddot,Ec,
          F["J"].min(1),F["J"].max(1),np.abs(U).max(1),np.abs(V).max(1),np.abs(F["q_gradient"]).max(1),
          np.abs(F["f_gradient"]).max(1),np.abs(F["sigma_rr_local"]).max(1),np.abs(F["sigma_tt_local"]).max(1)]
    return dict(zip(CURVE_NAMES,vals))

def predict_uv(model,r,t,device):
    U=np.empty((len(t),len(r))); V=np.empty_like(U); tic=time.perf_counter(); model.eval()
    for j in range(0,len(t),HP["EVALUATION_BATCH"]):
        ix=np.arange(j,min(j+HP["EVALUATION_BATCH"],len(t))); nr=len(r)
        rr=torch.tensor(np.tile(r,(len(ix),1)).reshape(-1,1),device=device)
        tt=torch.tensor(t[ix,None],device=device).repeat_interleave(nr,0)
        with torch.no_grad(): u,v=model(rr,tt)
        U[ix]=u.cpu().numpy().reshape(len(ix),nr); V[ix]=v.cpu().numpy().reshape(len(ix),nr)
    return U,V,time.perf_counter()-tic

def read_truth(reference_dir,name,nr,nt):
    idx=FIELD_NAMES.index(name)+1; z=np.loadtxt(reference_dir/f"field_{idx:02d}_{name}.txt",usecols=2)
    return z.reshape(nr,nt).T

def metric(pred,true):
    e=pred-true; mse=float(np.mean(e**2)); den=float(np.linalg.norm(true.ravel()))
    return {"RMSE":math.sqrt(mse),"MSE":mse,"MAE":float(np.mean(np.abs(e))),
            "L2":float(np.linalg.norm(e.ravel())/(den+1e-300)),"MaxError":float(np.max(np.abs(e)))}

def save_triplet(path,r,t,z):
    R,T=np.meshgrid(r,t); np.savetxt(path,np.column_stack((R.ravel(),T.ravel(),z.ravel())),fmt="%.16e",delimiter="\t")

def heatmap(path,r,t,z,title,vmin=None,vmax=None):
    fig,ax=plt.subplots(figsize=(8.2,5.6)); im=ax.imshow(z,origin="lower",aspect="auto",extent=[r[0],r[-1],t[0],t[-1]],cmap="jet",vmin=vmin,vmax=vmax)
    ax.set(xlabel="r (m)",ylabel="t (s)",title=title); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def curve_plot(path,t,y,title,ylim=None):
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(t,y,color=plt.cm.jet(.72),lw=1.15)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set(xlabel="t (s)",ylabel="value",title=title); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def load_model(root,device):
    p=Physics(); model=RPINN(p).to(device); ck=torch.load(root/"results"/"model.pt",map_location=device,weights_only=False)
    model.load_state_dict(ck["state_dict"]); return model

def evaluate(root,model,device,timing=None,metrics_only=False):
    p=Physics(); out=root/"results"; ref=root.parent/"FVM"/"spherical_gradient_results"
    sample=np.loadtxt(ref/"field_01_u.txt",usecols=(0,1)); r=np.unique(sample[:,0]); t=np.unique(sample[:,1])
    U,V,eval_seconds=predict_uv(model,r,t,device); pred_fields=fields_numpy(U,V,t,p); rows=[]
    field_dir=out/"fields"; field_dir.mkdir(exist_ok=True)
    for name in FIELD_NAMES:
        truth=read_truth(ref,name,len(r),len(t)); pred=pred_fields[name]; err=np.abs(pred-truth); rows.append({"Field":name,**metric(pred,truth)})
        if not metrics_only:
            lo=min(float(truth.min()),float(pred.min())); hi=max(float(truth.max()),float(pred.max()))
            save_triplet(field_dir/f"{name}_truth.txt",r,t,truth); save_triplet(field_dir/f"{name}_prediction.txt",r,t,pred); save_triplet(field_dir/f"{name}_maxerror.txt",r,t,err)
            heatmap(field_dir/f"{name}_truth.png",r,t,truth,f"{name}: FVM truth",lo,hi)
            heatmap(field_dir/f"{name}_prediction.png",r,t,pred,f"{name}: prediction",lo,hi)
            heatmap(field_dir/f"{name}_maxerror.png",r,t,err,f"{name}: absolute error",0,float(err.max()) or 1)
            em=err.max(axis=1); np.savetxt(field_dir/f"{name}_maxerror_over_time.txt",np.c_[t,np.zeros_like(t),em],fmt="%.16e",delimiter="\t")
            curve_plot(field_dir/f"{name}_maxerror_over_time.png",t,em,f"{name}: MaxError")
    curve_dir=out/"curves"; curve_dir.mkdir(exist_ok=True); pred_curves=curves_numpy(pred_fields,t,p)
    for j,name in enumerate(CURVE_NAMES,1):
        truth=np.loadtxt(ref/f"curve_{j:02d}_{name}.txt",usecols=2); pred=pred_curves[name]; err=np.abs(pred-truth)
        rows.append({"Field":"curve:"+name,**metric(pred,truth)})
        if not metrics_only:
            z=np.zeros_like(t); lim=(min(float(truth.min()),float(pred.min())),max(float(truth.max()),float(pred.max())))
            if lim[0]==lim[1]: lim=(lim[0]-1,lim[1]+1)
            np.savetxt(curve_dir/f"{name}_truth.txt",np.c_[t,z,truth],fmt="%.16e",delimiter="\t")
            np.savetxt(curve_dir/f"{name}_prediction.txt",np.c_[t,z,pred],fmt="%.16e",delimiter="\t")
            np.savetxt(curve_dir/f"{name}_maxerror.txt",np.c_[t,z,err],fmt="%.16e",delimiter="\t")
            curve_plot(curve_dir/f"{name}_truth.png",t,truth,f"{name}: FVM truth",lim)
            curve_plot(curve_dir/f"{name}_prediction.png",t,pred,f"{name}: prediction",lim)
            curve_plot(curve_dir/f"{name}_maxerror.png",t,err,f"{name}: absolute error")
    meta={}; mp=out/"training_metadata.json"
    if mp.exists(): meta=json.loads(mp.read_text(encoding="utf-8"))
    if timing: meta.update(timing)
    meta.update(EvaluationGrid=f"{len(r)} x {len(t)} (r x t)",EvaluationPointCount=len(r)*len(t),EvaluationSeconds=eval_seconds)
    keys=["EvaluationGrid","EvaluationPointCount","TotalTrainingIterations","TotalTrainingSeconds","EvaluationSeconds",
      "MeanEpochSeconds","StdEpochSeconds","MedianEpochSeconds","MinEpochSeconds","MaxEpochSeconds",
      "PeakTrainMemoryAllocatedBytes","PeakTrainMemoryAllocatedMiB","PeakTrainMemoryReservedBytes",
      "PeakTrainMemoryReservedMiB","FinalTotalLoss","MemoryMetricSource"]
    with open(out/"metrics_and_timing.txt","w",encoding="utf-8") as f:
        for k in keys: f.write(f"{k}\t{meta.get(k,'NA')}\n")
        f.write("\nField\tRMSE\tMSE\tMAE\tL2\tMaxError\n")
        for x in rows: f.write("\t".join([x["Field"]]+[f"{x[k]:.16e}" for k in ("RMSE","MSE","MAE","L2","MaxError")])+"\n")
    print(f"evaluated R-PINN: {len(rows)} fields, {eval_seconds:.6f}s")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("train","evaluate","all","smoke"),default="all")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--metrics-only",action="store_true")
    args=ap.parse_args(); root=Path(__file__).resolve().parent; device=torch.device(args.device)
    if args.mode=="smoke":
        p=Physics(); seed_all(HP["SEED"]); model=RPINN(p).to(device); opt=torch.optim.Adam(model.parameters(),lr=HP["LEARNING_RATE"])
        r0=p.a+(p.b-p.a)*torch.rand(2,1,device=device); t0=p.tEnd*torch.rand(2,1,device=device)
        ak=torch.zeros_like(r0); am=torch.zeros_like(r0); opt.zero_grad(set_to_none=True)
        lp,_=attention_pde_loss(model,p,r0,t0,ak,am); lb,_=boundary_loss(model,p,2,device); li,_=initial_loss(model,p,device)
        total=lp+lb+li; total.backward(); opt.step(); print(float(total.detach()),float(ak.max()),float(am.max())); return
    model=None; timing=None
    if args.mode in ("train","all"): model,timing=train(root,device)
    if args.mode in ("evaluate","all"):
        if model is None: model=load_model(root,device)
        evaluate(root,model,device,timing,args.metrics_only)

if __name__=="__main__": main()

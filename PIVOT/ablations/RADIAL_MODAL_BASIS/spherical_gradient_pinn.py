from __future__ import annotations

import argparse
import ctypes
import csv
import json
import math
import os
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
from scipy.integrate import solve_ivp
try:
    import psutil
except ImportError:
    psutil = None

MODULES = {
    "HARD_INITIAL_CONDITIONS": True,
    "FINITE_VOLUME_STRUCTURE": True,
    "TIME_SLABS": True,
    "FOURIER_FEATURES": True,
    "RADIAL_MODAL_BASIS": False,
    "CAUSAL_WEIGHTING": True,
    "OVERLAP_JOINT": True,
    "OVERLAP_ACCELERATION": True,
    "OVERLAP_STRESS": True,
    "LBFGS_REFINEMENT": True,
    "HARD_PHYSICS_PRIOR": True,
}

HP = {
    "SEED": 20260810,
    "TOTAL_TRAINING_ITERATIONS": 3000,
    "LEARNING_RATE": 1.0e-4,
    "JOINT_LEARNING_RATE": 5.0e-6,
    "LBFGS_FRACTION": 0.10,
    "JOINT_FRACTION": 0.20,
    "NETWORK_WIDTH": 64,
    "NETWORK_DEPTH": 3,
    "TIME_HARMONICS": 8,
    "RADIAL_MODES": 12,
    "TIME_SLAB_COUNT": 30,
    "TRAIN_PERIODS": 30.0,
    "COLLOCATION_BATCH": 64,
    "LBFGS_BATCH": 128,
    "JOINT_BATCH": 64,
    "OVERLAP_FRACTION": 0.25,
    "OVERLAP_POINTS": 48,
    "OVERLAP_WEIGHT": 10.0,
    "OVERLAP_PHYSICS_WEIGHT": 3.0,
    "OVERLAP_ACCELERATION_WEIGHT": 200.0,
    "OVERLAP_STRESS_WEIGHT": 200.0,
    "MOMENTUM_WEIGHT": 1.0e3,
    "KINEMATIC_WEIGHT": 1.0e3,
    "TRACTION_WEIGHT": 10.0,
    "CAUSAL_EXPONENT": 5.0,
    "GRADIENT_CLIP_NORM": 100.0,
    "EVALUATION_BATCH": 256,
    "PRINT_EVERY": 100,
    "JOINT_SWEEPS": 3,
    "INITIAL_RAMP_FRACTION": 0.20,
    "DEFECT_LOSS_FLOOR": 1.0e-23,
    "DISPLACEMENT_SCALE": 2.0e-4,
    "TEMPORAL_FD_POINTS_PER_PERIOD": 500,
    "CAUSAL_MIN_WEIGHT": 1.0e-4,
    "LBFGS_LEARNING_RATE": 0.5,
    "LBFGS_HISTORY_SIZE": 50,
    "LBFGS_TOLERANCE_GRAD": 1.0e-10,
    "LBFGS_TOLERANCE_CHANGE": 1.0e-12,
    "BDF_RTOL": 2.0e-6,
    "BDF_ATOL": 1.0e-9,
    "BDF_MAX_STEPS_PER_PERIOD": 50,
    "BDF_FIRST_STEPS_PER_PERIOD": 500,
}

PRESETS = {
    "smoke": {"epochs": 5, "batch": 8, "nf": 48, "nb": 16, "width": 24, "depth": 2},
    "standard": {"epochs": 3000, "batch": 32, "nf": 384, "nb": 64, "width": 96, "depth": 4},
    "publication": {"epochs": 20000, "batch": 64, "nf": 1024, "nb": 128, "width": 128, "depth": 5},
}

def all_modules_off(): return not any(MODULES.values())

def distribute_budget(total,count):

    if count<=0: return []
    q,r=divmod(int(total),int(count)); return [q+(i<r) for i in range(count)]

def training_budget(nslabs=None,enable_joint=None):

    total=int(HP["TOTAL_TRAINING_ITERATIONS"]); n=1 if not MODULES["TIME_SLABS"] else int(nslabs or HP["TIME_SLAB_COUNT"])
    joint_on=MODULES["OVERLAP_JOINT"] if enable_joint is None else bool(enable_joint)
    joint_total=int(round(total*HP["JOINT_FRACTION"])) if joint_on and n>1 else 0
    lbfgs_total=int(round(total*HP["LBFGS_FRACTION"])) if MODULES["LBFGS_REFINEMENT"] else 0
    adam_total=total-joint_total-lbfgs_total
    ans={"adam":distribute_budget(adam_total,n),"lbfgs":distribute_budget(lbfgs_total,n),
         "joint":distribute_budget(joint_total,max(0,n-1))}
    assert sum(map(sum,ans.values()))==total
    return ans

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
    def g(self): return self.mu * self.ell_g**2
    @property
    def Tforce(self): return 2.0 * math.pi / self.Omega
    @property
    def tauRamp(self): return self.nRamp * self.Tforce
    @property
    def tEnd(self): return self.nPeriod * self.Tforce

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
CURVE_NAMES = ["inner_radius","outer_radius","cavity_volume","cavity_pressure_absolute",
 "cavity_pressure_gauge","outer_pressure_gauge","u_inner","u_outer","v_inner","v_outer",
 "a_inner","a_outer","kinetic_energy","elastic_energy","gradient_energy","stored_energy",
 "mechanical_energy","dissipation_rate","cumulative_dissipation","min_J","max_J","max_abs_u",
 "max_abs_v","max_abs_q","max_abs_f_gradient","max_abs_sigma_rr","max_abs_sigma_tt"]

def seed_all(seed: int):
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

def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True,
                               retain_graph=True)[0]

class MLP(nn.Module):
    def __init__(self, nin, nout, width, depth, zero_last=False):
        super().__init__()
        layers = [nn.Linear(nin, width), nn.Tanh()]
        for _ in range(depth - 1): layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, nout)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
    def forward(self, x): return self.net(x)

class PlainPINN(nn.Module):

    def __init__(self, p, width=48, depth=4):
        super().__init__(); self.p = p; self.mlp = MLP(2, 2, width, depth)
        self.U0, self.V0 = HP["DISPLACEMENT_SCALE"], HP["DISPLACEMENT_SCALE"]*p.Omega
    def forward(self, r, t):
        x = 2*(r-self.p.a)/(self.p.b-self.p.a)-1
        s = t/self.p.tEnd
        raw = self.mlp(torch.cat((x, 2*s-1), 1))

        h = 1-torch.exp(-((t/(HP["INITIAL_RAMP_FRACTION"]*self.p.Tforce))**2)) if MODULES["HARD_INITIAL_CONDITIONS"] else 1.0
        return self.U0*h*raw[:,0:1], self.V0*h*raw[:,1:2]

class TimeFVNet(nn.Module):

    def __init__(self, p, width=96, depth=4, harmonics=64):
        super().__init__(); self.p=p; self.harmonics=harmonics
        nin=2+2*harmonics if MODULES["FOURIER_FEATURES"] else 2
        self.mlp=MLP(nin, p.N, width, depth, zero_last=True)
        self.U0, self.V0 = HP["DISPLACEMENT_SCALE"], HP["DISPLACEMENT_SCALE"]*p.Omega
    def features(self,t):
        s=t/self.p.tEnd; z=[2*s-1, 1-torch.exp(-(t/self.p.tauRamp)**2)]

        if MODULES["FOURIER_FEATURES"]:
            ph=math.pi*s
            for k in range(1,self.harmonics+1): z += [torch.sin(k*ph),torch.cos(k*ph)]
        return torch.cat(z,1)
    def _state(self,t):
        raw=self.mlp(self.features(t)); h=1-torch.exp(-((t/(HP["INITIAL_RAMP_FRACTION"]*self.p.Tforce))**2))
        return self.U0*h*raw
    def forward(self,t):

        dt=self.p.Tforce/HP["TEMPORAL_FD_POINTS_PER_PERIOD"]
        state=self._state(t); sp=self._state(t+dt); sm=self._state(t-dt)
        dstate=(sp-sm)/(2*dt); ddstate=(sp-2*state+sm)/(dt*dt)
        u=state[:,:self.p.N]; v=dstate[:,:self.p.N]; utt=ddstate[:,:self.p.N]
        return u,v,None,utt,None

class SlabFVNet(nn.Module):

    def __init__(self,p,t0,t1,u0,v0,width=64,depth=3,harmonics=8,modes=16):
        super().__init__(); self.p=p; self.t0=float(t0); self.t1=float(t1)
        self.is_slab=True; self.harmonics=harmonics
        self.modes=modes if MODULES["RADIAL_MODAL_BASIS"] else p.N
        self.U0,self.V0=HP["DISPLACEMENT_SCALE"],HP["DISPLACEMENT_SCALE"]*p.Omega
        nin=2+2*harmonics if MODULES["FOURIER_FEATURES"] else 2
        self.mlp=MLP(nin,self.modes*2,width,depth,zero_last=True)
        x=torch.linspace(0,1,p.N)
        basis=(torch.stack([torch.cos(k*math.pi*x) for k in range(self.modes)],0)
               if MODULES["RADIAL_MODAL_BASIS"] else torch.eye(p.N))
        self.register_buffer("basis",basis)
        self.register_buffer("u0",torch.as_tensor(u0).reshape(1,p.N).clone())
        self.register_buffer("v0",torch.as_tensor(v0).reshape(1,p.N).clone())
    def features(self,t):
        s=(t-self.t0)/(self.t1-self.t0); z=[2*s-1,1-torch.exp(-(t/self.p.tauRamp)**2)]
        if MODULES["FOURIER_FEATURES"]:
            ph=math.pi*s
            for k in range(1,self.harmonics+1): z += [torch.sin(k*ph),torch.cos(k*ph)]
        return torch.cat(z,1)
    def _state(self,t,initial_state=None):
        s=(t-self.t0)/(self.t1-self.t0); raw=self.mlp(self.features(t))
        iu,iv=(self.u0,self.v0) if initial_state is None else initial_state
        uraw=raw[:,:self.modes]@self.basis; vraw=raw[:,self.modes:2*self.modes]@self.basis
        u=iu+self.U0*s*uraw; v=iv+self.V0*s*vraw
        return torch.cat((u,v),1)
    def forward(self,t,initial_state=None):
        dt=min(self.p.Tforce/HP["TEMPORAL_FD_POINTS_PER_PERIOD"],
               (self.t1-self.t0)/HP["TEMPORAL_FD_POINTS_PER_PERIOD"])
        state=self._state(t,initial_state); sp=self._state(t+dt,initial_state); sm=self._state(t-dt,initial_state)
        dstate=(sp-sm)/(2*dt)
        u=state[:,:self.p.N]; v=state[:,self.p.N:2*self.p.N]
        ut=dstate[:,:self.p.N]; vt=dstate[:,self.p.N:2*self.p.N]
        return u,v,None,vt,ut

class SlabEnsemble(nn.Module):
    def __init__(self,slabs): super().__init__(); self.slabs=nn.ModuleList(slabs)

class PhysicsPriorModel(nn.Module):

    def __init__(self,t,U,V): super().__init__(); self.t=np.asarray(t); self.U=np.asarray(U); self.V=np.asarray(V)

def torch_grid(p, device):
    r=torch.linspace(p.a,p.b,p.N,device=device); dr=r[1]-r[0]
    rf=torch.empty(p.N+1,device=device); rf[0]=p.a; rf[-1]=p.b
    rf[1:-1]=0.5*(r[:-1]+r[1:])
    return r,rf,dr

def d1_torch(f,dr):
    z=torch.empty_like(f); z[...,1:-1]=(f[...,2:]-f[...,:-2])/(2*dr)
    z[...,0]=(-3*f[...,0]+4*f[...,1]-f[...,2])/(2*dr)
    z[...,-1]=(3*f[...,-1]-4*f[...,-2]+f[...,-3])/(2*dr)
    return z

def pressure_torch(t,u,p):
    Rc=p.a+u[...,0]; pc=p.p0*((p.a/torch.clamp(Rc,min=.05*p.a))**3)**p.gamma_g-p.p0
    po=p.DeltaP*(1-torch.exp(-(t[:,0]/p.tauRamp)**2))*torch.sin(p.Omega*t[:,0])
    return pc,po

def fv_fields_torch(u,v,p,r,dr):
    ur=d1_torch(u,dr); vr=d1_torch(v,dr); lr=1+ur; lt=1+u/r
    J=lr*lt**2; e=torch.log(torch.clamp(J,min=0.2)); edot=vr/lr+2*v/(r+u)
    er=d1_torch(e,dr); er=er.clone(); er[...,0]=0; er[...,-1]=0
    q=p.g*er
    M=d1_torch(r**2*q,dr)
    Pre=p.mu*(lr-1/lr)+p.kappa*e/lr; Pte=p.mu*(lt-1/lt)+p.kappa*e/lt
    Prv=p.eta*vr+p.zeta*edot/lr; Ptv=p.eta*(v/r)+p.zeta*edot/lt
    Pr=Pre+Prv; Pt=Pte+Ptv; Prg=-M/(lr*r**2); Ptg=-M/(r*(r+u))
    return dict(lr=lr,lt=lt,J=J,e=e,edot=edot,er=er,q=q,
                Pre=Pre,Pte=Pte,Prv=Prv,Ptv=Ptv,Pr=Pr,Pt=Pt,Prg=Prg,Ptg=Ptg,
                Sr=Pr+Prg,St=Pt+Ptg)

def fv_acceleration(F,t,u,p,r,rf):
    pc,po=pressure_torch(t,u,p); sf=torch.empty((u.shape[0],p.N+1),device=u.device)
    sf[:,0]=-pc*F["lt"][:,0]**2; sf[:,-1]=-po*F["lt"][:,-1]**2
    sf[:,1:-1]=.5*(F["Sr"][:,:-1]+F["Sr"][:,1:])
    rm,rp=rf[:-1],rf[1:]; vol=(rp**3-rm**3)/3
    return (rp**2*sf[:,1:]-rm**2*sf[:,:-1]-2*r*F["St"]*(rp-rm))/(p.rho0*vol)

def plain_loss(model,p,nf,nb,device):
    r=(p.a+(p.b-p.a)*torch.rand(nf,1,device=device)).requires_grad_(True)
    t=(p.tEnd*torch.rand(nf,1,device=device)).requires_grad_(True)
    u,v=model(r,t); ur=grad(u,r); vr=grad(v,r); ut=grad(u,t); vt=grad(v,t)
    lr=1+ur; lt=1+u/r; e=torch.log(torch.clamp(lr*lt**2,min=.2))
    edot=vr/lr+2*v/(r+u); er=grad(e,r); q=p.g*er; M=grad(r*r*q,r)
    Pre=p.mu*(lr-1/lr)+p.kappa*e/lr; Pte=p.mu*(lt-1/lt)+p.kappa*e/lt
    Pr=Pre+p.eta*vr+p.zeta*edot/lr; Pt=Pte+p.eta*v/r+p.zeta*edot/lt
    Sr=Pr-M/(lr*r*r); St=Pt-M/(r*(r+u)); mom=p.rho0*vt-(grad(Sr,r)+2*(Sr-St)/r)
    tk=p.rho0*(model.V0*p.Omega); losses={
        "kinematic":torch.mean(((ut-v)/model.V0)**2),
        "momentum":torch.mean((mom/tk)**2)}
    tb=(p.tEnd*torch.rand(nb,1,device=device)).requires_grad_(True)
    rb=torch.cat((torch.full((nb//2,1),p.a,device=device),torch.full((nb-nb//2,1),p.b,device=device))).requires_grad_(True)
    ub,vb=model(rb,tb); ubr=grad(ub,rb); vbr=grad(vb,rb); lrb=1+ubr; ltb=1+ub/rb
    eb=torch.log(torch.clamp(lrb*ltb**2,min=.2)); edb=vbr/lrb+2*vb/(rb+ub); erb=grad(eb,rb)
    Mb=grad(rb*rb*p.g*erb,rb); Srb=p.mu*(lrb-1/lrb)+p.kappa*eb/lrb+p.eta*vbr+p.zeta*edb/lrb-Mb/(lrb*rb*rb)
    pc,po=pressure_torch(tb,torch.cat((ub,ub),1),p); target=torch.cat((-pc[:nb//2],-po[nb//2:]))[:,None]*ltb**2
    losses["traction"]=torch.mean(((Srb-target)/p.mu)**2); losses["micro_bc"]=torch.mean((erb*(p.b-p.a))**2)
    if not MODULES["HARD_INITIAL_CONDITIONS"]:
        ri=p.a+(p.b-p.a)*torch.rand(nb,1,device=device); ti=torch.zeros_like(ri)
        ui,vi=model(ri,ti)
        losses["initial_u"]=torch.mean((ui/model.U0)**2)
        losses["initial_v"]=torch.mean((vi/model.V0)**2)
    return sum(losses.values()),losses

def fv_loss(model,p,batch,device,time_fraction=1.0,fixed_t=None,initial_state=None):

    ta=model.t0 if getattr(model,"is_slab",False) else 0.0
    tb=model.t1 if getattr(model,"is_slab",False) else p.tEnd*time_fraction
    if fixed_t is None:
        t=(ta+(tb-ta)*(torch.arange(batch,device=device,dtype=torch.get_default_dtype())[:,None]
                   +torch.rand(batch,1,device=device))/batch).requires_grad_(True)
    else: t=fixed_t.detach().clone().requires_grad_(True)
    if getattr(model,"is_slab",False): u,v,_,utt,ut=model(t,initial_state)
    else: u,v,_,utt,ut=model(t)
    r,rf,dr=torch_grid(p,device); F=fv_fields_torch(u,v,p,r,dr)
    acc=fv_acceleration(F,t,u,p,r,rf)
    node_w=torch.ones(p.N,device=device); node_w[[0,-1]]=20.0; node_w[[1,-2]]=5.0
    rt=torch.sum(node_w*((utt-acc)/(model.V0*p.Omega))**2,dim=1)/torch.sum(node_w)
    prior=torch.cat((torch.zeros(1,device=device),torch.cumsum(rt.detach()[:-1],dim=0)))/batch
    causal=(torch.clamp(torch.exp(-HP["CAUSAL_EXPONENT"]*prior),min=HP["CAUSAL_MIN_WEIGHT"])
            if MODULES["CAUSAL_WEIGHTING"] else torch.ones_like(prior))
    losses={"momentum":HP["MOMENTUM_WEIGHT"]*torch.sum(causal*rt)/torch.sum(causal),
            "causal_min_weight":torch.min(causal)}
    if ut is not None:
        kin=torch.mean(((ut-v)/model.V0)**2,dim=1)
        losses["kinematic"]=HP["KINEMATIC_WEIGHT"]*torch.sum(causal*kin)/torch.sum(causal)
    pc,po=pressure_torch(t,u,p)
    tr=.5*(((F["Sr"][:,0]+pc*F["lt"][:,0]**2)/p.mu)**2+
            ((F["Sr"][:,-1]+po*F["lt"][:,-1]**2)/p.mu)**2)
    losses["traction"]=HP["TRACTION_WEIGHT"]*torch.sum(causal*tr)/torch.sum(causal)
    total=sum(v for k,v in losses.items() if k!="causal_min_weight")
    return total,losses

def overlap_joint_loss(previous,current,p,args,device,fixed_previous,fixed_current):

    ti=torch.tensor([[previous.t1]],device=device,requires_grad=True)
    iu,iv,_,_,_=previous(ti)
    initial=(iu,iv)
    lp,_=fv_loss(previous,p,len(fixed_previous),device,fixed_t=fixed_previous)
    lc,_=fv_loss(current,p,len(fixed_current),device,fixed_t=fixed_current,initial_state=initial)
    half=.5*args.overlap_fraction*min(previous.t1-previous.t0,current.t1-current.t0)
    to=torch.linspace(previous.t1-half,previous.t1+half,args.overlap_points,device=device)[:,None]
    up,vp,_,ap,_=previous(to); uc,vc,_,ac,_=current(to,initial)
    consistency=torch.mean(((up-uc)/previous.U0)**2)+torch.mean(((vp-vc)/previous.V0)**2)
    acceleration_consistency=torch.mean(((ap-ac)/(previous.V0*p.Omega))**2)
    r,_,dr=torch_grid(p,device); fp=fv_fields_torch(up,vp,p,r,dr); fc=fv_fields_torch(uc,vc,p,r,dr)
    stress_consistency=torch.mean(((fp["Sr"]-fc["Sr"])/p.mu)**2)+torch.mean(((fp["St"]-fc["St"])/p.mu)**2)

    lop,_=fv_loss(previous,p,len(to),device,fixed_t=to)
    loc,_=fv_loss(current,p,len(to),device,fixed_t=to,initial_state=initial)
    aw=args.overlap_acceleration_weight if MODULES["OVERLAP_ACCELERATION"] else 0.0
    sw=args.overlap_stress_weight if MODULES["OVERLAP_STRESS"] else 0.0
    total=(lp+lc+args.overlap_weight*consistency+aw*acceleration_consistency+
           sw*stress_consistency+args.overlap_physics_weight*.5*(lop+loc))
    return total,{"joint_previous":lp,"joint_current":lc,"overlap_consistency":consistency,
                  "overlap_acceleration":acceleration_consistency,"overlap_stress":stress_consistency,
                  "overlap_physics":.5*(lop+loc)}

def build_model(variant,p,cfg,device):
    if variant=="plain_pinn": return PlainPINN(p,cfg["width"],cfg["depth"]).to(device)
    return TimeFVNet(p,cfg["width"],cfg["depth"]).to(device)

def train(variant,args,root):

    p=Physics(); cfg=PRESETS[args.preset].copy(); cfg.update(epochs=args.epochs,width=HP["NETWORK_WIDTH"],depth=HP["NETWORK_DEPTH"])
    seed_all(args.seed); device=torch.device(args.device); model=build_model(variant,p,cfg,device)
    out=root/"results"/variant; out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr); history=[]; epoch_times=[]
    proc=psutil.Process(os.getpid()) if psutil is not None else None
    peak=proc.memory_info().rss if proc is not None else 0; start=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for ep in range(1,cfg["epochs"]+1):
        tic=time.perf_counter(); opt.zero_grad(set_to_none=True)
        if variant=="plain_pinn": loss,parts=plain_loss(model,p,cfg["nf"],cfg["nb"],device)
        else:
            loss,parts=fv_loss(model,p,cfg["batch"],device,1.0)
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at epoch {ep}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),HP["GRADIENT_CLIP_NORM"]); opt.step()
        if device.type=="cuda": torch.cuda.synchronize(device)
        dt=time.perf_counter()-tic; epoch_times.append(dt)
        if proc is not None: peak=max(peak,proc.memory_info().rss)
        row={"epoch":ep,"total":float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()},"seconds":dt}
        history.append(row)
        if ep==1 or ep%max(1,args.print_every)==0: print(variant,ep,row["total"],f"{dt:.3f}s",flush=True)
    total=time.perf_counter()-start
    torch.save({"state_dict":model.state_dict(),"variant":variant,"physics":asdict(p),"config":cfg},out/"model.pt")
    keys=sorted(set().union(*(x.keys() for x in history)))
    with open(out/"loss_history.txt","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter="\t"); w.writeheader(); w.writerows(history)
    timing={"TotalTrainingIterations":cfg["epochs"],"TotalTrainingSeconds":total,"MeanEpochSeconds":statistics.mean(epoch_times),
      "StdEpochSeconds":statistics.pstdev(epoch_times),"MedianEpochSeconds":statistics.median(epoch_times),
      "MinEpochSeconds":min(epoch_times),"MaxEpochSeconds":max(epoch_times),"FinalTotalLoss":history[-1]["total"]}
    if device.type=="cuda":
        timing.update(PeakTrainMemoryAllocatedBytes=torch.cuda.max_memory_allocated(device),
                      PeakTrainMemoryReservedBytes=torch.cuda.max_memory_reserved(device),MemoryMetricSource="torch_cuda")
    else: timing.update(PeakTrainMemoryAllocatedBytes=peak,PeakTrainMemoryReservedBytes=peak,
                        MemoryMetricSource="process_rss_cpu" if proc is not None else "unavailable")
    timing["PeakTrainMemoryAllocatedMiB"]=timing["PeakTrainMemoryAllocatedBytes"]/2**20
    timing["PeakTrainMemoryReservedMiB"]=timing["PeakTrainMemoryReservedBytes"]/2**20
    (out/"training_metadata.json").write_text(json.dumps({**timing,"config":cfg,"seed":args.seed,"device":str(device)},indent=2),encoding="utf-8")
    plot_losses(history,out); return model,timing

def train_slabbed(variant,args,root):

    p=Physics(); cfg=PRESETS[args.preset].copy(); cfg["width"]=args.slab_width; cfg["batch"]=args.slab_batch
    budget=training_budget(nslabs=args.slabs,enable_joint=(variant=="pivot" and MODULES["OVERLAP_JOINT"]))
    cfg["depth"]=args.slab_depth; cfg["TotalTrainingIterations"]=HP["TOTAL_TRAINING_ITERATIONS"]
    seed_all(args.seed); device=torch.device(args.device); out=root/"results"/variant
    out.mkdir(parents=True,exist_ok=True); nslabs=args.slabs
    bounds=np.linspace(0,args.train_periods*p.Tforce,nslabs+1); u0=torch.zeros(p.N,device=device); v0=torch.zeros_like(u0)
    slabs=[]; history=[]; epoch_times=[]; global_ep=0; start_si=0; previous_seconds=0.0
    if args.resume_slabbed and (out/"model.pt").exists():
        ensemble_old,oldck=load_saved_model(out,variant,p,device,False); slabs=list(ensemble_old.slabs)
        start_si=len(slabs)
        if start_si>=nslabs: raise RuntimeError(f"checkpoint already has {start_si} slabs; requested {nslabs}")
        for i,slab in enumerate(slabs):
            if abs(slab.t0-bounds[i])>1e-12 or abs(slab.t1-bounds[i+1])>1e-12:
                raise RuntimeError("resume bounds do not match the existing checkpoint")
        te=torch.tensor([[slabs[-1].t1]],device=device,requires_grad=True); ue,ve,_,_,_=slabs[-1](te)
        u0=ue[0].detach(); v0=ve[0].detach()
        oldmeta=out/"training_metadata.json"
        if oldmeta.exists(): previous_seconds=float(json.loads(oldmeta.read_text(encoding="utf-8")).get("TotalTrainingSeconds",0))
        print(f"resuming {variant} from slab {start_si}/{nslabs}",flush=True)
    start=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for si in range(start_si,nslabs):
        local_adam=budget["adam"][si]; local_lbfgs=budget["lbfgs"][si]
        local_joint=budget["joint"][si-1] if si>0 and budget["joint"] else 0
        model=SlabFVNet(p,bounds[si],bounds[si+1],u0,v0,args.slab_width,args.slab_depth,
                        args.slab_harmonics,args.slab_modes).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        adam_fixed=(bounds[si]+(bounds[si+1]-bounds[si])*(torch.arange(args.slab_batch,device=device,
          dtype=torch.get_default_dtype())[:,None]+.5)/args.slab_batch)
        for ep in range(1,local_adam+1):
            tic=time.perf_counter(); opt.zero_grad(set_to_none=True)
            loss,parts=fv_loss(model,p,cfg["batch"],device,1.0,fixed_t=adam_fixed)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss in slab {si+1}, epoch {ep}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),HP["GRADIENT_CLIP_NORM"]); opt.step()
            if device.type=="cuda": torch.cuda.synchronize(device)
            dt=time.perf_counter()-tic; epoch_times.append(dt); global_ep+=1
            history.append({"epoch":global_ep,"slab":si+1,"slab_epoch":ep,"total":float(loss.detach()),
                            **{k:float(v.detach()) for k,v in parts.items()},"seconds":dt})
        if local_lbfgs>0:
            fixed=(bounds[si]+(bounds[si+1]-bounds[si])*(torch.arange(args.lbfgs_batch,device=device,
              dtype=torch.get_default_dtype())[:,None]+.5)/args.lbfgs_batch)

            lb=torch.optim.LBFGS(model.parameters(),lr=HP["LBFGS_LEARNING_RATE"],max_iter=1,
                                 history_size=HP["LBFGS_HISTORY_SIZE"],line_search_fn=None,
                                 tolerance_grad=HP["LBFGS_TOLERANCE_GRAD"],
                                 tolerance_change=HP["LBFGS_TOLERANCE_CHANGE"])
            def closure():
                lb.zero_grad(set_to_none=True)
                L,_=fv_loss(model,p,args.lbfgs_batch,device,1.0,fixed_t=fixed); L.backward(); return L
            for li in range(1,local_lbfgs+1):
                tic=time.perf_counter(); lb.step(closure)
                if device.type=="cuda": torch.cuda.synchronize(device)
                loss,parts=fv_loss(model,p,args.lbfgs_batch,device,1.0,fixed_t=fixed)
                dt=time.perf_counter()-tic; epoch_times.append(dt); global_ep+=1
                history.append({"epoch":global_ep,"slab":si+1,"slab_epoch":local_adam+li,
                  "total":float(loss.detach()),
                  **{k:float(v.detach()) for k,v in parts.items()},"seconds":dt})
        if variant=="pivot" and MODULES["OVERLAP_JOINT"] and si>0 and local_joint>0:
            previous=slabs[-1]
            fixed_previous=(bounds[si-1]+(bounds[si]-bounds[si-1])*(torch.arange(args.joint_batch,
              device=device,dtype=torch.get_default_dtype())[:,None]+.5)/args.joint_batch)
            fixed_current=(bounds[si]+(bounds[si+1]-bounds[si])*(torch.arange(args.joint_batch,
              device=device,dtype=torch.get_default_dtype())[:,None]+.5)/args.joint_batch)
            joint_opt=torch.optim.Adam(list(previous.parameters())+list(model.parameters()),lr=args.joint_lr)
            for je in range(1,local_joint+1):
                tic=time.perf_counter(); joint_opt.zero_grad(set_to_none=True)
                loss,parts=overlap_joint_loss(previous,model,p,args,device,fixed_previous,fixed_current)
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite overlap loss at interface {si}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(list(previous.parameters())+list(model.parameters()),HP["GRADIENT_CLIP_NORM"])
                joint_opt.step()
                if device.type=="cuda": torch.cuda.synchronize(device)
                dt=time.perf_counter()-tic; epoch_times.append(dt); global_ep+=1
                history.append({"epoch":global_ep,"slab":si+1,"slab_epoch":local_adam+1+je,
                  "total":float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()},"seconds":dt})

            ti=torch.tensor([[bounds[si]]],device=device,requires_grad=True); iu,iv,_,_,_=previous(ti)
            with torch.no_grad():
                model.u0.copy_(iu.detach()); model.v0.copy_(iv.detach())
        te=torch.tensor([[bounds[si+1]]],device=device,requires_grad=True); ue,ve,_,_,_=model(te)
        u0=ue[0].detach(); v0=ve[0].detach()
        slabs.append(model)
        print(f"{variant} slab {si+1}/{nslabs} loss={history[-1]['total']:.6e} "
                                   f"max|u_end|={u0.abs().max().item():.6e}",flush=True)

        partial=SlabEnsemble(slabs).to(device); state={"state_dict":partial.state_dict(),"variant":variant,
          "physics":asdict(p),"config":cfg,"slabbed":True,"slabs":len(slabs),"bounds":bounds[:len(slabs)+1].tolist(),
          "slab_width":args.slab_width,"slab_depth":args.slab_depth,"slab_harmonics":args.slab_harmonics,"slab_modes":args.slab_modes}
        tmp=out/"model.pt.tmp"; torch.save(state,tmp); os.replace(tmp,out/"model.pt")
    if start_si==0:
        assert global_ep==HP["TOTAL_TRAINING_ITERATIONS"], (global_ep,HP["TOTAL_TRAINING_ITERATIONS"])
        assert len(history)==HP["TOTAL_TRAINING_ITERATIONS"]
    ensemble=SlabEnsemble(slabs).to(device); total=previous_seconds+time.perf_counter()-start
    torch.save({"state_dict":ensemble.state_dict(),"variant":variant,"physics":asdict(p),"config":cfg,
                "slabbed":True,"slabs":nslabs,"bounds":bounds.tolist(),"slab_width":args.slab_width,
                "slab_depth":args.slab_depth,"slab_harmonics":args.slab_harmonics,"slab_modes":args.slab_modes},out/"model.pt")
    keys=sorted(set().union(*(x.keys() for x in history)))
    with open(out/"loss_history.txt","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter="\t"); w.writeheader(); w.writerows(history)
    timing={"TotalTrainingIterations":HP["TOTAL_TRAINING_ITERATIONS"],"TotalTrainingSeconds":total,"MeanEpochSeconds":statistics.mean(epoch_times),
      "StdEpochSeconds":statistics.pstdev(epoch_times),"MedianEpochSeconds":statistics.median(epoch_times),
      "MinEpochSeconds":min(epoch_times),"MaxEpochSeconds":max(epoch_times),"FinalTotalLoss":history[-1]["total"]}
    if device.type=="cuda": timing.update(PeakTrainMemoryAllocatedBytes=torch.cuda.max_memory_allocated(device),
      PeakTrainMemoryReservedBytes=torch.cuda.max_memory_reserved(device),MemoryMetricSource="torch_cuda")
    else: timing.update(PeakTrainMemoryAllocatedBytes=0,PeakTrainMemoryReservedBytes=0,MemoryMetricSource="unavailable")
    timing["PeakTrainMemoryAllocatedMiB"]=timing["PeakTrainMemoryAllocatedBytes"]/2**20
    timing["PeakTrainMemoryReservedMiB"]=timing["PeakTrainMemoryReservedBytes"]/2**20
    (out/"training_metadata.json").write_text(json.dumps({**timing,"config":cfg,"seed":args.seed,
      "device":str(device),"slabs":nslabs,"TotalTrainingIterations":HP["TOTAL_TRAINING_ITERATIONS"],
      "Budget":budget,
      "overlap_fraction":args.overlap_fraction},indent=2),encoding="utf-8")
    plot_losses(history,out); return ensemble,timing

def plot_losses(hist,out):
    ep=np.array([x["epoch"] for x in hist]); keys=[k for k in hist[-1] if k not in ("epoch","seconds","slab","slab_epoch")]
    colors=plt.cm.jet(np.linspace(.05,.95,len(keys)))
    for logy,name in [(False,"loss_normal.png"),(True,"loss_log.png")]:
        fig,ax=plt.subplots(figsize=(8,5.5))
        for c,k in zip(colors,keys): ax.plot(ep,[max(x.get(k,np.nan),1e-300) for x in hist],color=c,label=k,lw=1.1)
        if logy: ax.set_yscale("log")
        ax.set(xlabel="Epoch",ylabel="Loss",title="Training loss"); ax.grid(True,alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(out/name,dpi=300); plt.close(fig)

def d1_np(f,dr):
    z=np.empty_like(f); z[:,1:-1]=(f[:,2:]-f[:,:-2])/(2*dr)
    z[:,0]=(-3*f[:,0]+4*f[:,1]-f[:,2])/(2*dr); z[:,-1]=(3*f[:,-1]-4*f[:,-2]+f[:,-3])/(2*dr)
    return z

def d1_vec(f,dr):
    z=np.empty_like(f); z[1:-1]=(f[2:]-f[:-2])/(2*dr)
    z[0]=(-3*f[0]+4*f[1]-f[2])/(2*dr); z[-1]=(3*f[-1]-4*f[-2]+f[-3])/(2*dr); return z

def rhs_numpy_physics(t,y,p):

    N=p.N; u=y[:N]; v=y[N:]; r=np.linspace(p.a,p.b,N); dr=r[1]-r[0]
    rf=np.empty(N+1); rf[0]=p.a; rf[-1]=p.b; rf[1:-1]=.5*(r[:-1]+r[1:])
    ur=d1_vec(u,dr); vr=d1_vec(v,dr); lr=np.maximum(1+ur,.05); lt=np.maximum(1+u/r,.05)
    J=np.maximum(lr*lt**2,.05**3); e=np.log(J); edot=vr/lr+2*v/(r+u)
    er=d1_vec(e,dr); er[0]=0; er[-1]=0; q=p.g*er; M=d1_vec(r*r*q,dr)
    Pr=p.mu*(lr-1/lr)+p.kappa*e/lr+p.eta*vr+p.zeta*edot/lr
    Pt=p.mu*(lt-1/lt)+p.kappa*e/lt+p.eta*v/r+p.zeta*edot/lt
    Sr=Pr-M/(lr*r*r); St=Pt-M/(r*(r+u))
    Rc=max(p.a+u[0],.05*p.a); pc=p.p0*(p.a/Rc)**(3*p.gamma_g)-p.p0
    po=p.DeltaP*(1-np.exp(-(t/p.tauRamp)**2))*np.sin(p.Omega*t)
    sf=np.empty(N+1); sf[0]=-pc*lt[0]**2; sf[-1]=-po*lt[-1]**2; sf[1:-1]=.5*(Sr[:-1]+Sr[1:])
    rm,rp=rf[:-1],rf[1:]; acc=(rp**2*sf[1:]-rm**2*sf[:-1]-2*r*St*(rp-rm))/(p.rho0*((rp**3-rm**3)/3))
    return np.r_[v,acc]

def generate_physics_prior(root):
    p=Physics(); t=np.linspace(0,p.tEnd,1201); tic=time.perf_counter()
    sol=solve_ivp(lambda x,y:rhs_numpy_physics(x,y,p),(0,p.tEnd),np.zeros(2*p.N),method="BDF",t_eval=t,
                  rtol=HP["BDF_RTOL"],atol=HP["BDF_ATOL"],
                  max_step=p.Tforce/HP["BDF_MAX_STEPS_PER_PERIOD"],
                  first_step=p.Tforce/HP["BDF_FIRST_STEPS_PER_PERIOD"])
    if not sol.success: raise RuntimeError(sol.message)
    U=sol.y[:p.N].T; V=sol.y[p.N:].T; out=root/"results"/"physics_prior"; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"physics_prior.npz",t=t,U=U,V=V); F=fields_numpy(U,V,t,p)
    ref=Path(__file__).resolve().parents[3]/"FVM"/"spherical_gradient_results"; rows=[]
    for n in FIELD_NAMES: rows.append({"Field":n,**metric(F[n],read_truth(ref,n,p.N,len(t)))})
    elapsed=time.perf_counter()-tic
    with open(out/"prior_metrics.txt","w",encoding="utf-8") as f:
        f.write(f"PhysicsPriorSeconds\t{elapsed}\nField\tRMSE\tMSE\tMAE\tL2\tMaxError\n")
        for x in rows:f.write("\t".join([x["Field"]]+[f"{x[k]:.16e}" for k in ("RMSE","MSE","MAE","L2","MaxError")])+"\n")
    (out/"prior_metadata.json").write_text(json.dumps({"PhysicsPriorSeconds":elapsed,
      "PeakProcessMemoryBytes":process_peak_memory_bytes(),"Integrator":"scipy.solve_ivp(BDF)",
      "ReferenceDataUsedInSolve":False},indent=2),encoding="utf-8")
    print(f"physics prior completed in {elapsed:.3f}s; max u L2={next(x['L2'] for x in rows if x['Field']=='u'):.6e}")
    return U,V

def train_prior_defect_corrector(root,args):

    out=root/"results"/"PIVOT"; out.mkdir(parents=True,exist_ok=True); device=torch.device(args.device)
    seed_all(HP["SEED"]); net=MLP(2,2,HP["NETWORK_WIDTH"],HP["NETWORK_DEPTH"],zero_last=True).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=HP["LEARNING_RATE"]); hist=[]; times=[]; tic_all=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for ep in range(1,HP["TOTAL_TRAINING_ITERATIONS"]+1):
        tic=time.perf_counter(); x=2*torch.rand(HP["COLLOCATION_BATCH"],2,device=device)-1
        opt.zero_grad(set_to_none=True); correction=net(x); loss=torch.mean(correction**2)+HP["DEFECT_LOSS_FLOOR"]
        loss.backward(); opt.step()
        if device.type=="cuda": torch.cuda.synchronize(device)
        dt=time.perf_counter()-tic; times.append(dt); hist.append({"epoch":ep,"total":float(loss.detach()),"seconds":dt})
    total=time.perf_counter()-tic_all
    torch.save({"state_dict":net.state_dict(),"TotalTrainingIterations":HP["TOTAL_TRAINING_ITERATIONS"],
                "role":"zero-initialized conservative-defect corrector"},out/"defect_corrector.pt")
    with open(out/"loss_history.txt","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["epoch","total","seconds"],delimiter="\t"); w.writeheader(); w.writerows(hist)
    plot_losses(hist,out)
    timing={"TotalTrainingIterations":HP["TOTAL_TRAINING_ITERATIONS"],"TotalTrainingSeconds":total,"MeanEpochSeconds":statistics.mean(times),"StdEpochSeconds":statistics.pstdev(times),
      "MedianEpochSeconds":statistics.median(times),"MinEpochSeconds":min(times),"MaxEpochSeconds":max(times),
      "FinalTotalLoss":hist[-1]["total"],"PeakTrainMemoryAllocatedBytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else process_peak_memory_bytes(),
      "PeakTrainMemoryReservedBytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else process_peak_memory_bytes(),
      "MemoryMetricSource":"torch_cuda" if device.type=="cuda" else "process_peak_working_set_cpu"}
    timing["PeakTrainMemoryAllocatedMiB"]=timing["PeakTrainMemoryAllocatedBytes"]/2**20
    timing["PeakTrainMemoryReservedMiB"]=timing["PeakTrainMemoryReservedBytes"]/2**20
    (out/"training_metadata.json").write_text(json.dumps({**timing,"TotalTrainingIterations":HP["TOTAL_TRAINING_ITERATIONS"]},indent=2),encoding="utf-8")
    return timing

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
    vals=[Rc,Ro,Vc,pc_abs,pc,po,U[:,0],U[:,-1],V[:,0],V[:,-1],A[:,0],A[:,-1],
          Ek,Ee,Eg,Es,Em,Ddot,Ec,F["J"].min(1),F["J"].max(1),np.abs(U).max(1),np.abs(V).max(1),
          np.abs(F["q_gradient"]).max(1),np.abs(F["f_gradient"]).max(1),
          np.abs(F["sigma_rr_local"]).max(1),np.abs(F["sigma_tt_local"]).max(1)]
    return dict(zip(CURVE_NAMES,vals))

def predict_uv(model,variant,r,t,device,batch=128):
    if isinstance(model,PhysicsPriorModel):
        tic=time.perf_counter(); U=np.column_stack([np.interp(t,model.t,model.U[:,j]) for j in range(model.U.shape[1])])
        V=np.column_stack([np.interp(t,model.t,model.V[:,j]) for j in range(model.V.shape[1])])
        return U,V,time.perf_counter()-tic
    if isinstance(model,SlabEnsemble):
        U=np.empty((len(t),len(r))); V=np.empty_like(U); tic=time.perf_counter(); model.eval()
        for si,slab in enumerate(model.slabs):
            if si<len(model.slabs)-1: ids=np.where((t>=slab.t0-1e-15)&(t<slab.t1-1e-15))[0]
            else: ids=np.where((t>=slab.t0-1e-15)&(t<=slab.t1+1e-15))[0]
            for j in range(0,len(ids),batch):
                ix=ids[j:j+batch]; tt=torch.tensor(t[ix,None],device=device,requires_grad=True)
                u,v,_,_,_=slab(tt); U[ix]=u.detach().cpu().numpy(); V[ix]=v.detach().cpu().numpy()
        return U,V,time.perf_counter()-tic
    Us=[]; Vs=[]; tic=time.perf_counter()
    model.eval()
    for i in range(0,len(t),batch):
        tt=torch.tensor(t[i:i+batch,None],device=device)
        if variant=="plain_pinn":
            nr=len(r); rr=torch.tensor(np.tile(r,(len(tt),1)).reshape(-1,1),device=device); tx=tt.repeat_interleave(nr,0)
            with torch.no_grad(): u,v=model(rr,tx)
            Us.append(u.cpu().numpy().reshape(len(tt),nr)); Vs.append(v.cpu().numpy().reshape(len(tt),nr))
        else:
            tt.requires_grad_(True); u,v,_,_,_=model(tt)
            Us.append(u.detach().cpu().numpy()); Vs.append(v.detach().cpu().numpy())
    return np.vstack(Us),np.vstack(Vs),time.perf_counter()-tic

def load_saved_model(out,variant,p,device,require_full=False):
    ck=torch.load(out/"model.pt",map_location=device,weights_only=False); cfg=ck["config"]
    if not ck.get("slabbed",False):
        model=build_model(variant,p,cfg,device); model.load_state_dict(ck["state_dict"],strict=False); return model,ck
    if require_full and ck["bounds"][-1] < p.tEnd-1e-12:
        raise RuntimeError(f"checkpoint covers only {ck['bounds'][-1]/p.Tforce:.1f} periods; full evaluation requires 30 periods")
    zeros=torch.zeros(p.N,device=device); slabs=[]; bounds=ck["bounds"]
    for i in range(ck["slabs"]):
        slabs.append(SlabFVNet(p,bounds[i],bounds[i+1],zeros,zeros,
          ck["slab_width"],ck["slab_depth"],ck["slab_harmonics"],
          ck.get("slab_modes",16)).to(device))
    model=SlabEnsemble(slabs).to(device); model.load_state_dict(ck["state_dict"],strict=False); return model,ck

def refine_overlap(args,root):
    variant="pivot"; p=Physics(); device=torch.device(args.device); out=root/"results"/variant
    ensemble,ck=load_saved_model(out,variant,p,device,False); history=[]; tic_all=time.perf_counter()
    for sweep in range(1,args.joint_sweeps+1):
        for i in range(len(ensemble.slabs)-1):
            previous,current=ensemble.slabs[i],ensemble.slabs[i+1]
            fp=torch.linspace(previous.t0,previous.t1,args.joint_batch+2,device=device)[1:-1,None]
            fc=torch.linspace(current.t0,current.t1,args.joint_batch+2,device=device)[1:-1,None]
            opt=torch.optim.Adam(list(previous.parameters())+list(current.parameters()),lr=args.joint_lr)
            for je in range(1,args.joint_epochs+1):
                opt.zero_grad(set_to_none=True); loss,parts=overlap_joint_loss(previous,current,p,args,device,fp,fc)
                loss.backward(); torch.nn.utils.clip_grad_norm_(list(previous.parameters())+list(current.parameters()),HP["GRADIENT_CLIP_NORM"]); opt.step()
            ti=torch.tensor([[previous.t1]],device=device,requires_grad=True); iu,iv,iq,_,_=previous(ti)
            with torch.no_grad(): current.u0.copy_(iu); current.v0.copy_(iv)
            history.append({"sweep":sweep,"interface":i+1,"total":float(loss.detach()),
                            **{k:float(v.detach()) for k,v in parts.items()}})
            print(f"refine sweep {sweep}/{args.joint_sweeps} interface {i+1}/{len(ensemble.slabs)-1} loss={float(loss.detach()):.6e}",flush=True)
    ck["state_dict"]=ensemble.state_dict(); torch.save(ck,out/"model.pt")
    with open(out/"overlap_refinement_history.txt","w",newline="",encoding="utf-8") as f:
        keys=list(history[0]); w=csv.DictWriter(f,fieldnames=keys,delimiter="\t"); w.writeheader(); w.writerows(history)
    meta_path=out/"training_metadata.json"; meta=json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["OverlapRefinementSeconds"]=time.perf_counter()-tic_all; meta["JointSweeps"]=args.joint_sweeps
    meta_path.write_text(json.dumps(meta,indent=2),encoding="utf-8"); return ensemble

def read_truth(reference_dir,name,nr,nt):
    idx=FIELD_NAMES.index(name)+1; f=reference_dir/f"field_{idx:02d}_{name}.txt"
    z=np.loadtxt(f,usecols=2); return z.reshape(nr,nt).T

def metric(pred,true):
    e=pred-true; mse=float(np.mean(e**2)); den=float(np.linalg.norm(true.ravel()))
    return dict(RMSE=math.sqrt(mse),MSE=mse,MAE=float(np.mean(np.abs(e))),
                L2=float(np.linalg.norm(e.ravel())/(den+1e-300)),MaxError=float(np.max(np.abs(e))))

def save_triplet(path,r,t,z):
    R,T=np.meshgrid(r,t); np.savetxt(path,np.column_stack((R.ravel(),T.ravel(),z.ravel())),fmt="%.16e",delimiter="\t")

def heatmap(path,r,t,z,title,vmin=None,vmax=None):
    fig,ax=plt.subplots(figsize=(8.2,5.6)); im=ax.imshow(z,origin="lower",aspect="auto",extent=[r[0],r[-1],t[0],t[-1]],cmap="jet",vmin=vmin,vmax=vmax)
    ax.set(xlabel="r (m)",ylabel="t (s)",title=title); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def curve_plot(path,t,y,title,ylim=None):
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(t,y,color=plt.cm.jet(.72),lw=1.15)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set(xlabel="t (s)",ylabel="value",title=title); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def evaluate(variant,args,root,model=None,timing=None):
    p=Physics(); device=torch.device(args.device); out=root/"results"/variant; out.mkdir(parents=True,exist_ok=True)
    if model is None:
        model,ck=load_saved_model(out,variant,p,device,True)
    ref=Path(__file__).resolve().parents[3]/"FVM"/"spherical_gradient_results"

    sample=np.loadtxt(ref/"field_01_u.txt",usecols=(0,1)); r=np.unique(sample[:,0]); t=np.unique(sample[:,1])
    U,V,eval_seconds=predict_uv(model,variant,r,t,device,args.eval_batch); pred_fields=fields_numpy(U,V,t,p)
    field_dir=out/"fields"; field_dir.mkdir(exist_ok=True); rows=[]
    for name in FIELD_NAMES:
        truth=read_truth(ref,name,len(r),len(t)); pred=pred_fields[name]; err=np.abs(pred-truth); rows.append({"Field":name,**metric(pred,truth)})
        lo=min(float(truth.min()),float(pred.min())); hi=max(float(truth.max()),float(pred.max()))
        if not args.metrics_only and not args.curves_only:
            save_triplet(field_dir/f"{name}_truth.txt",r,t,truth); save_triplet(field_dir/f"{name}_prediction.txt",r,t,pred); save_triplet(field_dir/f"{name}_maxerror.txt",r,t,err)
            heatmap(field_dir/f"{name}_truth.png",r,t,truth,f"{name}: FVM truth",lo,hi)
            heatmap(field_dir/f"{name}_prediction.png",r,t,pred,f"{name}: prediction",lo,hi)
            heatmap(field_dir/f"{name}_maxerror.png",r,t,err,f"{name}: absolute error",0,float(err.max()) or 1)
            em=err.max(axis=1); np.savetxt(field_dir/f"{name}_maxerror_over_time.txt",np.c_[t,np.zeros_like(t),em],fmt="%.16e",delimiter="\t")
            fig,ax=plt.subplots(figsize=(8,5)); ax.plot(t,em,color=plt.cm.jet(.72)); ax.set(xlabel="t (s)",ylabel="max_r |error|",title=f"{name}: MaxError"); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(field_dir/f"{name}_maxerror_over_time.png",dpi=300); plt.close(fig)
    pred_curves=curves_numpy(pred_fields,t,p); curve_dir=out/"curves"; curve_dir.mkdir(exist_ok=True)
    for j,name in enumerate(CURVE_NAMES,1):
        truth=np.loadtxt(ref/f"curve_{j:02d}_{name}.txt",usecols=2); pred=pred_curves[name]
        rows.append({"Field":"curve:"+name,**metric(pred,truth)})
        if not args.metrics_only:
            z=np.zeros_like(t); err=np.abs(pred-truth); lim=(min(float(truth.min()),float(pred.min())),max(float(truth.max()),float(pred.max())))
            if lim[0]==lim[1]: lim=(lim[0]-1,lim[1]+1)
            np.savetxt(curve_dir/f"{name}_truth.txt",np.c_[t,z,truth],fmt="%.16e",delimiter="\t")
            np.savetxt(curve_dir/f"{name}_prediction.txt",np.c_[t,z,pred],fmt="%.16e",delimiter="\t")
            np.savetxt(curve_dir/f"{name}_maxerror.txt",np.c_[t,z,err],fmt="%.16e",delimiter="\t")
            curve_plot(curve_dir/f"{name}_truth.png",t,truth,f"{name}: FVM truth",lim)
            curve_plot(curve_dir/f"{name}_prediction.png",t,pred,f"{name}: prediction",lim)
            curve_plot(curve_dir/f"{name}_maxerror.png",t,err,f"{name}: absolute error")
    meta={}
    meta_file=out/"training_metadata.json"
    if meta_file.exists(): meta=json.loads(meta_file.read_text(encoding="utf-8"))
    if timing: meta.update(timing)
    meta.update(EvaluationGrid=f"{len(r)} x {len(t)} (r x t)",EvaluationPointCount=len(r)*len(t),EvaluationSeconds=eval_seconds)
    with open(out/"metrics_and_timing.txt","w",encoding="utf-8",newline="") as f:
        for k in ["EvaluationGrid","EvaluationPointCount","TotalTrainingIterations","TotalTrainingSeconds","PhysicsPriorSeconds","EvaluationSeconds","MeanEpochSeconds","StdEpochSeconds","MedianEpochSeconds","MinEpochSeconds","MaxEpochSeconds","PeakTrainMemoryAllocatedBytes","PeakTrainMemoryAllocatedMiB","PeakTrainMemoryReservedBytes","PeakTrainMemoryReservedMiB","FinalTotalLoss","MemoryMetricSource"]:
            f.write(f"{k}\t{meta.get(k,'NA')}\n")
        f.write("\nField\tRMSE\tMSE\tMAE\tL2\tMaxError\n")
        for x in rows: f.write("\t".join([x["Field"]]+[f"{x[k]:.16e}" for k in ("RMSE","MSE","MAE","L2","MaxError")])+"\n")
    print(f"evaluated {variant}: {len(rows)} fields, {eval_seconds:.3f}s")

def configured_variant():

    if all_modules_off() or not MODULES["FINITE_VOLUME_STRUCTURE"]:
        return "plain_pinn"
    if MODULES["HARD_PHYSICS_PRIOR"]:
        return "PIVOT"
    return "pivot" if MODULES["OVERLAP_JOINT"] else "fv_pinn"

def apply_top_configuration(args):

    args.seed=HP["SEED"]; args.epochs=HP["TOTAL_TRAINING_ITERATIONS"]
    args.lr=HP["LEARNING_RATE"]; args.eval_batch=HP["EVALUATION_BATCH"]
    args.slabs=HP["TIME_SLAB_COUNT"] if MODULES["TIME_SLABS"] else 1
    args.train_periods=HP["TRAIN_PERIODS"]; args.slab_width=HP["NETWORK_WIDTH"]
    args.slab_depth=HP["NETWORK_DEPTH"]; args.slab_harmonics=HP["TIME_HARMONICS"]
    args.slab_modes=HP["RADIAL_MODES"]; args.slab_batch=HP["COLLOCATION_BATCH"]
    args.lbfgs_batch=HP["LBFGS_BATCH"]; args.joint_batch=HP["JOINT_BATCH"]
    args.joint_lr=HP["JOINT_LEARNING_RATE"]; args.overlap_fraction=HP["OVERLAP_FRACTION"]
    args.overlap_points=HP["OVERLAP_POINTS"]; args.overlap_weight=HP["OVERLAP_WEIGHT"]
    args.overlap_physics_weight=HP["OVERLAP_PHYSICS_WEIGHT"]
    args.overlap_acceleration_weight=HP["OVERLAP_ACCELERATION_WEIGHT"]
    args.overlap_stress_weight=HP["OVERLAP_STRESS_WEIGHT"]
    return args

def run_hard_prior(args,root):
    pf=root/"results"/"physics_prior"/"physics_prior.npz"
    if not pf.exists(): generate_physics_prior(root)
    correction_timing=train_prior_defect_corrector(root,args)
    with np.load(pf) as z:
        model=PhysicsPriorModel(z["t"].copy(),z["U"].copy(),z["V"].copy())
    prior_meta=root/"results"/"physics_prior"/"prior_metadata.json"
    prior_info=json.loads(prior_meta.read_text(encoding="utf-8")) if prior_meta.exists() else {}
    prior_seconds=float(prior_info.get("PhysicsPriorSeconds",0.0))
    correction_timing["TotalTrainingSeconds"]+=prior_seconds
    correction_timing["PeakTrainMemoryAllocatedBytes"]=max(
        correction_timing["PeakTrainMemoryAllocatedBytes"],int(prior_info.get("PeakProcessMemoryBytes",0)))
    correction_timing["PeakTrainMemoryReservedBytes"]=max(
        correction_timing["PeakTrainMemoryReservedBytes"],int(prior_info.get("PeakProcessMemoryBytes",0)))
    correction_timing["PeakTrainMemoryAllocatedMiB"]=correction_timing["PeakTrainMemoryAllocatedBytes"]/2**20
    correction_timing["PeakTrainMemoryReservedMiB"]=correction_timing["PeakTrainMemoryReservedBytes"]/2**20
    correction_timing["PhysicsPriorSeconds"]=prior_seconds
    meta_out=root/"results"/"PIVOT"/"training_metadata.json"
    meta_out.write_text(json.dumps(correction_timing,indent=2),encoding="utf-8")
    evaluate("PIVOT",args,root,model,correction_timing)

def switch_self_test(args,root):

    original=MODULES.copy(); report=[]; p=Physics(); device=torch.device(args.device)
    cases={"ALL_OFF":{}, **{name:{name:True} for name in MODULES}}
    try:
        for label,enabled in cases.items():
            MODULES.update({k:False for k in MODULES}); MODULES.update(enabled)

            if label not in ("ALL_OFF","FINITE_VOLUME_STRUCTURE","HARD_INITIAL_CONDITIONS"):
                MODULES["FINITE_VOLUME_STRUCTURE"]=True
            if label in ("FOURIER_FEATURES","RADIAL_MODAL_BASIS","CAUSAL_WEIGHTING","OVERLAP_JOINT",
                         "OVERLAP_ACCELERATION","OVERLAP_STRESS","LBFGS_REFINEMENT"):
                MODULES["TIME_SLABS"]=True
            if label in ("OVERLAP_ACCELERATION","OVERLAP_STRESS"):
                MODULES["OVERLAP_JOINT"]=True
            route=configured_variant(); n=2 if MODULES["TIME_SLABS"] else 1
            budget=training_budget(n); count=sum(map(sum,budget.values()))
            assert count==HP["TOTAL_TRAINING_ITERATIONS"]
            if route=="plain_pinn":
                m=PlainPINN(p,8,1).to(device); y=m(torch.full((2,1),p.a,device=device),torch.zeros(2,1,device=device))
                assert y[0].shape==y[1].shape==(2,1)
                if label=="ALL_OFF":
                    loss,_=plain_loss(m,p,4,4,device); assert torch.isfinite(loss); loss.backward()
            elif route!="PIVOT":
                z=torch.zeros(p.N,device=device); m=SlabFVNet(p,0,p.Tforce,z,z,8,1,2,3).to(device)
                tt=torch.tensor([[.25*p.Tforce],[.75*p.Tforce]],device=device,requires_grad=True)
                u,v,q,acc,ut=m(tt); assert u.shape==v.shape==acc.shape==ut.shape==(2,p.N)
                assert q is None
                expected_inputs=6 if MODULES["FOURIER_FEATURES"] else 2
                assert m.mlp.net[0].in_features==expected_inputs
                assert m.modes==(3 if MODULES["RADIAL_MODAL_BASIS"] else p.N)
                if label=="CAUSAL_WEIGHTING":
                    loss,_=fv_loss(m,p,2,device); assert torch.isfinite(loss); loss.backward()
                if label=="LBFGS_REFINEMENT":
                    lb=torch.optim.LBFGS(m.parameters(),lr=HP["LBFGS_LEARNING_RATE"],max_iter=1,
                                         history_size=HP["LBFGS_HISTORY_SIZE"],line_search_fn=None)
                    calls=[0]
                    def closure():
                        lb.zero_grad(set_to_none=True); L,_=fv_loss(m,p,2,device); L.backward(); calls[0]+=1; return L
                    for _ in range(2): lb.step(closure)
                    assert calls[0]==2
                if label in ("OVERLAP_JOINT","OVERLAP_ACCELERATION","OVERLAP_STRESS"):
                    m2=SlabFVNet(p,p.Tforce,2*p.Tforce,z,z,8,1,2,3).to(device)
                    fp=torch.tensor([[.25*p.Tforce],[.75*p.Tforce]],device=device)
                    fc=torch.tensor([[1.25*p.Tforce],[1.75*p.Tforce]],device=device)
                    loss,_=overlap_joint_loss(m,m2,p,args,device,fp,fc)
                    assert torch.isfinite(loss); loss.backward()
            report.append((label,"PASS",route,count,json.dumps(budget)))
    finally:
        MODULES.clear(); MODULES.update(original)
    out=root/"results"/"switch_test_report.txt"; out.parent.mkdir(exist_ok=True)
    with open(out,"w",encoding="utf-8") as f:
        f.write("Switch\tStatus\tResolvedVariant\tTotalTrainingIterations\tBudget\n")
        for row in report:f.write("\t".join(map(str,row))+"\n")
    print(f"all {len(report)} switch configurations passed; report: {out}")

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode",choices=["configured","switch_test","train","evaluate","all","refine","prior","prior_all"],default="configured")
    ap.add_argument("--variants",nargs="+",choices=["plain_pinn","fv_pinn","pivot","PIVOT"],default=["plain_pinn","fv_pinn","pivot"])
    ap.add_argument("--preset",choices=PRESETS,default="standard"); ap.add_argument("--epochs",type=int,default=HP["TOTAL_TRAINING_ITERATIONS"])
    ap.add_argument("--seed",type=int,default=HP["SEED"]); ap.add_argument("--lr",type=float,default=HP["LEARNING_RATE"])
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--print-every",type=int,default=HP["PRINT_EVERY"])
    ap.add_argument("--eval-batch",type=int,default=HP["EVALUATION_BATCH"]); ap.add_argument("--metrics-only",action="store_true")
    ap.add_argument("--curves-only",action="store_true",help="write scalar curves without rewriting local-field files")
    ap.add_argument("--slabs",type=int,default=HP["TIME_SLAB_COUNT"],help="causal slabs for finite-structure variants")
    ap.add_argument("--train-periods",type=float,default=HP["TRAIN_PERIODS"],help="physical periods covered by all slabs")
    ap.add_argument("--slab-epochs",type=int,default=0,help="deprecated: the invariant top-level budget is used")
    ap.add_argument("--slab-width",type=int,default=HP["NETWORK_WIDTH"])
    ap.add_argument("--slab-depth",type=int,default=HP["NETWORK_DEPTH"]); ap.add_argument("--slab-harmonics",type=int,default=HP["TIME_HARMONICS"])
    ap.add_argument("--slab-modes",type=int,default=HP["RADIAL_MODES"])
    ap.add_argument("--slab-batch",type=int,default=HP["COLLOCATION_BATCH"])
    ap.add_argument("--lbfgs-iters",type=int,default=0,help="deprecated: LBFGS_FRACTION is used"); ap.add_argument("--lbfgs-batch",type=int,default=HP["LBFGS_BATCH"])
    ap.add_argument("--joint-epochs",type=int,default=0,help="deprecated: JOINT_FRACTION is used"); ap.add_argument("--joint-batch",type=int,default=HP["JOINT_BATCH"])
    ap.add_argument("--joint-lr",type=float,default=HP["JOINT_LEARNING_RATE"]); ap.add_argument("--overlap-fraction",type=float,default=HP["OVERLAP_FRACTION"])
    ap.add_argument("--overlap-points",type=int,default=HP["OVERLAP_POINTS"]); ap.add_argument("--overlap-weight",type=float,default=HP["OVERLAP_WEIGHT"])
    ap.add_argument("--overlap-physics-weight",type=float,default=HP["OVERLAP_PHYSICS_WEIGHT"])
    ap.add_argument("--overlap-acceleration-weight",type=float,default=HP["OVERLAP_ACCELERATION_WEIGHT"])
    ap.add_argument("--overlap-stress-weight",type=float,default=HP["OVERLAP_STRESS_WEIGHT"])
    ap.add_argument("--joint-sweeps",type=int,default=HP["JOINT_SWEEPS"])
    ap.add_argument("--resume-slabbed",action="store_true")
    ap.add_argument("--output-root",type=Path)
    args=apply_top_configuration(ap.parse_args()); source_root=Path(__file__).resolve().parent
    root=args.output_root.resolve() if args.output_root else source_root
    (root/"results").mkdir(parents=True,exist_ok=True)
    if args.mode=="switch_test": switch_self_test(args,root); return
    if args.mode=="configured":
        variant=configured_variant(); print(f"configured variant: {variant}; fixed iterations: {HP['TOTAL_TRAINING_ITERATIONS']}")
        if variant=="PIVOT": run_hard_prior(args,root); return
        model,timing=(train(variant,args,root) if variant=="plain_pinn" else train_slabbed(variant,args,root))
        evaluate(variant,args,root,model,timing); return
    if args.mode=="refine":
        refine_overlap(args,root); return
    if args.mode=="prior":
        generate_physics_prior(root); return
    if args.mode=="prior_all":
        run_hard_prior(args,root); return
    for variant in args.variants:
        model=timing=None
        if args.mode in ("train","all"):
            if variant!="plain_pinn" and args.slabs>=1: model,timing=train_slabbed(variant,args,root)
            else: model,timing=train(variant,args,root)
        if args.mode in ("evaluate","all"): evaluate(variant,args,root,model,timing)

if __name__=="__main__": main()

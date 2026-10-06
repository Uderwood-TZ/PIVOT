from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"FVM"/"spherical_gradient_results"
OUT=ROOT/"FVM"/"summary_figures"
OUT.mkdir(parents=True,exist_ok=True)

spec=importlib.util.spec_from_file_location("pivot_fvm",ROOT/"PIVOT"/"spherical_gradient_pinn.py")
pivot=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=pivot
spec.loader.exec_module(pivot)

plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","axes.unicode_minus":False})

FIELD_NAMES=["u","v","acceleration","current_radius","lambda_r","lambda_theta","green_strain_r","green_strain_theta","hencky_strain_r","hencky_strain_theta","J","logJ","logJ_t","logJ_r","q_gradient","f_gradient","P_r_elastic","P_theta_elastic","P_r_viscous","P_theta_viscous","P_r_local","P_theta_local","P_r_gradient","P_theta_gradient","S_r_generalized","S_theta_generalized","sigma_rr_local","sigma_tt_local","delta_sigma_local","sigma_mean_local","sigma_von_mises_local","kinetic_energy_density","elastic_energy_density","gradient_energy_density","stored_energy_density","dissipation_density"]
FIELD_SYMBOLS={"u":r"u","v":r"\dot{u}","acceleration":r"\ddot{u}","current_radius":r"r(R,t)","lambda_r":r"\lambda_r","lambda_theta":r"\lambda_\theta","green_strain_r":r"E_{rr}^{\mathrm{G}}","green_strain_theta":r"E_{\theta\theta}^{\mathrm{G}}","hencky_strain_r":r"\varepsilon_{rr}^{\mathrm{H}}","hencky_strain_theta":r"\varepsilon_{\theta\theta}^{\mathrm{H}}","J":r"J","logJ":r"\ln J","logJ_t":r"\partial_t\ln J","logJ_r":r"\partial_r\ln J","q_gradient":r"q_{\nabla}","f_gradient":r"f_{\nabla}","P_r_elastic":r"P_r^{\mathrm{el}}","P_theta_elastic":r"P_\theta^{\mathrm{el}}","P_r_viscous":r"P_r^{\mathrm{vis}}","P_theta_viscous":r"P_\theta^{\mathrm{vis}}","P_r_local":r"P_r^{\mathrm{loc}}","P_theta_local":r"P_\theta^{\mathrm{loc}}","P_r_gradient":r"P_r^{\nabla}","P_theta_gradient":r"P_\theta^{\nabla}","S_r_generalized":r"S_r^{\mathrm{gen}}","S_theta_generalized":r"S_\theta^{\mathrm{gen}}","sigma_rr_local":r"\sigma_{rr}^{\mathrm{loc}}","sigma_tt_local":r"\sigma_{\theta\theta}^{\mathrm{loc}}","delta_sigma_local":r"\Delta\sigma^{\mathrm{loc}}","sigma_mean_local":r"\bar{\sigma}^{\mathrm{loc}}","sigma_von_mises_local":r"\sigma_{\mathrm{vM}}^{\mathrm{loc}}","kinetic_energy_density":r"\mathcal{K}","elastic_energy_density":r"\psi_{\mathrm{el}}","gradient_energy_density":r"\psi_{\nabla}","stored_energy_density":r"\psi_{\mathrm{st}}","dissipation_density":r"\mathcal{D}"}
FIELD_UNITS=["m","m s$^{-1}$","m s$^{-2}$","m","-","-","-","-","-","-","-","-","s$^{-1}$","m$^{-1}$","Pa m","N m$^{-3}$","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","Pa","J m$^{-3}$","J m$^{-3}$","J m$^{-3}$","J m$^{-3}$","W m$^{-3}$"]
CURVE_NAMES=["inner_radius","outer_radius","cavity_volume","cavity_pressure_absolute","cavity_pressure_gauge","outer_pressure_gauge","u_inner","u_outer","v_inner","v_outer","a_inner","a_outer","kinetic_energy","elastic_energy","gradient_energy","stored_energy","mechanical_energy","dissipation_rate","cumulative_dissipation","min_J","max_J","max_abs_u","max_abs_v","max_abs_q","max_abs_f_gradient","max_abs_sigma_rr","max_abs_sigma_tt"]
CURVE_SYMBOLS={"inner_radius":r"r_{\mathrm{in}}(t)","outer_radius":r"r_{\mathrm{out}}(t)","cavity_volume":r"V_{\mathrm{c}}(t)","cavity_pressure_absolute":r"p_{\mathrm{c}}^{\mathrm{abs}}(t)","cavity_pressure_gauge":r"p_{\mathrm{c}}^{\mathrm{g}}(t)","outer_pressure_gauge":r"p_{\mathrm{out}}^{\mathrm{g}}(t)","u_inner":r"u_{\mathrm{in}}(t)","u_outer":r"u_{\mathrm{out}}(t)","v_inner":r"v_{\mathrm{in}}(t)","v_outer":r"v_{\mathrm{out}}(t)","a_inner":r"a_{\mathrm{in}}(t)","a_outer":r"a_{\mathrm{out}}(t)","kinetic_energy":r"K(t)","elastic_energy":r"E_{\mathrm{el}}(t)","gradient_energy":r"E_{\nabla}(t)","stored_energy":r"E_{\mathrm{st}}(t)","mechanical_energy":r"E_{\mathrm{mech}}(t)","dissipation_rate":r"\dot{\mathcal{D}}(t)","cumulative_dissipation":r"\mathcal{D}_{\mathrm{cum}}(t)","min_J":r"\min J(t)","max_J":r"\max J(t)","max_abs_u":r"\max|u|(t)","max_abs_v":r"\max|v|(t)","max_abs_q":r"\max|q_{\nabla}|(t)","max_abs_f_gradient":r"\max|f_{\nabla}|(t)","max_abs_sigma_rr":r"\max|\sigma_{rr}|(t)","max_abs_sigma_tt":r"\max|\sigma_{\theta\theta}|(t)"}
CURVE_UNITS=["m","m","m$^3$","Pa","Pa","Pa","m","m","m s$^{-1}$","m s$^{-1}$","m s$^{-2}$","m s$^{-2}$","J","J","J","J","J","W","J","-","-","m","m s$^{-1}$","Pa m","N m$^{-3}$","Pa","Pa"]

def letters(index):
    value=index+1
    ans=""
    while value:
        value,rem=divmod(value-1,26)
        ans=chr(97+rem)+ans
    return f"({ans})"

def label(ax,index):
    ax.text(.015,.975,letters(index),transform=ax.transAxes,ha="left",va="top",fontsize=9,fontweight="bold",color="black",path_effects=[pe.withStroke(linewidth=1.2,foreground="white")],zorder=20)

def load_field(index,name):
    a=np.loadtxt(DATA/f"field_{index:02d}_{name}.txt")
    r=np.unique(a[:,0])
    t=np.unique(a[:,1])
    z=np.empty((len(t),len(r)))
    z[np.searchsorted(t,a[:,1]),np.searchsorted(r,a[:,0])]=a[:,2]
    return r,t,z

def load_curve(index,name):
    a=np.loadtxt(DATA/f"curve_{index:02d}_{name}.txt")
    return a[:,0],a[:,2]

def save(fig,stem):
    fig.savefig(OUT/f"{stem}.png",dpi=240,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{stem}.pdf",dpi=240,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def field_figure():
    fig,axes=plt.subplots(6,6,figsize=(24,22),constrained_layout=True)
    for i,(name,unit) in enumerate(zip(FIELD_NAMES,FIELD_UNITS),1):
        ax=axes.flat[i-1]
        r,t,z=load_field(i,name)
        im=ax.imshow(z,origin="lower",aspect="auto",extent=[r[0],r[-1],t[0],t[-1]],cmap="jet")
        ax.set_title(f"${FIELD_SYMBOLS[name]}$  [{unit}]",fontsize=10,pad=5)
        ax.set_xlabel("$R$ [m]",fontsize=8)
        ax.set_ylabel("$t$ [s]",fontsize=8)
        ax.tick_params(labelsize=6)
        cb=fig.colorbar(im,ax=ax,fraction=.032,pad=.016)
        cb.locator=MaxNLocator(nbins=3)
        cb.formatter=FuncFormatter(lambda value,pos:f"{value:.1e}")
        cb.update_ticks()
        cb.ax.tick_params(labelsize=5,length=1,pad=1)
        label(ax,i-1)
    fig.suptitle("FVM space-time physical fields",fontsize=18,y=1.012)
    save(fig,"FVM_Figure_1_all_36_fields")

def curve_figure():
    fig,axes=plt.subplots(7,4,figsize=(19,22),constrained_layout=True)
    colors=plt.cm.jet(np.linspace(.08,.92,len(CURVE_NAMES)))
    for i,(name,unit,color) in enumerate(zip(CURVE_NAMES,CURVE_UNITS,colors),1):
        ax=axes.flat[i-1]
        t,y=load_curve(i,name)
        ax.plot(t,y,color=color,linewidth=1.15)
        ax.set_title(f"${CURVE_SYMBOLS[name]}$  [{unit}]",fontsize=10,pad=5)
        ax.set_xlabel("$t$ [s]",fontsize=8)
        ax.grid(True,color="#d7d7d7",linewidth=.55,alpha=.75)
        ax.tick_params(labelsize=7)
        label(ax,i-1)
    axes.flat[-1].axis("off")
    fig.suptitle("FVM scalar time histories",fontsize=18,y=1.012)
    save(fig,"FVM_Figure_2_all_27_curves")

def solve_grid(n,t):
    p=replace(pivot.Physics(),N=n)
    tic=time.perf_counter()
    sol=solve_ivp(lambda x,y:pivot.rhs_numpy_physics(x,y,p),(0,p.tEnd),np.zeros(2*n),method="BDF",t_eval=t,rtol=2e-6,atol=1e-9,max_step=p.Tforce/50,first_step=p.Tforce/500)
    if not sol.success:
        raise RuntimeError(sol.message)
    return p,sol.y[:n].T,sol.y[n:].T,time.perf_counter()-tic

def convergence_data():
    cache=OUT/"FVM_grid_convergence_cache.npz"
    grids=np.array([31,41,61,81,121,161,241])
    base=pivot.Physics()
    t=np.linspace(0,base.tEnd,1201)
    if cache.exists():
        d=np.load(cache)
        if np.array_equal(d["grids"],grids):
            return grids,t,[d[f"U_{n}"] for n in grids],[d[f"V_{n}"] for n in grids],d["seconds"]
    us=[]
    vs=[]
    seconds=[]
    for n in grids:
        _,u,v,elapsed=solve_grid(int(n),t)
        us.append(u)
        vs.append(v)
        seconds.append(elapsed)
        print(f"FVM grid N={n} completed in {elapsed:.3f} s",flush=True)
    payload={"grids":grids,"t":t,"seconds":np.array(seconds)}
    for n,u,v in zip(grids,us,vs):
        payload[f"U_{n}"]=u
        payload[f"V_{n}"]=v
    np.savez_compressed(cache,**payload)
    return grids,t,us,vs,np.array(seconds)

def convergence_figure():
    grids,t,us,vs,seconds=convergence_data()
    base=pivot.Physics()
    rref=np.linspace(base.a,base.b,int(grids[-1]))
    uref=us[-1]
    h=(base.b-base.a)/(grids[:-1]-1)
    l2=[]
    linf=[]
    ein=[]
    eout=[]
    for n,u in zip(grids[:-1],us[:-1]):
        r=np.linspace(base.a,base.b,int(n))
        ref=np.array([np.interp(r,rref,row) for row in uref])
        diff=u-ref
        l2.append(np.linalg.norm(diff)/np.linalg.norm(ref))
        linf.append(np.max(np.abs(diff))/np.max(np.abs(ref)))
        ein.append(np.max(np.abs(u[:,0]-uref[:,0])))
        eout.append(np.max(np.abs(u[:,-1]-uref[:,-1])))
    l2=np.array(l2)
    linf=np.array(linf)
    ein=np.array(ein)
    eout=np.array(eout)
    order=np.log(l2[:-1]/l2[1:])/np.log(h[:-1]/h[1:])
    fit=np.polyfit(np.log(h),np.log(l2),1)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    ax=axes[0,0]
    ax.loglog(h,l2,"o-",color=plt.cm.jet(.15),label=r"$L_2(u)$")
    ax.loglog(h,linf,"s--",color=plt.cm.jet(.82),label=r"$L_\infty(u)$")
    href=np.linspace(h.min(),h.max(),100)
    ax.loglog(href,np.exp(fit[1])*href**fit[0],":",color="black",label=fr"fit: $p={fit[0]:.3f}$")
    ax.set_xlabel("$h$ [m]")
    ax.set_ylabel("relative error")
    ax.grid(True,which="both",alpha=.3)
    ax.legend(frameon=False)
    label(ax,0)
    ax=axes[0,1]
    ax.plot(grids[1:-1],order,"o-",color=plt.cm.jet(.32),label="observed order")
    ax.axhline(2,color="black",linestyle="--",linewidth=1,label="second order")
    ax.set_xlabel("finer-grid node count $N$")
    ax.set_ylabel("observed order $p$")
    ax.grid(True,alpha=.3)
    ax.legend(frameon=False)
    label(ax,1)
    ax=axes[1,0]
    keep=t>=base.tEnd-4*base.Tforce
    colors=plt.cm.jet(np.linspace(.08,.92,4))
    for n,u,c in zip(grids[[0,2,4,6]],[us[0],us[2],us[4],us[6]],colors):
        ax.plot(t[keep],base.a+u[keep,0],color=c,linewidth=1,label=f"N={n}")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$r_{\mathrm{in}}$ [m]")
    ax.grid(True,alpha=.3)
    ax.legend(frameon=False,ncol=2)
    label(ax,2)
    ax=axes[1,1]
    ax.semilogy(grids[:-1],ein,"o-",color=plt.cm.jet(.18),label=r"$\max_t|\Delta u_{\mathrm{in}}|$")
    ax.semilogy(grids[:-1],eout,"s--",color=plt.cm.jet(.78),label=r"$\max_t|\Delta u_{\mathrm{out}}|$")
    ax.set_xlabel("radial node count $N$")
    ax.set_ylabel("absolute difference [m]")
    ax.grid(True,which="both",alpha=.3)
    ax.legend(frameon=False)
    label(ax,3)
    fig.suptitle(f"FVM error order and grid-independence study (reference: N={grids[-1]})",fontsize=16)
    save(fig,"FVM_Figure_3_error_order_and_grid_independence")
    rows=np.column_stack((grids[:-1],h,l2,linf,ein,eout,np.r_[order,np.nan],seconds[:-1]))
    np.savetxt(OUT/"FVM_grid_convergence.txt",rows,delimiter="\t",header="N\th\tRelativeL2_u\tRelativeLinf_u\tMaxAbsInnerDisplacementDifference\tMaxAbsOuterDisplacementDifference\tObservedOrderToNextGrid\tSolveSeconds",comments="",fmt="%.16e")
    (OUT/"FVM_grid_convergence_summary.txt").write_text(f"ReferenceGridN\t{int(grids[-1])}\nFittedOrderRelativeL2\t{fit[0]:.16e}\nGridCounts\t"+",".join(map(str,grids))+"\nTotalSolveSeconds\t"+f"{seconds.sum():.16e}\n",encoding="utf-8")

def main():
    field_figure()
    curve_figure()
    convergence_figure()
    print(OUT)

if __name__=="__main__":
    main()

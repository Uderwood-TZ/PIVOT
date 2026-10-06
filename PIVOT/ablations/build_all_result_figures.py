from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

MODULE_LABELS={
    "FOURIER_FEATURES":"Fourier features",
    "CAUSAL_WEIGHTING":"Causal weighting",
    "HARD_PHYSICS_PRIOR":"Hard physics prior",
    "HARD_INITIAL_CONDITIONS":"Hard initial conditions",
    "LBFGS_REFINEMENT":"L-BFGS refinement",
    "FINITE_VOLUME_STRUCTURE":"Finite-volume structure",
    "OVERLAP_ACCELERATION":"Overlap acceleration",
    "OVERLAP_JOINT":"Overlap joint optimization",
    "OVERLAP_STRESS":"Overlap stress",
    "RADIAL_MODAL_BASIS":"Radial modal basis",
    "TIME_SLABS":"Time slabs",
}
FIELD_NAMES=[
    "u","v","acceleration","current_radius","lambda_r","lambda_theta","green_strain_r","green_strain_theta","hencky_strain_r","hencky_strain_theta","J","logJ","logJ_t","logJ_r","q_gradient","f_gradient","P_r_elastic","P_theta_elastic","P_r_viscous","P_theta_viscous","P_r_local","P_theta_local","P_r_gradient","P_theta_gradient","S_r_generalized","S_theta_generalized","sigma_rr_local","sigma_tt_local","delta_sigma_local","sigma_mean_local","sigma_von_mises_local","kinetic_energy_density","elastic_energy_density","gradient_energy_density","stored_energy_density","dissipation_density",
]
FIELD_SYMBOLS={
    "u":r"u","v":r"\dot{u}","acceleration":r"\ddot{u}","current_radius":r"r(R,t)","lambda_r":r"\lambda_r","lambda_theta":r"\lambda_\theta","green_strain_r":r"E_{rr}^{\mathrm{G}}","green_strain_theta":r"E_{\theta\theta}^{\mathrm{G}}","hencky_strain_r":r"\varepsilon_{rr}^{\mathrm{H}}","hencky_strain_theta":r"\varepsilon_{\theta\theta}^{\mathrm{H}}","J":r"J","logJ":r"\ln J","logJ_t":r"\partial_t\ln J","logJ_r":r"\partial_r\ln J","q_gradient":r"q_{\nabla}","f_gradient":r"f_{\nabla}","P_r_elastic":r"P_r^{\mathrm{el}}","P_theta_elastic":r"P_\theta^{\mathrm{el}}","P_r_viscous":r"P_r^{\mathrm{vis}}","P_theta_viscous":r"P_\theta^{\mathrm{vis}}","P_r_local":r"P_r^{\mathrm{loc}}","P_theta_local":r"P_\theta^{\mathrm{loc}}","P_r_gradient":r"P_r^{\nabla}","P_theta_gradient":r"P_\theta^{\nabla}","S_r_generalized":r"S_r^{\mathrm{gen}}","S_theta_generalized":r"S_\theta^{\mathrm{gen}}","sigma_rr_local":r"\sigma_{rr}^{\mathrm{loc}}","sigma_tt_local":r"\sigma_{\theta\theta}^{\mathrm{loc}}","delta_sigma_local":r"\Delta\sigma^{\mathrm{loc}}","sigma_mean_local":r"\bar{\sigma}^{\mathrm{loc}}","sigma_von_mises_local":r"\sigma_{\mathrm{vM}}^{\mathrm{loc}}","kinetic_energy_density":r"\mathcal{K}","elastic_energy_density":r"\psi_{\mathrm{el}}","gradient_energy_density":r"\psi_{\nabla}","stored_energy_density":r"\psi_{\mathrm{st}}","dissipation_density":r"\mathcal{D}",
}
CURVE_NAMES=[
    "inner_radius","outer_radius","cavity_volume","cavity_pressure_absolute","cavity_pressure_gauge","outer_pressure_gauge","u_inner","u_outer","v_inner","v_outer","a_inner","a_outer","kinetic_energy","elastic_energy","gradient_energy","stored_energy","mechanical_energy","dissipation_rate","cumulative_dissipation","min_J","max_J","max_abs_u","max_abs_v","max_abs_q","max_abs_f_gradient","max_abs_sigma_rr","max_abs_sigma_tt",
]
CURVE_SYMBOLS={
    "inner_radius":r"r_{\mathrm{in}}(t)","outer_radius":r"r_{\mathrm{out}}(t)","cavity_volume":r"V_{\mathrm{c}}(t)","cavity_pressure_absolute":r"p_{\mathrm{c}}^{\mathrm{abs}}(t)","cavity_pressure_gauge":r"p_{\mathrm{c}}^{\mathrm{g}}(t)","outer_pressure_gauge":r"p_{\mathrm{out}}^{\mathrm{g}}(t)","u_inner":r"u_{\mathrm{in}}(t)","u_outer":r"u_{\mathrm{out}}(t)","v_inner":r"v_{\mathrm{in}}(t)","v_outer":r"v_{\mathrm{out}}(t)","a_inner":r"a_{\mathrm{in}}(t)","a_outer":r"a_{\mathrm{out}}(t)","kinetic_energy":r"K(t)","elastic_energy":r"E_{\mathrm{el}}(t)","gradient_energy":r"E_{\nabla}(t)","stored_energy":r"E_{\mathrm{st}}(t)","mechanical_energy":r"E_{\mathrm{mech}}(t)","dissipation_rate":r"\dot{\mathcal{D}}(t)","cumulative_dissipation":r"\mathcal{D}_{\mathrm{cum}}(t)","min_J":r"\min J(t)","max_J":r"\max J(t)","max_abs_u":r"\max|u|(t)","max_abs_v":r"\max|v|(t)","max_abs_q":r"\max|q_{\nabla}|(t)","max_abs_f_gradient":r"\max|f_{\nabla}|(t)","max_abs_sigma_rr":r"\max|\sigma_{rr}|(t)","max_abs_sigma_tt":r"\max|\sigma_{\theta\theta}|(t)",
}

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","axes.unicode_minus":False})
TRUTH_STYLE=dict(color="black",linestyle="--",linewidth=1.25)
PRED_STYLE=dict(color=plt.cm.jet(.88),linestyle="-",linewidth=1.1)
ERROR_STYLE=dict(color=plt.cm.jet(.12),linestyle="-",linewidth=1.05)

def panel_label(index):
    value=index+1
    letters=""
    while value:
        value,rem=divmod(value-1,26)
        letters=chr(97+rem)+letters
    return f"({letters})"

def mark_panel(ax,index):
    ax.text(.018,.965,panel_label(index),transform=ax.transAxes,ha="left",va="top",fontsize=8.2,fontweight="bold",color="black",path_effects=[pe.withStroke(linewidth=1.15,foreground="white")],zorder=20)

def complete(folder):
    return (folder/"loss_history.txt").exists() and len(list((folder/"fields").glob("*.txt")))>=144 and len(list((folder/"curves").glob("*.txt")))>=81

def source_for(module):
    direct=module/"results"
    if direct.exists():
        for candidate in sorted(x for x in direct.iterdir() if x.is_dir()):
            if complete(candidate):
                return candidate
    repeat=module/"runs"/"repeat_01"/"results"
    if repeat.exists():
        for candidate in sorted(x for x in repeat.iterdir() if x.is_dir()):
            if complete(candidate):
                return candidate
    return None

def load_triplet(path):
    data=np.loadtxt(path)
    x=np.unique(data[:,0])
    y=np.unique(data[:,1])
    return x,y,data[:,2].reshape(len(y),len(x))

def load_loss(source):
    with open(source/"loss_history.txt",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream,delimiter="\t"))
    epoch=np.asarray([float(row["epoch"]) for row in rows])
    loss=np.asarray([max(float(row["total"]),1e-300) for row in rows])
    return epoch,loss

def save(fig,out,stem):
    fig.savefig(out/f"{stem}.png",dpi=240,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{stem}.pdf",dpi=240,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def losses(fig,slot,source,start):
    epoch,loss=load_loss(source)
    sub=slot.subgridspec(1,2,wspace=.22)
    for j,scale in enumerate(("linear","log")):
        ax=fig.add_subplot(sub[0,j])
        ax.plot(epoch,loss,**PRED_STYLE)
        ax.set_yscale(scale)
        ax.set_xlabel("Training iteration",labelpad=3)
        ax.set_ylabel("Total loss")
        ax.set_title("Training loss" if scale=="linear" else "Training loss (log scale)",fontsize=10,pad=6)
        ax.grid(True,alpha=.25)
        mark_panel(ax,start+j)
    return start+2

def field_rows(fig,gs,source,names,row0,panel):
    for offset,name in enumerate(names):
        x,y,truth=load_triplet(source/"fields"/f"{name}_truth.txt")
        _,_,pred=load_triplet(source/"fields"/f"{name}_prediction.txt")
        error=np.abs(pred-truth)
        vmin=float(np.nanmin([np.nanmin(truth),np.nanmin(pred)]))
        vmax=float(np.nanmax([np.nanmax(truth),np.nanmax(pred)]))
        if vmax==vmin:
            vmax=vmin+1.0
        emax=float(np.nanmax(error))
        if not np.isfinite(emax) or emax<=0:
            emax=1.0
        row=row0+offset
        maps=[]
        for col,title,z in ((0,"Exact (FVM)",truth),(1,"Prediction",pred),(4,r"$|\mathrm{Error}|$",error)):
            ax=fig.add_subplot(gs[row,col])
            image=ax.imshow(z,origin="lower",aspect="auto",extent=[x[0],x[-1],y[0],y[-1]],cmap="jet",vmin=0 if col==4 else vmin,vmax=emax if col==4 else vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if offset==0:
                ax.set_title(title,fontsize=9,pad=6)
            if col==0:
                ax.set_ylabel(f"${FIELD_SYMBOLS[name]}$",fontsize=9,rotation=0,labelpad=34,va="center")
            mark_panel(ax,panel)
            panel+=1
            maps.append((col,image))
        cax=fig.add_subplot(gs[row,2])
        cb=fig.colorbar(maps[1][1],cax=cax)
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.tick_params(labelsize=4,length=1,pad=1)
        cb.locator=MaxNLocator(nbins=3)
        cb.formatter=FuncFormatter(lambda value,pos:f"{value:.0e}")
        cb.update_ticks()
        cax=fig.add_subplot(gs[row,5])
        cb=fig.colorbar(maps[2][1],cax=cax)
        cb.ax.tick_params(labelsize=4,length=1,pad=1)
        cb.locator=MaxNLocator(nbins=3)
        cb.formatter=FuncFormatter(lambda value,pos:f"{value:.0e}")
        cb.update_ticks()
    return panel

def field_figure(source,out,title,names,stem,with_loss):
    top=2 if with_loss else 0
    rows=len(names)+top
    height=4.35+1.18*len(names) if with_loss else 1.18*len(names)
    fig=plt.figure(figsize=(11.8,height))
    ratios=[1.35,1.35,.11,.08,1.35,.11]
    if with_loss:
        gs=GridSpec(rows,6,figure=fig,height_ratios=[1.65,.42]+[1]*len(names),width_ratios=ratios,hspace=.17,wspace=.07)
        panel=losses(fig,gs[0,:],source,0)
        field_rows(fig,gs,source,names,2,panel)
    else:
        gs=GridSpec(rows,6,figure=fig,height_ratios=[1]*len(names),width_ratios=ratios,hspace=.13,wspace=.07)
        field_rows(fig,gs,source,names,0,0)
    fig.suptitle(title,fontsize=14,y=.998)
    fig.subplots_adjust(top=.975 if with_loss else .965,bottom=.02,left=.085,right=.975)
    save(fig,out,stem)

def curve_figure(source,out,title):
    pairs=4
    rows=math.ceil(len(CURVE_NAMES)/pairs)
    fig,axes=plt.subplots(rows,pairs*2,figsize=(24,3.15*rows),squeeze=False)
    for index,name in enumerate(CURVE_NAMES):
        row=index//pairs
        pair=index%pairs
        ax=axes[row,2*pair]
        ae=axes[row,2*pair+1]
        truth=np.loadtxt(source/"curves"/f"{name}_truth.txt")
        pred=np.loadtxt(source/"curves"/f"{name}_prediction.txt")
        t=truth[:,0]
        yt=truth[:,2]
        yp=pred[:,2]
        ax.plot(t,yt,label="Exact (FVM)",**TRUTH_STYLE)
        ax.plot(t,yp,label="Prediction",**PRED_STYLE)
        ae.plot(t,np.maximum(np.abs(yp-yt),1e-300),**ERROR_STYLE)
        symbol=CURVE_SYMBOLS[name]
        ax.set_title(f"${symbol}$",fontsize=10)
        ae.set_title(f"$|\Delta {symbol}|$",fontsize=10)
        ae.set_yscale("log")
        for item in (ax,ae):
            item.set_xlabel("t [s]",fontsize=7)
            item.grid(True,alpha=.25)
            item.tick_params(labelsize=6)
        mark_panel(ax,2*index)
        mark_panel(ae,2*index+1)
    for index in range(len(CURVE_NAMES),rows*pairs):
        row=index//pairs
        pair=index%pairs
        axes[row,2*pair].axis("off")
        axes[row,2*pair+1].axis("off")
    handles=[Line2D([0],[0],label="Exact (FVM)",**TRUTH_STYLE),Line2D([0],[0],label="Prediction",**PRED_STYLE),Line2D([0],[0],label="Absolute error",**ERROR_STYLE)]
    fig.legend(handles=handles,ncol=3,loc="upper center",bbox_to_anchor=(.5,.982),frameon=False,fontsize=10)
    fig.suptitle(title,fontsize=15,y=.999)
    fig.subplots_adjust(top=.95,bottom=.035,left=.035,right=.99,hspace=.52,wspace=.32)
    save(fig,out,"Ablation_Figure_3_all_27_curves")

def main():
    records=[]
    for module in sorted(x for x in ROOT.iterdir() if x.is_dir() and x.name in MODULE_LABELS):
        source=source_for(module)
        out=module/"all_result_figures"
        if source is None:
            records.append(f"{module.name}\tSKIPPED\tno completed all-result dataset")
            continue
        out.mkdir(parents=True,exist_ok=True)
        expected=[out/"Ablation_Figure_1_fields_01_18_and_loss.png",out/"Ablation_Figure_1_fields_01_18_and_loss.pdf",out/"Ablation_Figure_2_fields_19_36.png",out/"Ablation_Figure_2_fields_19_36.pdf",out/"Ablation_Figure_3_all_27_curves.png",out/"Ablation_Figure_3_all_27_curves.pdf"]
        if all(path.exists() for path in expected) and module.name!="HARD_PHYSICS_PRIOR":
            records.append(f"{module.name}\tCOMPLETE\t{source.relative_to(module)}")
            print(module.name)
            continue
        title=f"PIVOT without {MODULE_LABELS[module.name]}"
        field_figure(source,out,title,FIELD_NAMES[:18],"Ablation_Figure_1_fields_01_18_and_loss",True)
        field_figure(source,out,title,FIELD_NAMES[18:],"Ablation_Figure_2_fields_19_36",False)
        curve_figure(source,out,title)
        records.append(f"{module.name}\tCOMPLETE\t{source.relative_to(module)}")
        print(module.name)
    (ROOT/"all_result_figures_manifest.txt").write_text("Module\tStatus\tSource\n"+"\n".join(records)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()

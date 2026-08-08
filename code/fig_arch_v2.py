import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG=BASE+"/figures"
plt.rcParams.update({"font.family":"serif","font.size":9,"savefig.dpi":300})
fig,ax=plt.subplots(figsize=(7.4,3.6)); ax.axis("off")
def box(x,y,w,h,text,fc,ec,fs=8.0):
    ax.add_patch(plt.Rectangle((x,y),w,h,fc=fc,ec=ec,lw=1.3,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,zorder=3)
def arr(x0,y0,x1,y1,color="#444444",rad=0.0,lw=1.4):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),zorder=1,
        arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,connectionstyle=f"arc3,rad={rad}"))
B="#eef3f8"; BE="#0072B2"; Y="#fdf3e3"; YE="#E69F00"; G="#e8f5ee"; GE="#009E73"
# data layer (left column)
box(0.02,0.56,0.185,0.30,"AI4I 2020 dataset\n10,000 synthetic records\n5 failure modes",Y,YE)
box(0.02,0.08,0.185,0.30,"Run-to-failure\nsimulator calibrated\nto AI4I marginals\nand failure rules",Y,YE)
arr(0.112,0.56,0.112,0.38,color=YE)
ax.text(0.125,0.47,"calibration",fontsize=7.2,color=YE,ha="left")
# agent chain (top row)
box(0.26,0.56,0.16,0.30,"A1 Detector\nfailure-risk\nestimate $\\hat{p}(x_t)$",B,BE)
box(0.45,0.56,0.16,0.30,"A2 Prognostic\ntool-life RUL\n$\\widehat{R}(x_t)$",B,BE)
box(0.64,0.56,0.16,0.30,"A3 Planner\ncost-aware rule\nact if $\\widehat{R}\\leq h$",B,BE)
box(0.64,0.08,0.16,0.30,"A4 Explainer\nattribution drivers\n+ consistency gate",B,BE)
arr(0.42,0.71,0.45,0.71); ax.text(0.435,0.735,"risk",fontsize=7.2,ha="center")
arr(0.61,0.71,0.64,0.71); ax.text(0.625,0.735,"RUL",fontsize=7.2,ha="center")
arr(0.72,0.56,0.72,0.38,color=BE); ax.text(0.732,0.47,"proposed action",fontsize=7.2,ha="left")
arr(0.205,0.78,0.26,0.74,color=YE)  # benchmark data -> detector
arr(0.205,0.20,0.45,0.58,color=YE,rad=-0.15)  # simulator -> prognostic
# gate back to planner / action
box(0.855,0.32,0.135,0.30,"Maintenance\naction\n(replace or\ncontinue)",G,GE,fs=7.0)
arr(0.80,0.30,0.86,0.40,color=GE)
ax.text(0.895,0.27,"gated\nrecommendation",fontsize=7.0,color=GE,ha="center",va="top")
ax.text(0.43,0.23,"gate: act only when attribution across\nthe model ensemble identifies tool wear\nas a top-2 driver (wear-consistent\nexplanation)",fontsize=6.8,color="#D55E00",ha="center")
arr(0.515,0.16,0.635,0.14,color="#D55E00",rad=0.2,lw=1.0)
ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.savefig(FIG+"/fig_architecture.pdf",bbox_inches="tight")
fig.savefig(FIG+"/fig_architecture.png",bbox_inches="tight",dpi=300)
print("done")


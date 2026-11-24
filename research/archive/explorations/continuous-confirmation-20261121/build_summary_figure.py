"""Compact primary confirmation figure from frozen-analysis outputs."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'confirmation/results/paired_contrasts.csv').open()))
sel=[next(r for r in rows if r['geometry']==g and r['model']=='pb' and r['budget']=='8192' and r['primary']=='1') for g in ('original','ellipse')]
mean=-np.array([float(r['mean_difference']) for r in sel])*1000
lo=-np.array([float(r['ci97_5_high']) for r in sel])*1000;hi=-np.array([float(r['ci97_5_low']) for r in sel])*1000
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.3,2.6),layout='constrained')
ax.errorbar(mean,[1,0],xerr=np.array([mean-lo,hi-mean]),fmt='o',markersize=8,color='#0F6C73',capsize=5,lw=2)
ax.set_yticks([1,0],['Original rings','Known-coordinate ellipse']);ax.axvline(0,color='#888888',lw=1);ax.set_xlim(-.1,4.8);ax.set_ylim(-.55,1.55)
ax.set_xlabel('Loss reduction over the same confidence mask + half-turn (milli-nats/input)')
ax.set_title('Fresh PB models: 8,192 calibration forwards, exactly 64 extra views per 256 inputs',fontsize=11)
for y,label in [(1,'16 / 16 improved'),(0,'13 / 16 improved')]:ax.text(4.05,y,label,va='center',fontsize=10)
ax.grid(axis='x',alpha=.15);out=ROOT/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'confirmation_primary.png',dpi=220);fig.savefig(out/'confirmation_primary.pdf');plt.close(fig)

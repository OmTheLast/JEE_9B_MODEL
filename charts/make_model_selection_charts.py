"""Rebuild the public aggregate-only model-selection charts."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent

def pilot():
    rows=list(csv.DictReader((ROOT/'model_selection_pilot24.csv').open()))
    lookup={(r['cell'],r['condition']):r for r in rows}
    keys=['chemistry/advanced','chemistry/main','mathematics/advanced','mathematics/main','physics/advanced','physics/main']
    labels=['Chem\nAdvanced','Chem\nMain','Math\nAdvanced','Math\nMain','Physics\nAdvanced','Physics\nMain']
    x=np.arange(6); width=.34; colors=('#278166','#17394a')
    four=[int(lookup[k,'qwen35-4b']['correct']) for k in keys]; nine=[int(lookup[k,'qwen35-9b-control']['correct']) for k in keys]
    fc=[int(lookup[k,'qwen35-4b']['caps']) for k in keys]; nc=[int(lookup[k,'qwen35-9b-control']['caps']) for k in keys]
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for ax,a,b,title,ylabel in [(axes[0],four,nine,'Strict answers by subject and exam','Correct / 4'),(axes[1],fc,nc,'Generation reached the 4,096-token limit','Caps / 4')]:
        p=ax.bar(x-width/2,a,width,label='Qwen3.5-4B',color=colors[0]); q=ax.bar(x+width/2,b,width,label='Qwen3.5-9B control',color=colors[1])
        ax.set_title(title,loc='left',fontweight='bold'); ax.set_ylabel(ylabel); ax.set_ylim(0,4.5); ax.set_yticks(range(5)); ax.grid(axis='y',alpha=.22); ax.bar_label(p); ax.bar_label(q)
    axes[0].legend(frameon=False,ncol=2); axes[1].set_xticks(x,labels)
    fig.suptitle('24-question reused development pilot: 4B is close overall, but fails Chemistry Advanced',fontsize=17,fontweight='bold',y=.98)
    fig.subplots_adjust(left=.08,right=.98,top=.90,bottom=.15,hspace=.30); fig.text(.5,.025,'Reused 2025 development questions; not an untouched or official JEE benchmark. Procedure review pending.',ha='center',color='#52656d')
    fig.savefig(ROOT/'model_selection_pilot24.png',dpi=180,bbox_inches='tight'); fig.savefig(ROOT/'model_selection_pilot24.svg',bbox_inches='tight'); plt.close(fig)

def completion():
    rows=list(csv.DictReader((ROOT/'completion_control_pilot24.csv').open())); labels=['Raw single pass','Natural-language\ncontroller','Forced serializer +\nformat normalizer']
    four=[int(r['qwen35_4b_correct']) for r in rows]; nine=[int(r['qwen35_9b_correct']) for r in rows]; x=np.arange(3); width=.34
    fig,ax=plt.subplots(figsize=(11,6.4)); p=ax.bar(x-width/2,four,width,label='Qwen3.5-4B',color='#278166'); q=ax.bar(x+width/2,nine,width,label='Qwen3.5-9B control',color='#17394a')
    ax.bar_label(p,padding=3); ax.bar_label(q,padding=3); ax.set_ylim(0,24); ax.set_yticks(range(0,25,4)); ax.set_ylabel('Strict correct answers / 24'); ax.set_xticks(x,labels); ax.grid(axis='y',alpha=.22); ax.legend(frameon=False)
    ax.set_title('Answer commitment recovered supported results without changing weights',loc='left',fontweight='bold',fontsize=16); fig.subplots_adjust(left=.09,right=.98,top=.88,bottom=.20)
    fig.text(.5,.045,'Reused 2025 development questions. Serializer is trigger-only on capped drafts; this is a harness gain, not a training or benchmark gain.',ha='center',color='#52656d')
    fig.savefig(ROOT/'completion_control_pilot24.png',dpi=180,bbox_inches='tight'); fig.savefig(ROOT/'completion_control_pilot24.svg',bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': pilot(); completion()

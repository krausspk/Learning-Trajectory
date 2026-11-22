"""Figures for initialization retrieval from fixed-probe CKA representations."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATASETS=('mnist','fashion_mnist','kmnist')
NAMES={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST'}
COLORS={'mnist':'#4477AA','fashion_mnist':'#CC6677','kmnist':'#C78C19'}
KEYS=('one_minus_cka|hidden|none|raw','one_minus_cka|hidden|none|standardized','one_minus_cka|logits|none|raw')
LABELS=('Raw hidden CKA','Standardized hidden CKA','Raw logit CKA')
SHORT=('Raw hidden','Standardized\nhidden','Logits')
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':12,
       'axes.titleweight':'semibold','xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
       'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':250}


def _save(fig,out,name):
    paths=[]
    for suffix in ('png','pdf'):
        path=out/f'{name}.{suffix}';fig.savefig(path,bbox_inches='tight',facecolor='white');paths.append(path)
    plt.close(fig)
    return paths


def _rows(table,dataset,key,checkpoint=None):
    rows=table[table.dataset.eq(dataset)&table.metric_key.eq(key)]
    if checkpoint is not None:rows=rows[rows.checkpoint.eq(checkpoint)]
    if rows.empty or rows.checkpoint.duplicated().any():raise ValueError(f'Missing/duplicate figure input {dataset}/{key}/{checkpoint}')
    return rows.sort_values('checkpoint')


def _footer(fig,text):
    fig.text(.5,.015,text,ha='center',va='bottom',fontsize=8.5,color='#475363')


def endpoint_matrices(out,table):
    fig,axes=plt.subplots(1,3,figsize=(14,5.7))
    fig.subplots_adjust(left=.06,right=.915,bottom=.22,top=.8,wspace=.25)
    fig.suptitle('Initialization retrieval from raw hidden representations at E=20',fontsize=14,y=.98)
    for dataset,ax in zip(DATASETS,axes):
        with np.load(out/f'data/{dataset}_retrieval.npz',allow_pickle=False) as archive:
            k=archive['metric_keys'].astype(str).tolist().index(KEYS[0]);ranks=rankdata(archive['distances'][k,20],axis=1,method='average')
            ids=archive['run_ids'].astype(str).tolist()
        stat=_rows(table,dataset,KEYS[0],20).iloc[0]
        im=ax.imshow(ranks,cmap='viridis_r',vmin=1,vmax=10,interpolation='nearest')
        for i in range(10):
            ax.add_patch(Rectangle((i-.47,i-.47),.94,.94,fill=False,edgecolor='white',linewidth=2.8))
            ax.add_patch(Rectangle((i-.47,i-.47),.94,.94,fill=False,edgecolor='black',linewidth=1.2))
            for j in range(10):ax.text(j,i,f'{ranks[i,j]:g}',ha='center',va='center',fontsize=7.5,color='white' if ranks[i,j]>=6 else '#172028')
        ax.set_xticks(range(10),ids,rotation=55,ha='right');ax.set_yticks(range(10),ids)
        ax.set(title=f'{NAMES[dataset]}\nMean rank {stat.mean_rank:.1f}; top-1 {stat.top1:.0%}',xlabel='Candidate initial run (E=0)',ylabel='Current query run (E=20)')
    cax=fig.add_axes([.94,.245,.015,.52]);fig.colorbar(im,cax=cax,ticks=[1,3,5.5,8,10]).set_label('Candidate rank within each query (lower is better)')
    _footer(fig,'Each cell ranks 1 − biased centered linear CKA on the same 3,000 fixed probe inputs; black outlines identify the true parent after scoring.\nNo neuron alignment or weight-distance matching. 10 candidate initializations per dataset; chance mean rank 5.5 and top-1 10%.')
    return _save(fig,out,'representation_endpoint_matrices')


def over_training(out,table):
    fig,axes=plt.subplots(1,3,figsize=(13.7,5.3),sharey=True)
    fig.subplots_adjust(left=.07,right=.99,bottom=.26,top=.82,wspace=.17)
    fig.suptitle('Representation-based initialization retrieval over training',fontsize=14,y=.98)
    for key,title,ax in zip(KEYS,LABELS,axes):
        for dataset in DATASETS:
            rows=_rows(table,dataset,key);x=rows.checkpoint.to_numpy()
            ax.plot(x,rows.mean_rank,color=COLORS[dataset],label=NAMES[dataset],lw=1.8,marker='o',markersize=3)
            ax.fill_between(x,np.maximum(1,rows.mean_rank_ci_low),np.minimum(10,rows.mean_rank_ci_high),color=COLORS[dataset],alpha=.13,lw=0)
        ax.axhline(5.5,color='#697584',ls='--',lw=1)
        ax.set(title=title,xlabel='Checkpoint E (half-epoch steps)',xlim=(-.4,20.4),ylim=(10.2,.8))
        ax.set_xticks([0,2,5,10,15,20]);ax.set_yticks([1,3,5.5,8,10]);ax.grid(axis='y',color='#DEE4EA',lw=.6)
    axes[0].set_ylabel('Mean correct-parent rank (lower is better)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.12),ncol=3,frameon=False)
    _footer(fig,'Shading: 95% paired-run jackknife intervals, displayed within [1,10]. Dashed line: chance mean rank 5.5.\nE=0 is a self-retrieval sanity check without inference; the raw hidden E=20 endpoint is primary, other time points and variants exploratory.')
    return _save(fig,out,'representation_over_training')


def endpoint_comparison(out,table):
    fig,axes=plt.subplots(2,3,figsize=(13.7,7.5),sharey='row')
    fig.subplots_adjust(left=.075,right=.99,bottom=.18,top=.84,wspace=.15,hspace=.37)
    fig.suptitle('Endpoint comparison across representation definitions',fontsize=14,y=.98)
    for col,dataset in enumerate(DATASETS):
        primary=_rows(table,dataset,KEYS[0],20).iloc[0]
        axes[0,col].set_title(f'{NAMES[dataset]}\nPrimary Holm p = {primary.p_holm_less:.4g}',fontsize=11)
        for row,(statistic,chance,limits) in enumerate((('mean_rank',5.5,(10.2,.8)),('top1',.1,(-.03,1.03)))):
            ax=axes[row,col]
            for x,key in enumerate(KEYS):
                item=_rows(table,dataset,key,20).iloc[0];estimate=item[statistic]
                lo,hi=item[statistic+'_ci_low'],item[statistic+'_ci_high']
                lo=max(1 if row==0 else 0,lo);hi=min(10 if row==0 else 1,hi)
                ax.plot([x,x],[lo,hi],color=COLORS[dataset],lw=1.5)
                ax.plot(x,estimate,'D' if x==0 else 'o',color=COLORS[dataset],ms=6)
                ax.annotate(f'{estimate:.1f}' if row==0 else f'{estimate:.0%}',(x,estimate),xytext=(7,0),textcoords='offset points',va='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.4))
            ax.axhline(chance,color='#697584',lw=1,ls='--');ax.grid(axis='y',color='#DEE4EA',lw=.6)
            ax.set(ylim=limits,xlim=(-.45,2.5));ax.set_xticks(range(3),SHORT)
        axes[0,col].set_yticks([1,3,5.5,8,10]);axes[1,col].set_yticks([0,.1,.25,.5,.75,1],['0','10%','25%','50%','75%','100%'])
    axes[0,0].set_ylabel('Mean correct-parent rank\n(lower is better)');axes[1,0].set_ylabel('Correct top-1 fraction\n(higher is better)')
    _footer(fig,'Points: E=20 estimates; bars: 95% paired-run jackknife intervals, clipped only for display. Diamonds identify the prespecified raw-hidden endpoint.\nHolm p adjusts the three primary mean-rank tests; top-1 and alternative representations are descriptive/exploratory. No equivalence or absence-of-information test.')
    return _save(fig,out,'representation_endpoint_comparison')


def make_plots(out:Path):
    out=Path(out).resolve();table=pd.read_csv(out/'retrieval_by_dataset.csv')
    paths=[]
    with plt.rc_context(STYLE):
        paths+=endpoint_matrices(out,table);paths+=over_training(out,table);paths+=endpoint_comparison(out,table)
    return paths


if __name__=='__main__':
    for path in make_plots(Path(__file__).resolve().parents[1]):print(path)

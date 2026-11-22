"""Plots of candidate retrieval; only reads completed inference outputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATASETS = ('mnist','fashion_mnist','kmnist')
NAMES = {'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST'}
COLORS = {'mnist':'#4477AA','fashion_mnist':'#CC6677','kmnist':'#C78C19'}
PRIMARY = 'normalized_frobenius|total|optimal|canonical'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':230})


def primary(frame):
    result = frame
    for field,value in zip(('metric','layer','alignment','gauge'),PRIMARY.split('|')):
        result = result[result[field].eq(value)]
    return result


def save(fig,out,name):
    for suffix in ('png','pdf'):
        fig.savefig(out/f'{name}.{suffix}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_plots(out:Path):
    out=Path(out)
    table=primary(pd.read_csv(out/'retrieval_by_dataset.csv'))
    fig,axes=plt.subplots(1,3,figsize=(14.2,5.5))
    fig.subplots_adjust(bottom=.19,top=.78,wspace=.36)
    fig.suptitle('Can the trained network identify its own initialization?',fontsize=15,y=.98)
    for ax,dataset in zip(axes,DATASETS):
        with np.load(out/'data'/f'{dataset}_retrieval.npz') as z:
            index=list(z['metric_keys'].astype(str)).index(PRIMARY)
            distances=z['distances'][index,20]
            ids=z['run_ids'].astype(str)
        ranks=rankdata(distances,axis=1,method='average')
        row=table[(table.dataset==dataset)&(table.checkpoint==20)].iloc[0]
        im=ax.imshow(ranks,vmin=1,vmax=10,cmap='viridis_r',interpolation='nearest')
        for i in range(10):
            ax.add_patch(plt.Rectangle((i-.46,i-.46),.92,.92,fill=False,ec='#f75b4c',lw=1.7))
        ax.set_xticks(range(10),ids,rotation=90);ax.set_yticks(range(10),ids)
        ax.set(xlabel='Candidate initialization (E0)',ylabel='Query network (E20)')
        ax.set_title(f'{NAMES[dataset]}\nMean true rank = {row.mean_rank:.2f}; top-1 = {row.top1:.0%}\nHolm p = {row.p_holm_less:.4g}',fontsize=11,pad=12)
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,pad=.03,label='Candidate rank within query (1 = closest)',ticks=[1,3,5,7,10])
    fig.text(.5,.025,'All 100 query/candidate combinations treated identically. Red boxes mark the true initialization after matching.\nCanonical gauge, unit total weight norm, joint optimal neuron permutation. Chance: mean rank 5.5; top-1 10%.',ha='center',fontsize=9)
    save(fig,out,'retrieval_endpoint_matrices')
    fig,axes=plt.subplots(1,3,figsize=(14.2,5.2))
    fig.subplots_adjust(bottom=.23,top=.85,wspace=.31)
    specs=[('mean_rank','Mean true-candidate rank',5.5,(10.1,.9)),('top1','Top-1 retrieval rate',.1,(-.04,1.04)),('mrr','Mean reciprocal rank',sum(1/i for i in range(1,11))/10,(0,1.04))]
    for ax,(stat,label,chance,limits) in zip(axes,specs):
        for dataset in DATASETS:
            rows=table[table.dataset.eq(dataset)].sort_values('checkpoint')
            y=rows[stat].to_numpy(float);x=rows.checkpoint.to_numpy()
            lo=rows[f'{stat}_ci_low'].to_numpy(float);hi=rows[f'{stat}_ci_high'].to_numpy(float)
            ax.plot(x,y,'o-',ms=3,lw=1.6,label=NAMES[dataset],color=COLORS[dataset])
            bound=(1,10) if stat=='mean_rank' else (0,1)
            ax.fill_between(x,np.maximum(lo,bound[0]),np.minimum(hi,bound[1]),alpha=.12,color=COLORS[dataset])
        ax.axhline(chance,color='gray',ls='--',lw=1,label='Chance')
        ax.set(xlabel='Checkpoint E (half epochs)',ylabel=label,ylim=limits,xlim=(-.4,20.4))
        ax.set_xticks([0,1,2,5,10,15,20]);ax.grid(alpha=.2)
        if table.groupby('checkpoint')[stat].nunique().eq(1).all():
            ax.text(.5,.74,'All three dataset curves coincide',transform=ax.transAxes,ha='center',fontsize=9,bbox=dict(fc='white',ec='#dddddd'))
    axes[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('Initialization retrieval throughout training',fontsize=15)
    fig.text(.5,.03,'10 networks per dataset. Shading: chance-adjusted delete-one-network jackknife, transformed back to the full 10-candidate scale.\nZero-width intervals at perfect retrieval are not population certainty. E0: sanity only. MRR chance shown for untied ranks.',ha='center',fontsize=8.5)
    save(fig,out,'retrieval_over_training')
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.8),sharey=True)
    fig.subplots_adjust(bottom=.24,top=.85,wspace=.2)
    for ax,dataset in zip(axes,DATASETS):
        with np.load(out/'data'/f'{dataset}_retrieval.npz') as z:
            d=z['distances'][list(z['metric_keys'].astype(str)).index(PRIMARY),20]
            ids=z['run_ids'].astype(str)
        rank=np.diag(rankdata(d,axis=1,method='average'))
        ax.bar(np.arange(10),rank,color=COLORS[dataset],alpha=.8)
        ax.axhline(5.5,color='gray',ls='--',lw=1)
        ax.set_xticks(np.arange(10),ids,rotation=45);ax.set(title=NAMES[dataset],xlabel='Query network',ylim=(0,10.5))
        for i,value in enumerate(rank):ax.text(i,value+.12,f'{value:g}',ha='center',fontsize=9)
    axes[0].set_ylabel('Rank of true initialization (smaller is better)')
    fig.suptitle('Every network remains visible in the endpoint analysis',fontsize=15)
    fig.text(.5,.025,'Descriptive ranks for all 30 queries. Dashed line: expected rank 5.5 under random identity mapping.\nInference permutes the complete identity mapping within each dataset; it does not treat candidate distances as independent observations.',ha='center',fontsize=8.5)
    save(fig,out,'retrieval_endpoint_ranks')

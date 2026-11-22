"""Five figure families for the complete crossed initialization/shuffle study."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS=('mnist','fashion_mnist','kmnist')
NAMES={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST'}
COLORS={'mnist':'#4477AA','fashion_mnist':'#CC6677','kmnist':'#C78C19'}
WKEY='normalized_frobenius|total|optimal|canonical';CKEY='one_minus_cka|hidden|none|raw'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':230,'pdf.fonttype':42})

def save(fig,out,name):
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def read_arrays(out):
    result={}
    for ds in DATASETS:
        with np.load(out/'data'/f'{ds}_distances.npz') as z:result[ds]={k:z[k] for k in ('endpoint_distances','endpoint_cka_distances','accuracy','loss','run_ids')}
    return result

def training_plot(out,arrays):
    fig,axes=plt.subplots(2,3,figsize=(14,10))
    fig.subplots_adjust(top=.91,bottom=.11,hspace=.30,wspace=.42)
    for row,metric in enumerate(('accuracy','loss')):
        for col,ds in enumerate(DATASETS):
            ax=axes[row,col];v=arrays[ds][metric][20]
            im=ax.imshow(v,aspect='auto',cmap='viridis' if metric=='accuracy' else 'magma_r')
            ax.set_xticks(range(5),[f'S{i}' for i in range(1,6)]);ax.set_yticks(range(10),[f'I{i}' for i in range(1,11)])
            ax.set(xlabel='Shuffle schedule',ylabel='Initialization',title=f'{NAMES[ds]}: test {metric}')
            mid=(v.min()+v.max())/2
            for i in range(10):
                for s in range(5):ax.text(s,i,f'{v[i,s]:.3f}',ha='center',va='center',fontsize=7.3,color='white' if (v[i,s]<mid)==(metric=='accuracy') else 'black')
            fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle('Complete 10 x 5 factorial training outcomes (E20)',fontsize=15)
    fig.text(.5,.025,'150 runs: 10 exact initial states x 5 independent shuffle schedules x 3 datasets. Same architecture and 10-epoch SGD protocol.\nIndependent panel scales; every cell is one deterministic training run, not a replicate within a cell.',ha='center',fontsize=9)
    save(fig,out,'factorial_training')

def matrix_plot(out,arrays):
    fig,axes=plt.subplots(2,3,figsize=(14,10))
    fig.subplots_adjust(top=.91,bottom=.1,hspace=.30,wspace=.35)
    for row,(field,label) in enumerate((('endpoint_distances','Canonical unit weights'),('endpoint_cka_distances','Hidden representation: 1-CKA'))):
        for col,ds in enumerate(DATASETS):
            ax=axes[row,col];v=arrays[ds][field].reshape(50,50)
            im=ax.imshow(v,cmap='viridis',interpolation='nearest')
            ax.set_xticks(np.arange(2,50,5),range(1,11));ax.set_yticks(np.arange(2,50,5),range(1,11))
            for b in np.arange(4.5,49.5,5):ax.axhline(b,c='white',lw=.4,alpha=.6);ax.axvline(b,c='white',lw=.4,alpha=.6)
            ax.set(xlabel='Initialization (5 shuffles per block)',ylabel='Initialization (5 shuffles per block)',title=f'{NAMES[ds]}\n{label}',aspect='equal')
            fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle('Endpoint geometry of all 50 networks per dataset',fontsize=15)
    fig.text(.5,.025,'Network order: initialization first, shuffle second. Each 5 x 5 diagonal block shares an initialization.\nWeight distance and representation distance define different geometries; color scales are separate. Self-distances are included only for display.',ha='center',fontsize=9)
    save(fig,out,'endpoint_retrieval_matrices')

def factor_plot(out):
    tables=[pd.read_csv(out/'initialization_retrieval.csv'),pd.read_csv(out/'shuffle_retrieval.csv')]
    fig,axes=plt.subplots(1,2,figsize=(13,5.8))
    fig.subplots_adjust(top=.83,bottom=.24,wspace=.25)
    for ax,frame,factor in zip(axes,tables,('Initialization across shuffle schedules','Shuffle schedule across initializations')):
        for k,(key,color,label) in enumerate(((WKEY,'#4477AA','Weights'),(CKEY,'#CC6677','Hidden CKA'))):
            rows=pd.concat([frame[(frame.dataset==ds)&(frame.metric_key==key)] for ds in DATASETS]);x=np.arange(3)+(k-.5)*.30
            y=rows.rank_score.to_numpy();lo=np.maximum(0,rows.rank_score_ci_low.to_numpy());hi=np.minimum(1,rows.rank_score_ci_high.to_numpy())
            ax.bar(x,y,width=.28,color=color,alpha=.8,label=label)
            ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',ecolor='#23313d',capsize=3,lw=1)
            for xx,r in zip(x,rows.itertuples()):ax.text(xx,1.025,f'{100*r.top1:.0f}%',ha='center',fontsize=9,rotation=0)
        ax.axhline(.5,color='gray',ls='--',lw=1);ax.set_xticks(range(3),[NAMES[d] for d in DATASETS]);ax.set(ylim=(0,1.1),ylabel='Normalized rank score (chance = 0.5)',title=factor)
        ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right')
    fig.suptitle('Which factor can be identified across changes of the other?',fontsize=15)
    fig.text(.5,.03,'Labels above bars: top-1 rate. Chance is 10% for initialization and 20% for shuffle retrieval.\nIntervals: factor-block jackknife (10 initialization or 5 shuffle levels), conditional on the other chosen levels.\nOnly weight-based initialization retrieval is primary; CKA and shuffle results are secondary. Perfect-sample intervals can have zero width.',ha='center',fontsize=9)
    save(fig,out,'factor_retrieval_results')

def trajectory_plot(out):
    frame=pd.read_csv(out/'initial_trajectory_retrieval.csv');fig,axes=plt.subplots(1,2,figsize=(12,5.3))
    fig.subplots_adjust(top=.84,bottom=.24,wspace=.24)
    for ax,stat,label,chance,limits in zip(axes,('mean_rank','top1'),('Mean true-initialization rank','Top-1 retrieval rate'),(5.5,.1),((10.1,.9),(-.03,1.06))):
        for ds in DATASETS:
            rows=frame[frame.dataset==ds].sort_values('checkpoint');x=rows.checkpoint.to_numpy();y=rows[stat].to_numpy()
            ax.plot(x,y,'o-',lw=1.7,ms=3,color=COLORS[ds],label=NAMES[ds]);lower,upper=(1,10) if stat=='mean_rank' else (0,1)
            ax.fill_between(x,np.maximum(lower,rows[stat+'_ci_low'].to_numpy()),np.minimum(upper,rows[stat+'_ci_high'].to_numpy()),color=COLORS[ds],alpha=.12)
        ax.axhline(chance,color='gray',ls='--');ax.set(xlabel='Checkpoint E (half epochs)',ylabel=label,ylim=limits,xlim=(-.4,20.4));ax.set_xticks([0,1,2,5,10,15,20]);ax.grid(alpha=.2)
        if frame.groupby('checkpoint')[stat].nunique().eq(1).all():ax.text(.5,.7,'The three dataset curves coincide',transform=ax.transAxes,ha='center',fontsize=9)
    axes[0].legend(loc='lower right',fontsize=9)
    fig.suptitle('Initial-state retrieval across independently varied training order',fontsize=15)
    fig.text(.5,.03,'50 current networks per dataset, each compared with the 10 original initializations. Scores first average over shuffles within each initialization.\nUncertainty deletes complete initialization groups (all five schedules), not individual pair values. E0 is sanity only.',ha='center',fontsize=9)
    save(fig,out,'initial_trajectory_retrieval')

def geometry_plot(out):
    groups=pd.read_csv(out/'distance_groups.csv');perf=pd.read_csv(out/'performance_decomposition.csv');perf=perf[perf.checkpoint==20]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    fig.subplots_adjust(top=.9,bottom=.15,hspace=.42,wspace=.24)
    names=['same_initialization_different_shuffle','different_initialization_same_shuffle','different_initialization_different_shuffle']
    labels=['Same init, different shuffle','Different init, same shuffle','Different init and shuffle']
    colors=['#4477AA','#CC6677','#999999']
    for ax,key,title in zip(axes[0],(WKEY,CKEY),('Canonical unit weight distances','Hidden 1-CKA distances')):
        for k,(group,label,color) in enumerate(zip(names,labels,colors)):
            rows=pd.concat([groups[(groups.dataset==ds)&(groups.metric_key==key)&(groups.group==group)] for ds in DATASETS]);ax.bar(np.arange(3)+(k-1)*.24,rows['mean'],width=.23,color=color,label=label)
        ax.set_xticks(range(3),[NAMES[d] for d in DATASETS]);ax.set(title=title,ylabel='Mean pair distance');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,loc='upper left')
    axes[0,0].set_ylim(top=axes[0,0].get_ylim()[1]*1.3)
    for ax,metric in zip(axes[1],('accuracy','loss')):
        rows=pd.concat([perf[(perf.dataset==ds)&(perf.metric==metric)] for ds in DATASETS]);bottom=np.zeros(3)
        for field,label,color in [('initialization_fraction','Initialization','#4477AA'),('shuffle_fraction','Shuffle','#CC6677'),('interaction_residual_fraction','Interaction / remainder','#999999')]:
            values=rows[field].to_numpy();ax.bar(range(3),values,bottom=bottom,color=color,label=label)
            for x,v,b in zip(range(3),values,bottom):
                if v>.07:ax.text(x,b+v/2,f'{v:.0%}',ha='center',va='center',color='white',fontsize=9)
            bottom+=values
        ax.set_xticks(range(3),[NAMES[d] for d in DATASETS]);ax.set(title=f'Descriptive {metric} sum-of-squares decomposition',ylabel='Fraction of total variation',ylim=(0,1.06));ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
    fig.suptitle('Geometry and training performance answer different questions',fontsize=15)
    fig.text(.5,.025,'Distance-group means are descriptive; dependent pair counts are 100, 225 and 900 per dataset. Blocked contrast tests are in the CSV tables.\nPerformance SS: initialization 9 df, shuffle 4 df, interaction/remainder 36 df; these fractions are not random variance-component estimates.\nOne run per factorial cell: interaction and residual are not separately estimable. No ANOVA p-values are claimed.',ha='center',fontsize=9)
    save(fig,out,'factor_geometry_and_performance')

def make_plots(out:Path):
    out=Path(out);arrays=read_arrays(out)
    training_plot(out,arrays);matrix_plot(out,arrays);factor_plot(out);trajectory_plot(out);geometry_plot(out)
    sources=list((out/'data').glob('*_distances.npz'))+[out/n for n in ['initialization_retrieval.csv','shuffle_retrieval.csv','initial_trajectory_retrieval.csv','distance_groups.csv','performance_decomposition.csv']]
    (out/'plot_sources.json').write_text(json.dumps({'sources':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'plotting_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figure_pairs':5,'visual_review':'pending'},indent=2)+'\n')

if __name__=='__main__':make_plots(Path(__file__).resolve().parents[1])

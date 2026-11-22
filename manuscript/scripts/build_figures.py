#!/usr/bin/env python3
"""Rebuild manuscript figures from the immutable local CSV snapshots.

No models are trained and no inferential statistics are recomputed here.
All intervals are those in the source analyses. Figure-specific notes and
source hashes are recorded in provenance/figure_manifest.json.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'figure'
OUT.mkdir(exist_ok=True)
DS = ['mnist', 'fashion_mnist', 'kmnist']
NAMES = ['MNIST', 'Fashion-MNIST', 'KMNIST']
COLORS = ['#2478A6', '#C76347', '#558760']
WEIGHT, REPR, GRAY = '#2478A6', '#B05C80', '#7D858C'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
    'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.8,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'axes.axisbelow': True,
})
manifest = []

def read(name):
    return pd.read_csv(DATA / name)

def panel(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', pad=11, fontweight='bold')
    ax.grid(axis='y', alpha=.15, lw=.6)

def save(fig, n, sources, note, is_method=False):
    if not is_method: n += 2 if n < 5 else 3
    jpg = OUT / f'figure{n}.jpg'
    fig.savefig(jpg, dpi=400, pil_kwargs={'quality': 97, 'subsampling': 0})
    plt.close(fig)
    manifest.append({'figure': jpg.relative_to(ROOT).as_posix(),
                     'sources': sources, 'note': note,
                     'sha256': hashlib.sha256(jpg.read_bytes()).hexdigest()})

def err(ax, x, value, low, high, color, label=None, marker='o'):
    ax.errorbar(x, value, yerr=np.vstack([np.maximum(0,value-low),np.maximum(0,high-value)]),
                fmt=marker, color=color, ms=5, capsize=3, lw=1.1, label=label)

def figure1():
    rsa, weight, rep = read('rsa.csv'), read('weight_retrieval.csv'), read('representation_retrieval.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7.3,3.35))
    fig.subplots_adjust(left=.09,right=.975,bottom=.23,top=.83,wspace=.52)
    for ds, name, col in zip(DS,NAMES,COLORS):
        s=rsa[(rsa.dataset==ds)&(rsa.metric=='frobenius')&(rsa.layer=='total')&(rsa.alignment=='optimal')&(rsa.gauge=='canonical')].sort_values('checkpoint')
        axes[0].plot(s.checkpoint/2,s.rho,color=col,label=name)
        w=weight[weight.dataset==ds].sort_values('checkpoint')
        # Offsets are unnecessary: label explicitly that curves coincide.
        axes[1].plot(w.checkpoint/2,w.mean_rank,color=col,ls=['-','--',':'][DS.index(ds)])
        r=rep[(rep.dataset==ds)&(rep.layer=='hidden')&(rep.gauge=='raw')].sort_values('checkpoint')
        axes[2].plot(r.checkpoint/2,r.mean_rank,color=col)
    panel(axes[0],'A','Ensemble geometry')
    axes[0].axhline(0,color=GRAY,ls='--',lw=.9);axes[0].set(ylim=(-.5,1.05),ylabel='Distance correlation')
    panel(axes[1],'B','Weight retrieval')
    panel(axes[2],'C','Hidden CKA')
    for ax in axes[1:]:
        ax.axhline(5.5,color=GRAY,ls='--',lw=.9)
        ax.set(ylim=(10.4,.6),yticks=[1,5.5,10],ylabel='Own-initialization rank')
    axes[1].text(5,3.1,'All datasets:\nrank 1 throughout',ha='center',fontsize=8.3)
    for ax in axes: ax.set(xlim=(0,10),xticks=[0,5,10],xlabel='Training epochs')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,.005))
    save(fig,1,['rsa.csv','weight_retrieval.csv','representation_retrieval.csv'],
         'Point estimates; dashed horizontal lines mark zero correlation or chance rank 5.5. E0 is a self-comparison.')

def figure2():
    d=read('controls.csv');d=d[d.checkpoint==20]
    protocols=[('Joint weights','normalized_frobenius|total|optimal|canonical'),
       ('Input weights, joint match','normalized_frobenius|W0|optimal|canonical'),
       ('Output weights, joint match','normalized_frobenius|W1|optimal|canonical'),
       ('Output weights, native order','normalized_frobenius|W1|identity|canonical'),
       ('Frequently active units','activity_high_normalized_frobenius|total|masked_optimal|canonical'),
       ('Large pair residuals','large_residual_rms|total|optimal|canonical')]
    values=np.array([[float(d[(d.metric_key==k)&(d.dataset==ds)].top1.iloc[0]) for ds in DS] for _,k in protocols])
    fig, ax=plt.subplots(figsize=(7.3,3.45));fig.subplots_adjust(left=.36,right=.96,top=.85,bottom=.17)
    ax.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    for i in range(values.shape[0]):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.0%}',ha='center',va='center',color='white' if values[i,j]>.6 else '#172A35',fontsize=11,fontweight='bold')
    ax.set(xticks=range(3),xticklabels=NAMES,yticks=range(6),yticklabels=[x[0] for x in protocols])
    ax.tick_params(length=0,pad=8)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('Endpoint identification under weight-space controls',loc='left',fontweight='bold',pad=13)
    fig.text(.65,.035,'Top-1 accuracy; chance = 10%',ha='center',fontsize=9)
    save(fig,2,['controls.csv'],'Descriptive endpoint top-1 rates. Layer contributions use shared canonicalization and alignment. Residuals are pair-specific costs, not historical updates.')

def figure3():
    init,shuffle,dist=read('crossed_initialization.csv'),read('crossed_shuffle.csv'),read('crossed_distances.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.3,5.7));fig.subplots_adjust(left=.1,right=.985,bottom=.14,top=.92,hspace=.61,wspace=.37)
    x=np.arange(3)
    for ax,df,letter,title in [(axes[0,0],init,'A','Identifying initialization'),(axes[0,1],shuffle,'B','Identifying batch schedule')]:
        for shift,metric,color,label in [(-.16,'normalized_frobenius',WEIGHT,'Weights'),(.16,'one_minus_cka',REPR,'Hidden CKA')]:
            s=df[df.metric==metric].set_index('dataset').loc[DS]
            ax.bar(x+shift,s.rank_score,width=.29,color=color,alpha=.88,label=label)
            err(ax,x+shift,s.rank_score.to_numpy(),s.rank_score_ci_low.to_numpy(),s.rank_score_ci_high.to_numpy(),'#26343B',marker='none')
        ax.axhline(.5,color=GRAY,ls='--',lw=1)
        ax.set(xticks=x,xticklabels=NAMES,ylim=(0,1.08),yticks=[0,.5,1],ylabel='Normalized rank score')
        panel(ax,letter,title)
    groups=['same_initialization_different_shuffle','different_initialization_same_shuffle','different_initialization_different_shuffle']
    for ax,metric,letter,title in [(axes[1,0],'normalized_frobenius','C','Weight distances'),(axes[1,1],'one_minus_cka','D','Representation distances')]:
        for j,(g,col) in enumerate(zip(groups,['#2478A6','#B05C80','#ADB4B9'])):
            s=dist[(dist.metric==metric)&(dist.group==g)].set_index('dataset').loc[DS]
            ax.bar(x+(j-1)*.24,s['mean'],width=.22,color=col)
        ax.set(xticks=x,xticklabels=NAMES,ylabel='Mean distance',ylim=(0,1.08 if metric=='normalized_frobenius' else .36))
        panel(ax,letter,title)
    axes[0,0].legend(frameon=True,facecolor='white',framealpha=.95,edgecolor='none',loc='lower right',fontsize=8)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=c) for c in ['#2478A6','#B05C80','#ADB4B9']],
        ['Same initialization','Same shuffle','Both different'],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    save(fig,3,['crossed_initialization.csv','crossed_shuffle.csv','crossed_distances.csv'],
         'Top: 95% factor-block jackknife intervals; normalized rank score has chance .5 for either candidate count. Bottom: descriptive means of dependent pairs, no pairwise SE.')

def figure4():
    d=read('perturbation_curves.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.3,5.1));fig.subplots_adjust(left=.105,right=.985,bottom=.17,top=.91,hspace=.47,wspace=.35)
    cols={.01:'#2478A6',.05:'#CF923C',.20:'#BB5364'}
    for row,mom in enumerate(['preserve','reset']):
        for eps,col in cols.items():
            s=d[(d.phase=='pilot')&(d.momentum==mom)&(d.order=='same')&np.isclose(d.epsilon,eps)].sort_values('update')
            if s.empty:raise ValueError('Missing perturbation curve')
            t=s['update'].to_numpy()/235
            for j,m in enumerate(['ratio','accuracy']):
                scale=100 if m=='accuracy' else 1
                axes[row,j].plot(t,s[m+'_mean']*scale,color=col,label=f'{eps:.0%} perturbation')
                axes[row,j].fill_between(t,s[m+'_low'].to_numpy()*scale,s[m+'_high'].to_numpy()*scale,color=col,alpha=.12,lw=0)
        baseline=d[(d.epsilon==0)&(d.momentum==mom)&(d.order=='same')].sort_values('update')
        axes[row,1].plot(baseline['update']/235,baseline.accuracy_mean*100,color='#26343B',ls='--',lw=1.2,label='Unperturbed')
        axes[row,0].axhline(1,color=GRAY,ls='--',lw=.9);axes[row,0].set(yscale='log',ylabel='Distance / initial displacement',ylim=(.85,22))
        axes[row,1].set(ylabel='Test accuracy (%)',ylim=(96.4,97.85))
        for ax in axes[row]:
            ax.set_xscale('symlog',linthresh=.2);ax.set(xticks=[0,.1,1,10],xticklabels=['0','0.1','1','10'],xlim=(0,10),xlabel='Further training epochs')
        label='Retained' if row==0 else 'Reset'
        panel(axes[row,0],'AC'[row],label+' momentum: distance')
        panel(axes[row,1],'BD'[row],label+' momentum: accuracy')
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.52,.005),fontsize=8)
    save(fig,4,['perturbation_curves.csv'],'MNIST pilot only. Bands are 95% seed bootstrap intervals after averaging five directions per seed. Same future batch schedule within each paired comparison.')

def expanded_figure():
    source='supplement/06__expanded_summary.csv'
    d=read(source);d=d[d.epsilon>0]
    assert len(d)==216 and (d.ratio_low>1).all()
    fig,axes=plt.subplots(3,3,figsize=(7.3,7.25),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.15,top=.935,hspace=.40,wspace=.18)
    colors=['#2478A6','#C76347','#558760']
    for i,ds in enumerate(DS):
        for j,cp in enumerate([2,10,20]):
            ax=axes[i,j]
            for k,(layer,col) in enumerate(zip(['W0','W1','both'],colors)):
                for mom,marker,offset in [('preserve','o',-.035),('reset','s',.035)]:
                    s=d[(d.dataset==ds)&(d.checkpoint==cp)&(d.layer==layer)&(d.momentum==mom)].sort_values('epsilon')
                    x=np.arange(4)+(k-1)*.19+offset
                    err(ax,x,s.ratio_mean.to_numpy(),s.ratio_low.to_numpy(),s.ratio_high.to_numpy(),col,marker=marker)
            ax.axhline(1,color=GRAY,ls='--',lw=.9)
            ax.set(yscale='log',ylim=(.82,115),yticks=[1,10,100],yticklabels=['1','10','100'],xticks=range(4),xticklabels=['1','5','10','20'])
            ax.set_title(f'{chr(65+i*3+j)}  {NAMES[i]} / epoch {cp//2}',fontsize=9,loc='left',fontweight='bold',pad=9)
            ax.grid(axis='y',alpha=.17)
            if j==0:ax.set_ylabel('Final / initial separation',fontsize=9)
            if i==2:ax.set_xlabel('Perturbation strength (%)',fontsize=9)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=c,lw=2,label=l) for c,l in zip(colors,['Input weights','Output weights','Both matrices'])]
    handles += [Line2D([0],[0],color='#303840',marker=m,ls='',label=l) for m,l in [('o','Retained momentum'),('s','Reset momentum')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.53,.015),fontsize=8.5)
    save(fig,7,[source],'All 216 expanded conditions. Pointwise 95% source bootstrap intervals, unadjusted across cells; ten sources and one direction per source/cell. Logarithmic vertical scale.',is_method=True)

def figure5():
    d=read('history_probes.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.5),gridspec_kw={'width_ratios':[1,1.45]})
    fig.subplots_adjust(left=.1,right=.99,bottom=.18,top=.86,wspace=.78)
    for key,label,col in [('both','Weights',WEIGHT),('functional_logits','Logits',REPR),('diagnostics','Diagnostics',GRAY)]:
        s=d[(d.condition=='reset')&(d.features==key)&(~d.negative_control)].sort_values('c_epochs')
        axes[0].plot(s.c_epochs,s.balanced_accuracy,'o-',color=col,label=label,ms=4)
    axes[0].axhline(.5,color=GRAY,ls='--',lw=.9)
    axes[0].set(ylim=(.4,1.055),xticks=[1,5,10],xlabel='Epochs on common task C',ylabel='History decoding accuracy')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(frameon=False,fontsize=8,loc='center left',bbox_to_anchor=(0,.38))
    panel(axes[0],'A','History during task C')
    labels=['Joint weights','Residual weights','Logits','Diagnostics','Random labels']
    rows=[]
    for feature,neg in [('both',False),('covariate_adjusted_both',False),('functional_logits',False),('diagnostics',False),('both',True)]:
        rows.append(d[(d.condition=='reset')&(d.c_epochs==10)&(d.features==feature)&(d.negative_control==neg)].iloc[0])
    for j,r in enumerate(rows):
        axes[1].barh(j,r.balanced_accuracy,color=[WEIGHT,'#558760',REPR,GRAY,'#C5C9CD'][j],height=.57)
        axes[1].errorbar(r.balanced_accuracy,j,xerr=[[r.balanced_accuracy-r.ba_ci_low],[r.ba_ci_high-r.balanced_accuracy]],fmt='none',color='#26343B',capsize=3,lw=1)
        axes[1].text(1.22,j,f'{100*r.balanced_accuracy:.4g}%',va='center',ha='right',fontsize=8)
    axes[1].axvline(.5,color=GRAY,ls='--',lw=.9);axes[1].set(yticks=range(5),yticklabels=labels,xlim=(0,1.25),xticks=[0,.5,1],xlabel='Held-out decoding accuracy');axes[1].invert_yaxis();axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    panel(axes[1],'B','Controls after task C')
    save(fig,5,['history_probes.csv'],'A: point estimates. B: seed-pair bootstrap intervals conditional on the fitted probe. Residual weights remove train-fitted linear diagnostic effects. Perfect empirical intervals do not imply population certainty.')

def figure6():
    d=read('trajectory_regression.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.3,5.35));fig.subplots_adjust(left=.105,right=.98,bottom=.16,top=.92,hspace=.55,wspace=.38)
    for row,cohort in enumerate(['one_bit','multi_bit']):
        q=d[(d.cohort==cohort)&(d.layer=='both')&(d.pca_dim==8)&(d.target=='absolute')]
        for model,color,label,ls in [('ridge',WEIGHT,'Ridge','-'),('train_mean',GRAY,'Training mean','--'),('persistence','#CF923C','Known endpoint',':')]:
            s=q[q.model==model].sort_values('lag_updates')
            score=s.r2.to_numpy(); lag=s.lag_updates.to_numpy()/235
            axes[row,0].plot(lag,np.where(score>=-.25,score,np.nan),color=color,label=label,ls=ls)
            below=score<-.25
            if below.any(): axes[row,0].plot(lag[below],np.full(below.sum(),-.25),'v',color=color,ms=3.5)
        axes[row,0].axhline(0,color='#26343B',lw=.7)
        axes[row,0].set(ylabel='Absolute-position R²',xlabel='Epochs before endpoint',ylim=(-.35,1.08),yticks=[0,.5,1])
        axes[row,0].text(.03,.18,'▼  Persistence below −0.25',transform=axes[row,0].transAxes,fontsize=7,color='#98661F')
        panel(axes[row,0],'AC'[row],('One pair' if row==0 else 'Four pairs')+': earlier states')
        absolute=q[q.time_index==0]
        disp=d[(d.cohort==cohort)&(d.layer=='both')&(d.pca_dim==8)&(d.target=='displacement')&(d.time_index==0)&(d.model=='ridge')]
        vals=[absolute[absolute.model=='train_mean'].mse.iloc[0],absolute[absolute.model=='ridge'].mse.iloc[0],disp.mse.iloc[0]]
        axes[row,1].bar(range(3),vals,color=[GRAY,WEIGHT,REPR],width=.6)
        axes[row,1].set(yscale='log',ylabel='Start-state target MSE',xticks=range(3),xticklabels=['Training\nmean','Absolute\nRidge','Displacement\nRidge'])
        for i,v in enumerate(vals):axes[row,1].text(i,v*1.18,f'{v:.3f}',ha='center',fontsize=8)
        axes[row,1].set_ylim(.015,14)
        panel(axes[row,1],'BD'[row],('One pair' if row==0 else 'Four pairs')+' · start error')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,.005))
    save(fig,6,['trajectory_regression.csv'],'Primary eight-coordinate PCA space. Point estimates; downward triangles at -0.25 denote persistence scores below that value. Displacement MSE is in the same coordinate units; mean-early-minus-endpoint has exactly the absolute mean-baseline MSE.')

def figure7():
    event,diag=read('event_probes.csv'),read('multibit_diagnostics.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.3,3.35));fig.subplots_adjust(left=.10,right=.975,bottom=.22,top=.84,wspace=.50)
    q=event[(event.cohort=='multi_bit')&(event.model=='ridge_classifier')]
    for feature,color,label in [('both',WEIGHT,'Weights'),('diagnostics',GRAY,'Diagnostics')]:
        s=q[q.features==feature].sort_values('bit')
        axes[0].plot(s.bit+1,s.accuracy,'o-',color=color,label=label,ms=4)
        err(axes[0],s.bit.to_numpy()+1,s.accuracy.to_numpy(),s.accuracy_ci_low.to_numpy(),s.accuracy_ci_high.to_numpy(),color,marker='none')
    axes[0].axhline(.5,color=GRAY,ls='--',lw=.9);axes[0].set(ylim=(.25,1.04),xticks=[1,2,3,4],xlabel='Task pair (oldest → newest)',ylabel='Event decoding accuracy');axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    panel(axes[0],'A','History bits')
    axes[1].hist(diag.C_accuracy*100,bins=np.arange(0,101,10),color=REPR,edgecolor='white')
    axes[1].axvline(diag.C_accuracy.mean()*100,color='#26343B',ls='--',lw=1)
    axes[1].set(xlabel='Final C accuracy (%)',ylabel='Number of runs',xlim=(0,100),xticks=[0,50,100])
    panel(axes[1],'B','Final accuracy')
    axes[2].scatter(diag.final_sample_dead_units,diag.C_accuracy*100,s=9,color=WEIGHT,alpha=.35,linewidths=0)
    axes[2].set(xlabel='Probe-inactive units / 64',ylabel='Final C accuracy (%)',xlim=(45,65),ylim=(0,100),xticks=[48,56,64])
    panel(axes[2],'C','Hidden activity')
    save(fig,7,['event_probes.csv','multibit_diagnostics.csv'],'A: 80 held-out runs / 40 seed pairs; seed-block bootstrap CIs. B–C: all 400 multibit runs, descriptive; inactivity on 1,024 probe images only. The first two weight-decoded bits survive Holm correction.')

if __name__=='__main__':
    from method_schematics import comparison_design, decoding_design
    save(comparison_design(),1,[], 'Protocol schematic for experiments 01–06. Node positions and paths are illustrative, not measured outcomes.',is_method=True)
    save(decoding_design(),2,[], 'Protocol schematic for experiments 07–08. Four-pair orders are an illustrative complementary pair. Source plans are preserved in provenance/evidence.',is_method=True)
    for fn in [figure1,figure2,figure3,figure4,expanded_figure,figure5,figure6,figure7]:fn()
    (ROOT/'provenance'/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Created two method schematics and eight result figures.')

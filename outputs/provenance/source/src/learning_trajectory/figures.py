"""All six main and ten supplementary figures, plotted from computed CSVs.

No reported paper value is used as a plotted observation. The figure functions
retain the paper's analysis distinctions (raw/canonical, direct/benchmark,
within/between, run-level/paired-block uncertainty).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from .statistics import jackknife

DATASETS = ('mnist', 'fashion_mnist', 'kmnist')
DATASET_NAMES = {'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST', 'kmnist': 'KMNIST'}
COLORS = {'mnist': '#4477AA', 'fashion_mnist': '#EE6677', 'kmnist': '#EEAA33',
          'within': '#4477AA', 'between': '#EE6677', 'canonical_alpha': '#7766AA',
          'raw_alpha': '#4477AA', 'one_minus_cka': '#228D8A',
          'euclidean': '#228D8A', 'snd': '#D69A27',
          'identity': '#999999', 'benchmark': '#7766AA', 'git_rebasin': '#228D8A'}
LAYERS = ('W0', 'W1', 'balanced', 'size_weighted')
LAYER_NAMES = {'W0': r'$W_0$', 'W1': r'$W_1$', 'balanced': 'Layer-balanced', 'size_weighted': 'Size-weighted', 'hidden': 'Hidden'}
LAYER_COLORS = {'W0': '#4477AA', 'W1': '#EE6677', 'balanced': '#7766AA', 'size_weighted': '#228D8A'}
METHODS = ('identity', 'benchmark', 'git_rebasin')
METHOD_NAMES = {'identity': 'Unmatched', 'benchmark': 'PIF-Hungarian', 'git_rebasin': 'Git Re-Basin'}
METRIC_NAMES = {'raw_alpha': 'Raw Alpha', 'canonical_alpha': 'Canonical Alpha', 'euclidean': 'Euclidean', 'snd': 'SND', 'one_minus_cka': r'Hidden $1-\mathrm{CKA}$'}
MAIN_CHECKPOINTS = tuple(range(21))
TITLES = {
    'figure_01': 'Dataset-dependent clustering in aligned weights',
    'figure_02': 'Layer-resolved weight-space separation',
    'figure_03': 'Alignment objectives and independent functional validation',
    'figure_04': 'Different prediction rules on identical images',
    'figure_05': 'Weight-space and representation-space similarity',
    'figure_06': 'Small early updates and rapid differentiation',
    'figure_S01': 'Training diagnostics across independent runs',
    'figure_S02': 'Complete layer-by-metric trajectories',
    'figure_S03': 'Run-level contributions to dataset separation',
    'figure_S04': 'Task-control performance and seed comparisons',
    'figure_S05': 'Alpha–CKA agreement within and between groups',
    'figure_S06': 'ReLU-rescaling stress test and canonical recovery',
    'figure_S07': 'High-resolution early SGD trajectories',
    'figure_S08': 'Matching trajectories and permutation validation',
    'figure_S09': 'Activation-based matching validation',
    'figure_S10': 'Descriptive weight-sign retention',
}


class FigureData:
    """Validated input tables; missing analyses fail before any figures are saved."""
    def __init__(self, output):
        self.output = Path(output)
        names = ('estimates', 'training', 'correlations', 'run_contributions',
                 'displacements', 'early_progress', 'sign_retention', 'gauge_pairs',
                 'gauge_estimates', 'gauge_sensitivity')
        self.sources = [self.output / 'summary' / f'{name}.csv' for name in names]
        self.sources += [self.output / 'validation' / f'{name}.csv' for name in ('gauge_checks', 'alignment_stability')]
        self.sources += [self.output / 'pairs' / 'main' / f'checkpoint_{cp:04d}.csv' for cp in MAIN_CHECKPOINTS]
        self.sources += [self.output / 'pairs' / 'task' / 'checkpoint_0020.csv']
        missing = [str(path) for path in self.sources if not path.is_file()]
        if missing:
            raise FileNotFoundError('Figures require all completed analyses; missing:\n' + '\n'.join(missing))
        for name in names:
            table = pd.read_csv(self.output / 'summary' / f'{name}.csv')
            if table.empty:
                raise ValueError(f'Cannot plot empty analysis table: {name}')
            setattr(self, name, table)
        self.gauge_checks = pd.read_csv(self.output / 'validation' / 'gauge_checks.csv')
        self.alignment_stability = pd.read_csv(self.output / 'validation' / 'alignment_stability.csv')
        self._main = {}
        self._pooled = {}
        self._pooled_replicates = []
        self.task_pairs = pd.read_csv(self.output / 'pairs' / 'task' / 'checkpoint_0020.csv')

    def select(self, cohort='main', metric='canonical_alpha', layer='balanced',
               alignment='profile', group='contrast', checkpoint=None):
        df = self.estimates
        mask = df.cohort.eq(cohort) & df.metric.eq(metric) & df.layer.eq(layer)
        mask &= df.alignment.eq(alignment) & df.group.eq(group)
        if checkpoint is not None:
            mask &= df.checkpoint.eq(checkpoint)
        result = df.loc[mask].sort_values('checkpoint')
        if result.empty:
            raise ValueError(f'Missing estimates: {cohort}/{metric}/{layer}/{alignment}/{group}/{checkpoint}')
        return result

    def endpoint(self, **kwargs):
        kwargs.setdefault('checkpoint', 20)
        rows = self.select(**kwargs)
        if len(rows) != 1:
            raise ValueError(f'Expected one endpoint: {kwargs}')
        return rows.iloc[0]

    def main_pairs(self, checkpoint=20):
        if checkpoint not in self._main:
            self._main[checkpoint] = pd.read_csv(self.output / 'pairs' / 'main' / f'checkpoint_{checkpoint:04d}.csv')
        return self._main[checkpoint]

    def pooled(self, metric, layer, alignment):
        key = metric, layer, alignment
        if key not in self._pooled:
            rows = []
            for cp in MAIN_CHECKPOINTS:
                df = self.main_pairs(cp)
                df = df[df.metric.eq(metric) & df.layer.eq(layer) & df.alignment.eq(alignment)]
                if df.empty:
                    raise ValueError(f'Missing pooled measurements: {key}, checkpoint={cp}')
                groups = _groups_from_pairs(df)
                result = jackknife(df[['index_a', 'index_b']].to_numpy(int), df.value.to_numpy(), groups)
                base = dict(cohort='main', checkpoint=cp, metric=metric, layer=layer, alignment=alignment, group='all')
                rows.append(dict(base, **{k: result[k] for k in ('estimate', 'se', 'ci_low', 'ci_high')}))
                for deleted, stratum, value in zip(result['deleted'], result['strata'], result['replicates']):
                    self._pooled_replicates.append(dict(base, deleted=deleted, stratum=stratum, value=value))
            self._pooled[key] = pd.DataFrame(rows)
        return self._pooled[key]


def _groups_from_pairs(frame):
    n = int(frame[['index_a', 'index_b']].to_numpy().max()) + 1
    groups = np.empty(n, dtype=object)
    groups[:] = None
    for side in ('a', 'b'):
        for index, label in frame[[f'index_{side}', f'group_{side}']].drop_duplicates().itertuples(index=False):
            if groups[index] is not None and groups[index] != label:
                raise ValueError('Conflicting dataset label in pair tables')
            groups[index] = label
    if any(g is None for g in groups):
        raise ValueError('Pair table does not cover every training run')
    return groups


def _where(frame, **criteria):
    mask = np.ones(len(frame), dtype=bool)
    for key, value in criteria.items():
        mask &= frame[key].eq(value).to_numpy()
    result = frame.loc[mask]
    if result.empty:
        raise ValueError(f'Missing plotting data: {criteria}')
    return result


def _new(rows=1, cols=3, width=None, height=None):
    return plt.subplots(rows, cols, figsize=(width or 4.2 * cols, height or 3.5 * rows), squeeze=False, layout='constrained')


def _panels(axes):
    for i, ax in enumerate(np.asarray(axes).flat):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#DCE2EA', linewidth=.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-.10, 1.06, chr(65+i), transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')


def _line(ax, frame, label=None, color=None, x='checkpoint', early=False, **kwargs):
    frame = frame.sort_values(x)
    xx = frame[x].to_numpy(float)
    if early:
        xx = np.log2(xx + 1)
    y = frame.estimate.to_numpy(float)
    ax.plot(xx, y, color=color, label=label, linewidth=1.8, **kwargs)
    ax.fill_between(xx, frame.ci_low.to_numpy(float), frame.ci_high.to_numpy(float), color=color, alpha=.15, linewidth=0)


def _early_axis(ax, steps, max_ticks=9):
    steps = np.asarray(sorted(set(int(v) for v in steps)))
    if len(steps) > max_ticks:
        indices = np.unique(np.r_[0, 1, 2, np.linspace(3, len(steps)-1, max_ticks-3).round().astype(int)])
        steps = steps[indices]
    ax.set_xticks(np.log2(steps + 1), [str(v) for v in steps])
    ax.set_xlabel('SGD updates (log₂(s + 1) axis)')


def _student(frame, value, x='checkpoint'):
    grouped = frame.groupby(x)[value]
    result = grouped.agg(['mean', 'std', 'count']).reset_index()
    if (result['count'] < 2).any():
        raise ValueError('Student-t bands require at least two independent runs')
    se = result['std'] / np.sqrt(result['count'])
    margin = t.ppf(.975, result['count'] - 1) * se
    return pd.DataFrame({x: result[x], 'estimate': result['mean'], 'ci_low': result['mean']-margin, 'ci_high': result['mean']+margin})


def _bar(ax, x, stat, color, label=None, width=.65):
    mean = float(stat['estimate'])
    error = np.array([[mean - float(stat['ci_low'])], [float(stat['ci_high']) - mean]])
    ax.bar(x, mean, color=color, label=label, width=width, zorder=2)
    ax.errorbar(x, mean, yerr=error, fmt='none', ecolor='#273241', capsize=3, linewidth=1, zorder=3)


def _point(ax, y, stat, color, label=None):
    mean = float(stat['estimate'])
    ax.errorbar(mean, y, xerr=[[mean-float(stat['ci_low'])], [float(stat['ci_high'])-mean]], fmt='o', color=color, capsize=3, label=label, markersize=5)


def _zero(ax):
    ax.axhline(0, color='#8190A3', linestyle='--', linewidth=.8)


def _dataset_lines(ax, training, value='accuracy', thin=False, early=False):
    x = 'update' if early else 'checkpoint'
    for dataset in DATASETS:
        df = _where(training, dataset=dataset)
        if thin:
            for _, run in df.groupby('run'):
                run = run.sort_values(x)
                xx = np.log2(run[x].to_numpy()+1) if early else run[x]
                ax.plot(xx, run[value], color=COLORS[dataset], alpha=.18, linewidth=.65)
        _line(ax, _student(df, value, x=x), DATASET_NAMES[dataset], COLORS[dataset], x=x, early=early)
    ax.set_xlabel('Training checkpoint' if not early else 'SGD updates')
    ax.set_ylabel('Test accuracy' if value == 'accuracy' else 'Cross-entropy loss')
    if value == 'accuracy':
        ax.set_ylim(0, 1.03)


def _group_bars(ax, data, metric, layer, alignment='profile', cohort='main'):
    for i, group in enumerate(('within', 'between')):
        s = data.endpoint(cohort=cohort, metric=metric, layer=layer, alignment=alignment, group=group)
        _bar(ax, i, s, COLORS[group])
    ax.set_xticks([0, 1], ['Within dataset', 'Between datasets'])


def _boxpoints(ax, distributions, labels, colors, ylabel):
    positions = np.arange(len(distributions))
    ax.boxplot(distributions, positions=positions, widths=.48, showfliers=False,
               boxprops={'color':'#8592A3'}, whiskerprops={'color':'#8592A3'}, capprops={'color':'#8592A3'}, medianprops={'color':'#273241'})
    for i, (v, color) in enumerate(zip(distributions, colors)):
        v = np.asarray(v)
        # Jitter is independent of measurement rank; it conveys no extra axis.
        jitter = np.random.default_rng(1701+i).uniform(-.15,.15,len(v))
        ax.scatter(i+jitter, v, s=12 if len(v)>30 else 19, color=color, alpha=.6, zorder=3, edgecolors='none')
        ax.plot([i-.22, i+.22], [v.mean()]*2, color='#202B3A', linewidth=1.6)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)


def _fig1(d):
    fig, axes = _new(1, 2, width=10.5, height=3.7); a, b = axes.flat
    a.axis('off')
    runs = d.training[d.training.cohort.eq('main')].drop_duplicates('run')
    for i, ds in enumerate(DATASETS):
        y = .82 - i*.30
        n = int(runs.dataset.eq(ds).sum())
        a.text(.02, y, f'{DATASET_NAMES[ds]}\n{n} independent runs', transform=a.transAxes, ha='left', va='center',
               bbox={'boxstyle':'round,pad=.6', 'facecolor':'#F0F5FA', 'edgecolor':COLORS[ds]}, fontsize=10)
        a.annotate('', xy=(.67,.5), xytext=(.43,y), xycoords='axes fraction', arrowprops={'arrowstyle':'->','color':'#718196'})
    a.text(.80,.5,'Hidden-neuron\npermutation\nalignment',transform=a.transAxes,ha='center',va='center',
           bbox={'boxstyle':'round,pad=.7','facecolor':'#EDF2F7','edgecolor':'#718196'})
    n = len(runs); checkpoints = d.training[d.training.cohort.eq('main')].checkpoint.nunique()
    a.text(.5, -.03, f'{n} networks · {n*(n-1)//2} unordered pairs · {checkpoints} checkpoints',transform=a.transAxes,ha='center',fontsize=9,color='#53657C')
    _group_bars(b,d,'canonical_alpha','balanced')
    b.set_ylabel('Canonical layer-balanced Alpha')
    delta = d.endpoint()
    b.set_title(f"Between − within = {delta.estimate:.4f}\n95% CI [{delta.ci_low:.4f}, {delta.ci_high:.4f}]",fontsize=10)
    _panels(axes); return fig


def _fig2(d):
    fig, axes = _new(); a,b,c = axes.flat
    for group in ('within','between'):
        _line(a,d.select(group=group),group.title(),COLORS[group])
    a.set(xlabel='Training checkpoint',ylabel='Canonical layer-balanced Alpha'); a.legend()
    for i, layer in enumerate(('W0','W1','balanced')):
        _point(b,2-i,d.endpoint(layer=layer),COLORS['canonical_alpha'])
        for offset,metric in [(-.12,'euclidean'),(.12,'snd')]:
            _point(c,2-i+offset,d.endpoint(metric=metric,layer=layer),COLORS[metric],METRIC_NAMES[metric] if i==0 else None)
    for ax in (b,c):
        ax.set_yticks([2,1,0],[LAYER_NAMES[k] for k in ('W0','W1','balanced')]); ax.set_xlabel('Between − within'); ax.axvline(0,color='#8190A3',linestyle='--',linewidth=.8)
    b.set_title('Canonical Alpha'); c.set_title('Complementary measures'); c.legend()
    _panels(axes); return fig


def _fig3(d):
    fig, axes = _new(); a,b,c = axes.flat
    for i,method in enumerate(METHODS):
        _bar(a,i,d.pooled('weight_cosine','total',method).iloc[-1],COLORS[method])
        for j,group in enumerate(('within','between')):
            offset = -.18 if j==0 else .18
            _bar(b,i+offset,d.endpoint(alignment=method,group=group),COLORS[group],group.title() if i==0 else None,width=.32)
            _bar(c,i+offset,d.endpoint(metric='activation_correlation',layer='hidden',alignment=method,group=group),COLORS[group],width=.32)
    for ax in axes.flat:
        ax.set_xticks(range(3),[METHOD_NAMES[m] for m in METHODS],rotation=18,ha='right')
    a.set(title='Global weight agreement',ylabel='Raw weight cosine similarity')
    b.set(title='Aligned weight structure',ylabel='Canonical layer-balanced Alpha'); b.legend()
    c.set(title='Independent functional validation',ylabel='Mean activation correlation')
    _panels(axes); return fig


def _fig4(d):
    fig,axes = _new(); a,b,c=axes.flat; a.axis('off')
    n=d.training[d.training.cohort.eq('task')].seed.nunique()
    a.text(.5,.83,'Shared MNIST images',transform=a.transAxes,ha='center',bbox={'boxstyle':'round,pad=.6','facecolor':'#F0F5FA','edgecolor':'#718196'})
    a.text(.5,.52,f'{n} paired seed blocks\nIdentical initial $W_0$\nand image order',transform=a.transAxes,ha='center',va='center',bbox={'boxstyle':'round,pad=.6','facecolor':'white','edgecolor':'#718196'})
    for x,label,color in [(.22,'Digit identity\n10 classes','#7766AA'),(.78,'Parity\n2 classes','#228D8A')]:
        a.annotate('',xy=(x,.22),xytext=(.5,.40),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':color})
        a.text(x,.12,label,transform=a.transAxes,ha='center',va='center',bbox={'boxstyle':'round,pad=.5','facecolor':'#F4F6FA','edgecolor':color})
    for i,(metric,layer,alignment) in enumerate([('raw_alpha','W0','incoming'),('one_minus_cka','hidden','none')]):
        q=dict(cohort='task',metric=metric,layer=layer,alignment=alignment)
        label='Matched $W_0$ Alpha' if metric=='raw_alpha' else METRIC_NAMES[metric]
        _line(b,d.select(**q),label,COLORS[metric]); _point(c,i,d.endpoint(**q),COLORS[metric])
    b.set(xlabel='Training checkpoint',ylabel='Between-task − within-task'); b.legend(); _zero(b)
    c.set_yticks([0,1],['Matched $W_0$ Alpha',METRIC_NAMES['one_minus_cka']]); c.set_xlabel('Final task contrast'); c.axvline(0,color='#8190A3',linestyle='--',linewidth=.8)
    _panels(axes); return fig


def _alpha_cka_pairs(d,checkpoint=20):
    df=d.main_pairs(checkpoint)
    alpha=_where(df,metric='raw_alpha',layer='W0',alignment='profile')[['index_a','index_b','within','value']].rename(columns={'value':'alpha'})
    cka=_where(df,metric='one_minus_cka',layer='hidden',alignment='none')[['index_a','index_b','value']].rename(columns={'value':'cka'})
    return alpha.merge(cka,on=['index_a','index_b'],validate='one_to_one')


def _fig5(d):
    fig,axes=_new(); a,b,c=axes.flat
    for i,(metric,layer,alignment,label) in enumerate([('raw_alpha','W0','profile','Matched $W_0$ Alpha'),('one_minus_cka','hidden','none',METRIC_NAMES['one_minus_cka'])]):
        frame=d.select(metric=metric,layer=layer,alignment=alignment)
        frame=frame[frame.checkpoint.isin([0,1,2,5,10,20])]
        _line(a,frame,label,COLORS[metric])
        for j,group in enumerate(('within','between')):
            _bar(b,i+(-.18 if j==0 else .18),d.endpoint(metric=metric,layer=layer,alignment=alignment,group=group),COLORS[group],group.title() if i==0 else None,width=.32)
    a.set(xlabel='Training checkpoint',ylabel='Between − within'); a.legend(); _zero(a)
    b.set_xticks([0,1],['Matched $W_0$ Alpha',METRIC_NAMES['one_minus_cka']]); b.set_ylabel('Final dissimilarity'); b.legend()
    pair=_alpha_cka_pairs(d)
    for group,value in [('within',True),('between',False)]:
        p=pair[pair.within.eq(value)]; c.scatter(p.alpha,p.cka,s=12,alpha=.55,color=COLORS[group],label=f'{group.title()} (n={len(p)})',edgecolors='none')
    corr=_where(d.correlations,checkpoint=20,layer='W0',method='pearson')
    text='Pearson r\n'+'\n'.join(f'{g.title()}: {float(_where(corr,group=g).estimate.iloc[0]):.3f}' for g in ('all','within','between'))
    c.text(.03,.97,text,transform=c.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    c.set(xlabel='Matched $W_0$ Alpha',ylabel=METRIC_NAMES['one_minus_cka']); c.legend(loc='lower right')
    _panels(axes); return fig


def _fig6(d):
    fig,axes=_new(); a,b,c=axes.flat
    training=_where(d.training,cohort='early'); training=training[training['update']<=64]
    _dataset_lines(a,training,early=True); a.legend(loc='lower right'); a.axhline(.1,color='#8190A3',linestyle='--',linewidth=.8)
    for layer in ('W0','W1'):
        frame=_where(d.displacements,cohort='early',layer=layer); frame=frame[frame['update']<=64]
        _line(b,_student(frame,'relative_frobenius',x='update'),LAYER_NAMES[layer],LAYER_COLORS[layer],x='update',early=True)
    b.set_ylabel('Relative Frobenius displacement'); b.legend()
    for metric,layer,alignment in [('canonical_alpha','balanced','profile'),('one_minus_cka','hidden','none')]:
        frame=d.select(cohort='early',metric=metric,layer=layer,alignment=alignment); frame=frame[frame.checkpoint<=64]
        _line(c,frame,METRIC_NAMES[metric],COLORS[metric],early=True)
        onset=frame[(frame.checkpoint>0)&(frame.ci_low>0)]
        if len(onset):
            c.axvline(np.log2(onset.checkpoint.iloc[0]+1),color=COLORS[metric],linestyle=':',linewidth=1)
    c.set_ylabel('Between − within'); c.legend(loc='lower right'); _zero(c)
    for ax in axes.flat: _early_axis(ax,training['update'])
    _panels(axes); return fig


def _s1(d):
    fig,axes=_new(2,2,width=11,height=7.5); a,b,c,e=axes.flat
    frame=_where(d.training,cohort='main')
    _dataset_lines(a,frame,thin=True); a.legend(loc='lower right')
    _dataset_lines(b,frame,value='loss',thin=True)
    for ax,value,label in [(c,'accuracy','Final test accuracy'),(e,'loss','Final cross-entropy loss')]:
        final=frame[frame.checkpoint.eq(20)]
        distributions=[_where(final,dataset=ds)[value] for ds in DATASETS]
        _boxpoints(ax,distributions,[DATASET_NAMES[ds] for ds in DATASETS],[COLORS[ds] for ds in DATASETS],label)
    _panels(axes); return fig


def _s2(d):
    fig,axes=_new(3,4,width=14,height=9.5)
    for i,metric in enumerate(('euclidean','snd','raw_alpha')):
        for j,layer in enumerate(LAYERS):
            ax=axes[i,j]
            for group in ('within','between'):
                _line(ax,d.select(metric=metric,layer=layer,group=group),group.title(),COLORS[group])
            ax.set_title(LAYER_NAMES[layer]); ax.set_xlabel('Checkpoint')
            if j==0: ax.set_ylabel(METRIC_NAMES[metric])
    axes[0,-1].legend(); _panels(axes); return fig


def _s3(d):
    fig,axes=_new()
    for ax,metric in zip(axes.flat,('euclidean','snd','raw_alpha')):
        frame=_where(d.run_contributions,metric=metric,alignment='profile')
        distributions=[_where(frame,layer=layer).value.to_numpy() for layer in LAYERS]
        ax.boxplot(distributions,positions=range(4),widths=.58,showfliers=False,
                   medianprops={'color':'#273241'},boxprops={'color':'#8190A3'},whiskerprops={'color':'#8190A3'},capprops={'color':'#8190A3'})
        for i,layer in enumerate(LAYERS):
            for j,ds in enumerate(DATASETS):
                values=_where(frame,layer=layer,dataset=ds).value.to_numpy()
                xx=i+(j-1)*.14+np.linspace(-.035,.035,len(values))
                ax.scatter(xx,values,s=19,color=COLORS[ds],label=DATASET_NAMES[ds] if i==0 else None,alpha=.7,edgecolors='none')
        ax.set_xticks(range(4),[LAYER_NAMES[l] for l in LAYERS],rotation=20,ha='right')
        ax.set(title=METRIC_NAMES[metric],ylabel='Run-level between − within'); _zero(ax)
    axes[0,0].legend(fontsize=7); _panels(axes); return fig


def _task_subgroups(d,metric,layer,alignment):
    frame=_where(d.task_pairs,metric=metric,layer=layer,alignment=alignment)
    training=_where(d.training,cohort='task').drop_duplicates('run').set_index('run')
    pairs=frame[['index_a','index_b']].to_numpy(int); groups=_groups_from_pairs(frame)
    seeds=np.empty(len(groups),dtype=int)
    for side in ('a','b'):
        for i,run in frame[[f'index_{side}',f'run_{side}']].drop_duplicates().itertuples(index=False):
            seeds[i]=int(training.loc[run,'seed'])
    def statistic(v,p):
        cross=groups[p[:,0]]!=groups[p[:,1]]
        same_seed=seeds[p[:,0]]==seeds[p[:,1]]
        conditions=[~cross & (groups[p[:,0]]=='identity'),~cross & (groups[p[:,0]]=='parity'),cross & same_seed,cross & ~same_seed]
        if any(not c.any() for c in conditions): raise ValueError('Missing task seed-comparison subgroup')
        return np.array([v[c].mean() for c in conditions])
    result=jackknife(pairs,frame.value.to_numpy(),groups,statistic,blocks=seeds)
    return [{key:np.asarray(result[key])[i] for key in ('estimate','ci_low','ci_high')} for i in range(4)]


def _s4(d):
    fig,axes=_new(2,2,width=11,height=7.5); a,b,c,e=axes.flat
    training=_where(d.training,cohort='task')
    for task,color in [('identity','#7766AA'),('parity','#228D8A')]:
        _line(a,_student(_where(training,task=task),'accuracy'),task.title(),color)
    a.set(xlabel='Checkpoint',ylabel='Test accuracy',ylim=(0,1.03)); a.legend()
    comparisons=[('euclidean','W0','incoming'),('snd','W0','incoming'),('raw_alpha','W0','incoming'),('one_minus_cka','hidden','none')]
    for i,(metric,layer,alignment) in enumerate(comparisons):
        _bar(b,i,d.endpoint(cohort='task',metric=metric,layer=layer,alignment=alignment),COLORS[metric])
    b.set_xticks(range(4),['Euclidean','SND','Alpha',r'$1-\mathrm{CKA}$']); b.set_ylabel('Between-task − within-task'); _zero(b)
    labels=['Identity–identity','Parity–parity','Cross, same seed','Cross, different seed']
    for ax,metric,layer,alignment,label in [(c,'raw_alpha','W0','incoming','Matched $W_0$ Alpha'),(e,'one_minus_cka','hidden','none',METRIC_NAMES['one_minus_cka'])]:
        for i,s in enumerate(_task_subgroups(d,metric,layer,alignment)):
            _bar(ax,i,s,['#7766AA','#228D8A','#EEAA33','#EE6677'][i])
        ax.set_xticks(range(4),labels,rotation=18,ha='right'); ax.set_ylabel(label)
    _panels(axes); return fig


def _s5(d):
    fig,axes=_new(2,2,width=11,height=7.5); a,b,c,e=axes.flat
    for layer in ('W0','W1','balanced'):
        frame=_where(d.correlations,layer=layer,method='pearson',group='all')
        _line(a,frame,LAYER_NAMES[layer],LAYER_COLORS[layer])
    a.set(xlabel='Checkpoint',ylabel='Pearson correlation'); a.legend(); _zero(a)
    pair=_alpha_cka_pairs(d)
    for ax,within,title in [(b,True,'Within-dataset pairs'),(c,False,'Between-dataset pairs')]:
        p=pair[pair.within.eq(within)]
        ax.scatter(p.alpha,p.cka,s=14,alpha=.65,color=COLORS['within' if within else 'between'],edgecolors='none')
        ax.set(title=title,xlabel='Matched $W_0$ Alpha',ylabel=METRIC_NAMES['one_minus_cka'])
    for i,group in enumerate(('all','within','between')):
        for j,layer in enumerate(('W0','W1','balanced')):
            stat=_where(d.correlations,checkpoint=20,group=group,layer=layer,method='pearson').iloc[0]
            _bar(e,i+(j-1)*.24,stat,LAYER_COLORS[layer],LAYER_NAMES[layer] if i==0 else None,width=.20)
    e.set_xticks(range(3),['All','Within','Between']); e.set_ylabel('Final Pearson correlation'); e.legend(); _zero(e)
    _panels(axes); return fig


def _s6(d):
    fig,axes=_new(2,2,width=11,height=7.5); a,b,c,e=axes.flat
    sensitivity=d.gauge_sensitivity
    for layer in LAYERS:
        f=_where(sensitivity,layer=layer).sort_values('C')
        a.plot(np.log2(f.C),f.mean_absolute_change,'o-',color=LAYER_COLORS[layer],label=LAYER_NAMES[layer],markersize=4)
        b.plot(np.log2(f.C),f.correlation_with_original,'o-',color=LAYER_COLORS[layer],markersize=4)
    f=_where(d.gauge_estimates,metric='raw_alpha',layer='balanced',group='contrast').sort_values('C')
    _line(c,f.assign(logC=np.log2(f.C)),color=COLORS['raw_alpha'],x='logC',marker='o',markersize=4)
    checks=d.gauge_checks.groupby('C')[['logit_error','canonical_weight_error']].max().reset_index()
    positive=checks[['logit_error','canonical_weight_error']].to_numpy(); positive=positive[positive>0]
    floor=float(positive.min()/10) if len(positive) else np.finfo(float).eps
    for field,label,color in [('logit_error','Logit error','#4477AA'),('canonical_weight_error','Canonical-weight error','#228D8A')]:
        e.plot(np.log2(checks.C),np.maximum(checks[field],floor),'o-',color=color,label=label,markersize=4)
    e.set_yscale('log'); e.legend(); e.text(.03,.03,'Zero errors plotted at axis floor',transform=e.transAxes,fontsize=7,color='#617187')
    a.set_ylabel('Mean absolute raw Alpha change'); a.legend()
    b.set_ylabel('Correlation with unscaled Alpha'); c.set_ylabel('Raw Alpha between − within'); e.set_ylabel('Maximum absolute numerical error')
    C=sorted(d.gauge_estimates.C.unique())
    for ax in axes.flat:
        ax.set_xticks(np.log2(C),[f'{v:g}' for v in C]); ax.set_xlabel('Rescaling envelope C')
    _panels(axes); return fig


def _s7(d):
    fig,axes=_new(2,3,width=13,height=7.5)
    training=_where(d.training,cohort='early'); _dataset_lines(axes[0,0],training,early=True); axes[0,0].legend()
    for ax,(metric,layer,alignment) in zip(list(axes.flat)[1:],[('euclidean','balanced','profile'),('snd','balanced','profile'),('raw_alpha','balanced','profile'),('canonical_alpha','balanced','profile'),('one_minus_cka','hidden','none')]):
        frame=d.select(cohort='early',metric=metric,layer=layer,alignment=alignment)
        _line(ax,frame,color=COLORS[metric],early=True); ax.set(title=METRIC_NAMES[metric],ylabel='Between − within'); _zero(ax)
    for ax in axes.flat: _early_axis(ax,training['update'],max_ticks=8)
    _panels(axes); return fig


def _s8(d):
    fig,axes=_new(2,3,width=13,height=7.5)
    for ax,metric,layer,title in [(axes[0,0],'weight_cosine','total','Raw weight cosine'),(axes[0,1],'weight_rmse','total','Raw weight RMSE'),(axes[0,2],'canonical_alpha','balanced','Canonical Alpha'),(axes[1,0],'activation_correlation','hidden','Activation correlation')]:
        for method in METHODS: _line(ax,d.pooled(metric,layer,method),METHOD_NAMES[method],COLORS[method])
        ax.set(title=title,xlabel='Checkpoint',ylabel=title)
    axes[0,0].legend(fontsize=7,loc='center right',bbox_to_anchor=(1,.28))
    e=axes[1,1]; _line(e,d.pooled('assignment_agreement','hidden','benchmark_vs_git'),color='#4477AA')
    e.axhline(1/64,color='#8190A3',linestyle='--',linewidth=.8,label='Chance (1/64)'); e.legend(); e.set(xlabel='Checkpoint',ylabel='Fraction identical assignments')
    f=axes[1,2]; f.axis('off'); checks=d.alignment_stability
    text=['Permutation implementation checks',f"Pairs tested: {checks.pair.nunique()}",f"Renumberings per pair: {checks['repeat'].nunique()}"]
    for method in ('benchmark','git_rebasin'):
        subset=_where(checks,method=method)
        equality=subset.assignment_equal.astype(str).str.lower().isin(['true','1'])
        text.append(f'{METHOD_NAMES[method]} order recovery: {equality.mean():.1%}')
    text += [f'Max objective error: {checks.objective_error.abs().max():.3g}',f'Max function error: {checks.logit_error.abs().max():.3g}']
    f.text(.03,.92,'\n\n'.join(text),transform=f.transAxes,va='top',fontsize=9)
    _panels(axes); return fig


def _activation_pairs(d):
    frame=_where(d.main_pairs(),metric='activation_correlation',layer='hidden')
    return frame.pivot(index=['index_a','index_b','within'],columns='alignment',values='value').reset_index()


def _s9(d):
    fig,axes=_new(2,2,width=11,height=7.5); a,b,c,e=axes.flat
    frame=_activation_pairs(d)
    for ax,within,title in [(a,True,'Within-dataset pairs'),(b,False,'Between-dataset pairs')]:
        f=frame[frame.within.eq(within)]
        _boxpoints(ax,[f[m] for m in METHODS],[METHOD_NAMES[m] for m in METHODS],[COLORS[m] for m in METHODS],'Activation correlation')
        ax.set_title(title)
    comparisons=[('benchmark','identity','PIF > unmatched','#7766AA'),('git_rebasin','identity','Git > unmatched','#228D8A'),('git_rebasin','benchmark','Git > PIF','#4477AA')]
    for i,group in enumerate(('all','within','between')):
        f=frame if group=='all' else frame[frame.within.eq(group=='within')]
        for j,(high,low,label,color) in enumerate(comparisons):
            fraction=float((f[high]>f[low]).mean())
            c.bar(i+(j-1)*.24,fraction,width=.20,color=color,label=label if i==0 else None)
    c.set_xticks(range(3),['All','Within','Between']); c.set(ylabel='Fraction of pairs improved',ylim=(0,1.30)); c.legend(fontsize=8,loc='upper center',ncol=1)
    for method in METHODS: _line(e,d.pooled('activation_correlation','hidden',method),METHOD_NAMES[method],COLORS[method])
    e.set(xlabel='Checkpoint',ylabel='Mean activation correlation'); e.legend(loc='center right')
    _panels(axes); return fig


def _s10(d):
    fig,axes=_new(2,2,width=11,height=7.5)
    for i,layer in enumerate(('W0','W1')):
        f=_where(d.sign_retention,cohort='main',layer=layer)
        for ds in DATASETS:
            q=_where(f,dataset=ds).groupby('checkpoint').retention.mean()
            axes[0,i].plot(q.index,q.values,label=DATASET_NAMES[ds],color=COLORS[ds],linewidth=1.8)
        axes[0,i].set(title=LAYER_NAMES[layer],xlabel='Checkpoint',ylabel='Sign retention from initialization')
        distributions=[1-_where(f,dataset=ds,checkpoint=20).retention for ds in DATASETS]
        _boxpoints(axes[1,i],distributions,[DATASET_NAMES[ds] for ds in DATASETS],[COLORS[ds] for ds in DATASETS],'Final sign-change fraction')
        axes[1,i].set_title(LAYER_NAMES[layer])
    axes[0,0].legend(); _panels(axes); return fig


BUILDERS = (_fig1,_fig2,_fig3,_fig4,_fig5,_fig6,_s1,_s2,_s3,_s4,_s5,_s6,_s7,_s8,_s9,_s10)
FIGURE_NAMES = tuple(TITLES)


def _save(fig, directory, name):
    fig.suptitle(TITLES[name],fontsize=14,fontweight='bold')
    paths=[]
    for extension in ('png','svg'):
        path=directory/f'{name}.{extension}'
        fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
        paths.append(path)
    plt.close(fig)
    return paths


def make_figures(output='outputs'):
    """Create PNG/SVG for all 16 figures; fail on absent required analyses."""
    data=FigureData(output)
    directory=Path(output)/'figures'
    directory.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    style={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
           'legend.fontsize':8,'legend.frameon':False,'xtick.labelsize':8,'ytick.labelsize':8,
           'svg.fonttype':'none','axes.linewidth':.8}
    with plt.rc_context(style):
        for name,builder in zip(FIGURE_NAMES,BUILDERS):
            fig=builder(data)
            artifacts.extend(_save(fig,directory,name))
            print(f'Figure complete: {name} (PNG + SVG)',flush=True)
    sources={str(path.relative_to(data.output)):hashlib.sha256(path.read_bytes()).hexdigest() for path in data.sources}
    pd.concat(list(data._pooled.values()),ignore_index=True).to_csv(data.output/'summary'/'pooled_estimates.csv',index=False,float_format='%.15g')
    pd.DataFrame(data._pooled_replicates).to_csv(data.output/'summary'/'pooled_jackknife_replicates.csv',index=False,float_format='%.15g')
    manifest={'description':'All plotted observations are independently computed; no reported reference values substituted.',
              'figures':[{'id':name,'title':TITLES[name],'png':f'{name}.png','svg':f'{name}.svg'} for name in FIGURE_NAMES],
              'source_sha256':sources,'count':len(FIGURE_NAMES)}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return artifacts


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs')
    make_figures(parser.parse_args().output)

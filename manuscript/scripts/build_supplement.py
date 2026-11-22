#!/usr/bin/env python3
"""Generate the typeset supplement from archived results; no model fitting.

All inferential quantities are copied from source analyses. Descriptive ranges,
counts, and group means below summarize existing rows without new inference.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'supplement_tables'
OUT.mkdir(exist_ok=True)
DS = ['mnist', 'fashion_mnist', 'kmnist']
NAMES = dict(zip(DS, ['MNIST', 'Fashion', 'KMNIST']))
used = set()
tables = []

def read(name):
    used.add(name)
    return pd.read_csv(DATA / name)

def sup(name):
    return read('supplement/' + name)

def js(name):
    name = 'supplement/' + name
    used.add(name)
    return json.loads((DATA / name).read_text())

def f(v, n=3):
    return '--' if pd.isna(v) else f'{v:.{n}f}'

def pct(v, n=1):
    return f(100*v, n)

def ci(row, stem, n=3, scale=1):
    return f'{f(row[stem+"_mean"]*scale,n)} [{f(row[stem+"_low"]*scale,n)}, {f(row[stem+"_high"]*scale,n)}]'

def rng(v, n=3, scale=1):
    return f'[{f(v.min()*scale,n)}, {f(v.max()*scale,n)}]'

def cohort_name(value):
    return {'one_bit':'One-pair','multi_bit':'Four-pair'}[value]

def table(name, caption, headers, rows, spec=None):
    assert rows and all(len(x)==len(headers) for x in rows), name
    cols=spec or ('l'+'r'*(len(headers)-1))
    h=' & '.join(headers)+r' \\'
    body='\n'.join(' & '.join(map(str,x))+r' \\' for x in rows)
    text=(r'\begingroup\footnotesize\setlength{\tabcolsep}{4pt}'+'\n'
          +r'\renewcommand{\arraystretch}{1.12}'+'\n'
          +r'\begin{longtable}{@{}'+cols+r'@{}}'+'\n'
          +r'\caption{'+caption+r'}\label{tab:s-'+name+r'}\\'+'\n'
          +r'\toprule'+'\n'+h+'\n'+r'\midrule\endfirsthead'+'\n'
          +r'\multicolumn{'+str(len(headers))+r'}{l}{\textit{Table \thetable\ continued}}\\'+'\n'
          +r'\toprule'+'\n'+h+'\n'+r'\midrule\endhead'+'\n'
          +r'\midrule\multicolumn{'+str(len(headers))+r'}{r}{\textit{Continued on next page}}\\\endfoot'+'\n'
          +r'\bottomrule\endlastfoot'+'\n'+body+'\n'+r'\end{longtable}\endgroup'+'\n')
    path=OUT/(name+'.tex');path.write_text(text)
    tables.append({'file':path.relative_to(ROOT).as_posix(),'rows':len(rows),
                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

LAYER={'total':'Both','both':'Both','W0':r'$W_0$','W1':r'$W_1$',
       'balanced':'Balanced','size_weighted':'Size-wtd.'}
GAUGE={'raw':'Raw','canonical':'Canon.'}
ALIGN={'optimal':'Opt.','identity':'Native','none':'None','masked_optimal':'Masked',
       'fixed_E0':'Fixed E0','fixed_E20':'Fixed E20','alpha_profile':r'$\alpha$ profile','snd_profile':'SND profile'}
METRIC={'frobenius':'Frob.','normalized_frobenius':'Unit Frob.','cosine':'Cosine',
        'alpha2':r'$\alpha_2$','snd':'SND','norm_only':'Norm',
        'small_residual_rms':'Small resid.','large_residual_rms':'Large resid.',
        'activity_low_normalized_frobenius':'Low activity',
        'activity_high_normalized_frobenius':'High activity','norm_rank_difference':'Norm rank'}

# 01: every geometry endpoint, plus the diagnostics that qualify its interpretation.
d=sup('01__rsa_endpoints_all_metrics.csv')
assert len(d)==156
rows=[]
for keys,g in d.groupby(['metric','layer','alignment','gauge'],sort=False):
    metric,layer,align,gauge=keys
    g=g.set_index('dataset').loc[DS]
    rows.append([METRIC[metric],LAYER[layer],ALIGN[align],GAUGE[gauge]]+
                [f'{r.rho:.3f} / {r.p_perm_greater:.4f}' for _,r in g.iterrows()])
table('rsa',r'All 52 RSA endpoint variants. Each dataset entry is Spearman $\rho$ / unadjusted one-sided permutation $p$. Only the three canonical, optimally matched, joint Frobenius endpoints form the primary Holm family. All other rows are exploratory.',
      ['Distance','Layer','Assignment','Gauge','MNIST','Fashion','KMNIST'],rows,'llllrrr')

n=sup('01__norm_controls.csv'); rows=[]
pred={'norm_sum':'Sum','norm_abs_difference':'Difference','norm_sum+norm_abs_difference':'Both'}
for keys,g in n.groupby(['analysis','checkpoint','gauge','layer','predictor'],sort=False):
    analysis,cp,gauge,layer,predictor=keys;g=g.set_index('dataset').loc[DS]
    rows.append(['Distance' if analysis=='norm_predicts_distance' else 'Partial RSA',int(cp),GAUGE[gauge],LAYER[layer],pred[predictor]]+
                [f'{r.rho:.3f} / {r.p_perm_greater:.4f}' for _,r in g.iterrows()])
table('rsa-norms',r'Norm sensitivity. Entries are $\rho$ / unadjusted $p$. Distance rows relate pairwise norms to distances; partial RSA conditions initial--final distance association on both norm predictors. These are descriptive sensitivity tests, not exact conditional tests under arbitrary covariate dependence.',
      ['Analysis','E','Gauge','Layer','Norm','MNIST','Fashion','KMNIST'],rows,'lllllrrr')
n=sup('01__norm_run_correlations.csv');rows=[]
for (gauge,layer),g in n.groupby(['gauge','layer'],sort=False):
    g=g.set_index('dataset').loc[DS]
    rows.append([GAUGE[gauge],LAYER[layer]]+[f'{r.rho:.3f} / {r.p_perm_greater:.4f}' for _,r in g.iterrows()])
table('norm-persistence',r'Initial--final correlation of individual network norms. Entries are Spearman $\rho$ / unadjusted one-sided permutation $p$. These exploratory controls concern norm persistence, rather than origin identification from full weights.',
      ['Gauge','Layer','MNIST','Fashion','KMNIST'],rows,'llrrr')
c=sup('01__data__assignment_change_summary.csv');cy=sup('01__data__cycle_consistency_summary.csv')
st=sup('01__numerical_stability.csv');conc=sup('01__distance_concentration.csv')
rows=[]
for ds in DS:
    a=c[(c.dataset==ds)&(c.checkpoint==20)].iloc[0]
    b=cy[cy.dataset==ds].set_index('checkpoint')
    e=conc[(conc.dataset==ds)&(conc.checkpoint==0)&(conc.metric=='frobenius')&(conc.layer=='total')&(conc.gauge=='canonical')&(conc.alignment=='optimal')].iloc[0]
    rows.append([NAMES[ds],pct(e.cv,3),pct(a.change_fraction_vs_E0_mean,2),
                 pct(b.loc[0,'mean_fraction_consistent'],2),pct(b.loc[20,'mean_fraction_consistent'],2),
                 f(st[st.dataset==ds].float32_vs_float64_rank_rho.min(),4)])
table('rsa-diagnostics','Initial distance concentration, alignment change, and numerical stability. Cycle consistency is the fraction of neuron correspondences agreeing around a network triplet; no complete triplet was exactly consistent (120 triplets per dataset and checkpoint). Rank stability is the minimum float32--float64 Spearman correlation over E0 and E20.',
      ['Dataset','Initial CV (\%)','Changed (\%)','Cycle E0 (\%)','Cycle E20 (\%)','Rank stability'],rows)
ne=sup('01__neighbor_retention.csv')
table('neighbors','Exploratory nearest-neighbor retention. Chance retention is $k/9$. Intervals are untruncated network-jackknife intervals and may extend outside the admissible range.',
      ['Dataset','$k$','Retention','Chance','95\% interval','$p$'],
      [[NAMES[r.dataset],int(r.k),f(r.retention),f(r.expected_uniform_retention),f'[{f(r.ci_low)}, {f(r.ci_high)}]',f(r.p_perm_greater,4)] for _,r in ne.iterrows()])

# 02--04: identity and all control endpoints.
w=read('weight_retrieval.csv');w=w[w.checkpoint==20]
q=sup('02__per_query_ranks.csv');assert (q.correct_rank==1).all()
table('retrieval','Primary E20 weight retrieval. MRR is mean reciprocal rank. All 630 query/checkpoint comparisons also ranked the correct candidate first.',
      ['Dataset','Mean rank','Top-1 (\%)','MRR','Holm $p$'],
      [[NAMES[r.dataset],f(r.mean_rank,1),pct(r.top1,0),f(r.mrr,1),f(r.p_holm_less,4)] for _,r in w.iterrows()])
d=sup('03__all_control_endpoints.csv');assert len(d)==144
rows=[]
for keys,g in d.groupby(['metric','layer','alignment','gauge'],sort=False):
    metric,layer,align,gauge=keys;g=g.set_index('dataset').loc[DS]
    rows.append([METRIC[metric],LAYER[layer],ALIGN[align],GAUGE[gauge]]+
                [f'{r.mean_rank:.2f} / {100*r.top1:.0f}' for _,r in g.iterrows()])
table('controls','All 48 initialization-control endpoints. Entries are mean correct-candidate rank / top-1 percentage; chance values are 5.5 / 10. The 42 planned comparisons and six post hoc norm-rank comparisons are exploratory. Full intervals and unadjusted permutation probabilities accompany the source table.',
      ['Cost','Layer','Assignment','Gauge','MNIST','Fashion','KMNIST'],rows,'llllrrr')
changes=sup('03__data__historical_weight_changes_descriptive.csv');changes=changes[changes.checkpoint==20]
table('changes','Actual E0--E20 changes in native raw weights, averaged over ten networks per dataset. These historical changes are distinct from candidate-dependent retrieval residuals.',
      ['Dataset','Layer','Relative displacement','Exactly unchanged (\%)','Change $\leq10^{-6}$ (\%)'],
      [[NAMES[ds],LAYER[layer],f(g.relative_frobenius_change.mean()),pct(g.fraction_exactly_unchanged.mean(),3),pct(g.fraction_abs_change_le_1e_minus6.mean(),3)] for (ds,layer),g in changes.groupby(['dataset','layer'])], 'llrrr')
r=read('representation_retrieval.csv');r=r[r.checkpoint==20]
agg=sup('04__aggregate.csv');agg=agg[agg.checkpoint==20]
rows=[]
for _,x in r.iterrows():
    rows.append([NAMES[x.dataset],x.layer,x.gauge.replace('_',' '),f(x.mean_rank),f'[{f(x.mean_rank_ci_low)}, {f(x.mean_rank_ci_high)}]',pct(x.top1,1),f(x.p_perm_less,4)])
for _,x in agg.iterrows():
    rows.append(['Aggregate',x.layer,x.gauge.replace('_',' '),f(x.mean_rank),f'[{f(x.mean_rank_ci_low)}, {f(x.mean_rank_ci_high)}]',pct(x.top1,1),f(x.p_perm_less,4)])
table('representations','All E20 representation-retrieval conditions. Hidden raw CKA is primary; feature-standardized hidden CKA and logits are secondary. Mean-rank intervals and one-sided probabilities retain the source analysis conventions; displayed probabilities are unadjusted.',
      ['Dataset','Features','Scale','Rank','95\% interval','Top-1 (\%)','$p$'],rows,'lllrrrr')

# 05: performance and factor robustness.
d=sup('05__performance_decomposition.csv');d=d[d.checkpoint==20]
table('performance','Descriptive crossed-design performance decomposition at E20. Fractions partition the observed sum of squares, not unbiased random-effect variance components. One observation per cell prevents separation of interaction and residual error; no ANOVA tests are asserted.',
      ['Dataset','Measure','Mean','Init. fraction','Schedule fraction','Interaction/rest'],
      [[NAMES[r.dataset],r.metric,f(r.grand_mean,4),f(r.initialization_fraction),f(r.shuffle_fraction),f(r.interaction_residual_fraction)] for _,r in d.iterrows()], 'llrrrr')
d=sup('05__distance_contrasts.csv')
table('contrasts','Geometric contrasts: distance with both factors different minus distance with the named factor shared. Positive values indicate closer endpoints when that factor is shared. Factor-block intervals and unadjusted secondary permutation tests are retained.',
      ['Dataset','Space','Shared factor','Contrast','95\% interval','$p$'],
      [[NAMES[r.dataset],'Weights' if r.metric=='normalized_frobenius' else 'CKA',r.factor.replace('initialization','Init.').replace('shuffle','Schedule'),f(r.contrast,4),f'[{f(r.ci_low,4)}, {f(r.ci_high,4)}]',f(r.p_perm_greater,4)] for _,r in d.iterrows()], 'lllrrr')
d=sup('05__leave_factor_level_out.csv')
table('factor-deletion','Deletion of levels of the other factor. Ranges summarize the recorded leave-one-level estimates; they are sensitivity ranges, not confidence intervals. Initialization retrieval deletes a schedule; schedule retrieval deletes an initialization.',
      ['Dataset','Identify','Space','Rank range','Top-1 range (\%)'],
      [[NAMES[ds],'Init.' if analysis=='initialization_retrieval' else 'Schedule','Weights' if metric=='normalized_frobenius' else 'CKA',rng(g.mean_rank),rng(g.top1,1,100)] for (ds,analysis,metric),g in d.groupby(['dataset','analysis','metric'])], 'lllrr')

# 06: the entire expanded grid, pilot controls, alternative matching and regions.
d=sup('06__expanded_summary.csv');d=d[d.epsilon>0];assert len(d)==216
assert (d.ratio_low>1).all() and (d.n_seeds==10).all()
for ds in DS:
    rows=[]
    for (cp,layer,eps),g in d[d.dataset==ds].groupby(['checkpoint','layer','epsilon'],sort=True):
        g=g.set_index('momentum');a,b=g.loc['preserve'],g.loc['reset']
        rows.append([int(cp),LAYER[layer],pct(eps,0),ci(a,'ratio',2),ci(b,'ratio',2),
                     f'{pct(a.accuracy_mean,2)} / {pct(b.accuracy_mean,2)}',
                     f'{pct(a.fixed_region_member_mean,0)} / {pct(b.fixed_region_member_mean,0)}'])
    table('expanded-'+ds,f'Complete extension: {NAMES[ds]}. Each row represents two momentum conditions, each with ten sources and one direction per source. Ratios have pointwise 95\% bootstrap intervals. Accuracy and fixed combined-region membership are percentages, shown retained / reset. E2, E10, and E20 are one, five, and ten epochs before ten further epochs.',
          ['E','Layer','$\epsilon$ (\%)','$r$: retained [CI]','$r$: reset [CI]','Accuracy R/S','Region R/S'],rows,'llrrrrr')
e=sup('06__expanded_endpoints.csv');e=e[e.epsilon>0];assert len(e)==2160
assert e.initial_fixed_structural_member.all()
assert e.perturbation_induced_return.sum()==115
assert (e.barrier>0).sum()==3
rows=[]
for (ds,cp),g in e.groupby(['dataset','checkpoint']):
    cells=d[(d.dataset==ds)&(d.checkpoint==cp)]
    rows.append([NAMES[ds],int(cp),int(g.perturbation_induced_return.sum()),
                rng(cells.accuracy_vs_baseline_mean,2,100),rng(cells.loss_vs_baseline_mean,4),rng(cells.cka_mean,3),f(g.barrier.max(),5)])
table('expanded-function','Expanded functional and connectivity controls. Accuracy and loss ranges span the 24 cell means per dataset/time, each relative to its paired baseline (accuracy in percentage points). CKA ranges likewise span cell means. Returns count individual perturbed runs re-entering the combined region after intervention-induced exit; none is a structural exit--return. Maximum barriers are across the 240 perturbations per dataset/time.',
      ['Dataset','E','Returns','Accuracy $\Delta$ (pp)','Loss $\Delta$','CKA range','Max. barrier'],rows,'llrrrrr')
p=sup('06__endpoint_summary.csv');pilot=p[(p.phase=='pilot')&(p.epsilon>0)]
table('pilot','Pilot endpoints, five directions per source and ten sources per cell. Distances and functional values are source means; ratio intervals average directions before source bootstrap. Local membership combines proximity to the paired baseline and the reference performance criteria.',
      ['Momentum','$\epsilon$ (\%)','$r$ [95\% CI]','Accuracy (\%)','Loss','CKA','Local (\%)'],
      [[r.momentum,pct(r.epsilon,0),ci(r,'ratio',3),pct(r.accuracy_mean,3),f(r.loss_mean,5),f(r.cka_mean,3),pct(r.local_member_mean,0)] for _,r in pilot.iterrows()], 'lrrrrrr')
table('pilot-regions','Sensitivity of pilot endpoint membership to the structural quantile and reference ensemble. Entries are percentages [pointwise 95\% bootstrap interval]; reference halves each contain five networks.',
      ['Momentum','$\epsilon$ (\%)','q95','q90','Half A','Half B'],
      [[r.momentum,pct(r.epsilon,0)]+[ci(r,k,0,100) for k in ['region_member','q90_member','subset_a_member','subset_b_member']] for _,r in pilot.iterrows()], 'lrrrrr')
mo=sup('06__momentum_paired_comparison.csv')
table('momentum','Paired pilot momentum effects: retained minus reset, after averaging directions within source. Accuracy and membership differences are proportions; ratios and distances have their original units. Intervals are pointwise source bootstraps.',
      ['$\epsilon$ (\%)','Measure','Difference','95\% interval'],
      [[pct(r.epsilon,0),r.metric.replace('_',' '),f(r['mean'],5),f'[{f(r.low,5)}, {f(r.high,5)}]'] for _,r in mo.iterrows()], 'rlrr')
noise=p[p.order=='different']
table('batch-noise','Pilot controls changing the future batch sequence. With no injected perturbation, the normalized ratio is undefined because its denominator is zero; the absolute distance measures schedule-induced separation.',
      ['$\epsilon$ (\%)','Runs','Distance [95\% CI]','Accuracy [95\% CI], \%','Loss [95\% CI]'],
      [[pct(r.epsilon,0),int(r.n_runs),ci(r,'distance',3),ci(r,'accuracy',2,100),ci(r,'loss',4)] for _,r in noise.iterrows()])
a=sup('06__alignment_robustness_summary.csv');assert len(a)==222
rows=[]
for stage,g in a.groupby('stage'):
    for label,key in [('Git Re-Basin','ratio_git'),('PIF/Hungarian','ratio_pif'),('Alpha matching','ratio_alpha_matching')]:
        below=int((g[key+'_high']<1).sum());above=int((g[key+'_low']>1).sum())
        rows.append([stage.title(),label,len(g),below,above,len(g)-below-above])
table('matching','Matching sensitivity across all pilot and expanded cells. Counts classify pointwise 95\% ratio intervals relative to one. These are dependent measurement variants of the same runs, not independent replications.',
      ['Stage','Assignment','Cells','Below 1','Above 1','Includes 1'],rows,'llrrrr')

# 07: all fitted history probes, functional comparisons, symmetry validation.
h=read('history_probes.csv')
feature={'both':'Both weights','W0':r'$W_0$','W1':r'$W_1$',
         'diagnostics':'Diagnostics','functional_logits':'Logits','covariate_adjusted_both':'Residual weights'}
rows=[]
for _,r in h.iterrows():
    name=feature[r.features]+(' (shuffled)' if r.negative_control else '')
    rows.append([r.condition,int(r.c_epochs),name,pct(r.balanced_accuracy,2),f'[{pct(r.ba_ci_low,2)}, {pct(r.ba_ci_high,2)}]',f(r.permutation_p,3),f(r.secondary_holm_p,3)])
table('history','All history probes. Accuracy is balanced accuracy in percent; chance is 50\%. Intervals resample the 40 held-out seed pairs. The primary endpoint is reset/both/C10; Holm probabilities concern the specified secondary family. Dashes indicate that a probability was not defined for that row.',
      ['Momentum','C epochs','Features','Accuracy','95\% interval','$p$','Holm $p$'],rows,'lrlrrrr')
d=sup('07__performance_paired.csv');d=d[(d.split=='test')&(d.c_epochs==10)]
table('history-function','Final-task differences on held-out history seeds: B--A--C minus A--B--C. Accuracy differences and intervals are in percentage points; loss remains cross-entropy. Forty seed pairs per condition.',
      ['Momentum','Measure','Difference','95\% interval'],
      [[r.condition,r.metric,f(r.diff_H1_minus_H0*(100 if r.metric=='accuracy' else 1),4),f'[{f(r.ci_low*(100 if r.metric=="accuracy" else 1),4)}, {f(r.ci_high*(100 if r.metric=="accuracy" else 1),4)}]'] for _,r in d.iterrows()], 'llrr')

# 08: decoder/target controls, event endpoints, functional differences and block successes.
d=read('trajectory_regression.csv')
s=d[(d.cohort.isin(['one_bit','multi_bit']))&(d.target=='absolute')&(d.time_index==0)&(d.model.isin(['ridge','mlp']))]
assert len(s)==20
table('start-robustness','Initial-state reconstruction across all planned layer/PCA variants and the nonlinear probe. Targets are absolute earlier coordinates; the endpoint is never a target checkpoint. MSE and $R^2$ describe the same held-out seed blocks. No row provides accurate individualized start reconstruction.',
      ['Cohort','Layer','PCA dim.','Probe','$R^2$','MSE','MSE 95\% interval'],
      [[cohort_name(r.cohort),LAYER[r.layer],int(r.pca_dim),r.model.upper(),f(r.r2,3),f(r.mse,5),f'[{f(r.mse_ci_low,5)}, {f(r.mse_ci_high,5)}]'] for _,r in s.iterrows()], 'llrlrrr')
rows=[]
for cohort in ['one_bit','multi_bit']:
    meta=js('08__analysis__'+cohort+'__summary.json')
    for dim,value in meta['target_pca_explained_variance'].items():rows.append([cohort_name(cohort),dim,pct(value,2)])
table('pca','Fraction of scaled training-trajectory variance captured by each target PCA basis. The basis is learned from training trajectories only.',
      ['Cohort','Coordinates','Explained variance (\%)'],rows,'lrr')
s=d[(d.cohort.isin(['one_bit','multi_bit']))&(d.target=='random')]
table('random-target','Random-coordinate negative controls. Independent random targets are fitted and evaluated with the regression pipeline; the mean predictor uses training targets only. Individual random-control realizations need not equal zero $R^2$.',
      ['Cohort','Probe','$R^2$','95\% interval','MSE'],
      [[cohort_name(r.cohort),r.model.replace('random_target_','').title(),f(r.r2,4),f'[{f(r.r2_ci_low,4)}, {f(r.r2_ci_high,4)}]',f(r.mse,5)] for _,r in s.iterrows()], 'llrrr')
ev=read('event_probes.csv');assert len(ev)==40
table('events','All event probes, including layer features, performance/norm controls, nonlinear probes, and shuffled-label controls. Positions are numbered oldest to newest. Accuracy intervals are seed-block bootstraps. Holm correction applies to the prespecified linear feature-by-position family in each cohort; no adjusted probabilities were computed for MLP or shuffled-label rows.',
      ['Cohort','Position','Features','Probe','Accuracy (\%)','95\% interval','Holm $p$'],
      [[cohort_name(r.cohort),int(r.bit)+1,r.features.replace('both','Both'),{'ridge_classifier':'Ridge','label_shuffle':'Shuffled','mlp':'MLP'}[r.model],pct(r.accuracy,2),f'[{pct(r.accuracy_ci_low,2)}, {pct(r.accuracy_ci_high,2)}]',f(r.p_holm,3)] for _,r in ev.iterrows()], 'lrllrrr')
per=sup('08__final_performance_controls.csv');per=per[per.split.isin(['all','test'])]
table('event-function','Paired final-performance differences for event cohorts. All = 200 seed pairs; test = 40 held-out pairs. Bit 1 minus bit 0 refers to complementary order labels. Accuracy differences are percentage points; loss is cross-entropy. These are descriptive intervals, not equivalence tests or isolated causal effects of one bit.',
      ['Cohort','Split','Position','Measure','Difference','95\% interval'],
      [[cohort_name(r.cohort),r.split,int(r.bit_index)+1,r.metric.replace('final_',''),f(r.paired_difference_bit1_minus_bit0*(100 if r.metric=='final_accuracy' else 1),3),f'[{f(r.ci_low*(100 if r.metric=="final_accuracy" else 1),3)}, {f(r.ci_high*(100 if r.metric=="final_accuracy" else 1),3)}]'] for _,r in per.iterrows()], 'llrlrr')
pair=sup('08__provenance__multi_bit_feasibility_final_pair_success_intervals.csv')
table('pair-success','Whole-seed-pair success, a stricter outcome than per-network event accuracy. A success requires correct prediction for both histories of a seed; the all-bits row additionally requires all four positions. Exact binomial intervals condition on the fitted probe.',
      ['Cohort','Position','Successful pairs','Rate (\%)','Exact 95\% interval'],
      [[cohort_name(r.cohort),'All four' if r.bit=='joint_all_bits' else str(int(r.bit)+1),f'{int(r.fully_correct_pairs)}/40',pct(r.pair_success_rate,1),f'[{pct(r.pair_success_ci_low,2)}, {pct(r.pair_success_ci_high,2)}]'] for _,r in pair.iterrows()], 'llrrr')

manifest={'generator':'scripts/build_supplement.py','tables':tables,
          'sources':[{'snapshot':'data/'+n,'sha256':hashlib.sha256((DATA/n).read_bytes()).hexdigest()} for n in sorted(used)]}
(ROOT/'provenance/supplement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Generated {len(tables)} supplementary tables from {len(used)} archived files.')

"""Transcribe reported estimates; never use these values to generate results."""
from pathlib import Path
import json, hashlib
ROOT = Path(__file__).resolve().parents[1]
targets=[]
def add(key, expected, *, cohort='main', checkpoint=20, metric, group, view=None, alignment=None, ci=None, source='paper', section='', figure=None, table=None, page=None, notes='', **extra):
    targets.append(dict(key=key,cohort=cohort,checkpoint=checkpoint,metric=metric,group=group,view=view,alignment=alignment,expected=expected,ci_low=ci[0] if ci else None,ci_high=ci[1] if ci else None,source=source,source_section=section,figure=figure,table=table,page=page,notes=notes,**extra))
for dataset,accuracy,sd,loss in [('mnist',.9737,.0024,.1127),('fashion_mnist',.8637,.0047,.4046),('kmnist',.8599,.007,.7992)]:
 for metric,value in [('test_accuracy_mean',accuracy),('test_accuracy_sd',sd),('test_loss_mean',loss)]:
  add(f'main.{dataset}.{metric}',value,metric=metric,group=dataset,source='supplement',section='S1',table='S1',figure='S1',page=2)
for checkpoint,within,between,contrast in [(0,.2688,.2691,.0003),(1,.0765,.2361,.1596),(20,.0789,.2379,.1590)]:
 for group,value in [('within',within),('between',between),('contrast',contrast)]:
  add(f'main.canonical_alpha.balanced.E{checkpoint}.{group}',value,checkpoint=checkpoint,metric='canonical_alpha',group=group,view='balanced',alignment='canonical_alpha',ci=(.1553,.1627) if checkpoint==20 and group=='contrast' else None,section='3.2',figure='1/2',page=10)
for view,within,between,delta,ci in [('W0',.1511,.4568,.3057,(.2985,.3129)),('W1',.0067,.0190,.0123,(.0113,.0133))]:
 for group,value in [('within',within),('between',between),('contrast',delta)]:
  add(f'main.canonical_alpha.{view}.{group}',value,metric='canonical_alpha',group=group,view=view,alignment='canonical_alpha',ci=ci if group=='contrast' else None,section='3.3',figure='2',page=10)
raw_rows=[('euclidean','W0',.02785,.02748,.02823),('euclidean','W1',.09751,.09536,.09966),('euclidean','balanced',.06268,.06156,.06381),('euclidean','size_weighted',.02873,.02836,.02911),('snd','W0',.08864,.08747,.08981),('snd','W1',.17724,.17428,.18021),('snd','balanced',.13294,.13125,.13463),('snd','size_weighted',.08975,.08859,.09092),('alpha','W0',.29380,.29010,.29749),('alpha','W1',.01793,.01644,.01942),('alpha','balanced',.15586,.15369,.15803),('alpha','size_weighted',.29032,.28667,.29397)]
for metric,view,delta,lo,hi in raw_rows:
 add(f'main.raw.{metric}.{view}.contrast',delta,metric=metric,group='contrast',view=view,alignment='raw_alpha' if metric=='alpha' else metric,ci=(lo,hi),source='supplement',section='S2',figure='S2',table='S2',page=4)
for metric in ['euclidean','snd','alpha']:
 for view in ['W0','W1','balanced','size_weighted']:
  add(f'main.run_positive.{metric}.{view}',1.,metric='positive_run_contribution_fraction',group='all',view=view,alignment='raw_alpha' if metric=='alpha' else metric,source='supplement',section='S3',figure='S3',page=6,base_metric=metric,notes='Every one of 30 run-level between-minus-within contributions positive; descriptive.')
for task,acc in [('identity',.9725),('parity',.9860)]:
 add(f'task.{task}.accuracy',acc,cohort='task',metric='test_accuracy_mean',group=task,source='supplement',section='S4',figure='S4',page=7)
for metric,delta,lo,hi in [('euclidean',.01152,.01114,.01191),('snd',.02602,.02503,.02702),('alpha',.04452,.04110,.04793),('one_minus_cka',.13658,.12445,.14872)]:
 add(f'task.{metric}.contrast',delta,cohort='task',metric=metric,group='contrast',view='hidden' if metric=='one_minus_cka' else 'W0',alignment=None if metric=='one_minus_cka' else 'incoming_'+metric,ci=(lo,hi),source='supplement',section='S4',figure='S4',table='S3',page=7)
for metric,within,between in [('alpha',.0367,.0812),('one_minus_cka',.1969,.3335)]:
 for group,value in [('within',within),('between',between)]:
  add(f'task.{metric}.{group}',value,cohort='task',metric=metric,group=group,view='hidden' if metric=='one_minus_cka' else 'W0',alignment=None if metric=='one_minus_cka' else 'incoming_alpha',section='3.5',figure='4',page=11)
for group,value in [('cross_same_seed',.07857),('cross_different_seed',.08153)]:
 add(f'task.alpha.{group}',value,cohort='task',metric='alpha',group=group,view='W0',alignment='incoming_alpha',source='supplement',section='S4',figure='S4',page=7)
for metric,within,between,delta,ci in [('one_minus_cka',.2544,.5758,.3215,(.3123,.3306)),('alpha',.0641,.3579,.2938,(.2901,.2975))]:
 for group,value in [('within',within),('between',between),('contrast',delta)]:
  add(f'main.representation.{metric}.{group}',value,metric=metric,group=group,view='hidden' if metric=='one_minus_cka' else 'W0',alignment=None if metric=='one_minus_cka' else 'raw_alpha',ci=ci if group=='contrast' else None,section='3.6',figure='5',page=11,notes='Alpha measured on raw W0 but assignment uses both incoming and outgoing profiles.')
for group,r,rlo,rhi,rho,slo,shi in [('all',.9269,.9162,.9377,.6916,.6384,.7448),('within',.6524,.5268,.7780,.4587,.4150,.5024),('between',.1071,-.0823,.2965,.1091,-.0517,.2699)]:
 for metric,value,lo,hi in [('pearson',r,rlo,rhi),('spearman',rho,slo,shi)]:
  add(f'main.alpha_cka.{metric}.{group}',value,metric=metric,group=group,view='W0',alignment='raw_alpha',ci=(lo,hi),source='supplement',section='S5',figure='S5',table='S4',page=9,notes='Correlation between raw matched W0 Alpha and hidden 1-CKA; complete-run deletion.')
for C,change,corr,delta,retained in [(1,0.,1.,.15586,1.),(1.25,.00797,.994,.14784,.949),(1.5,.01940,.986,.13314,.854),(2,.03507,.968,.11379,.730),(4,.04483,.935,.10145,.651),(8,.04462,.911,.10084,.647)]:
 for metric,value in [('mean_absolute_alpha_change',change),('alpha_correlation_with_original',corr),('alpha_dataset_contrast',delta),('alpha_contrast_fraction_retained',retained)]:
  add(f'gauge.C{C}.{metric}',value,cohort='gauge',metric=metric,group='all' if 'contrast' not in metric else 'contrast',view='balanced',alignment='raw_alpha',source='supplement',section='S6',figure='S6',table='S5',page=11,envelope=C,notes='Envelope exponent permutation seed is not specified in the publication.')
add('gauge.max_logit_error',4.26e-14,cohort='gauge',metric='max_absolute_logit_error',group='all',source='supplement',section='S6',figure='S6',page=11,comparison='upper_bound_roundoff',notes='Machine-dependent roundoff diagnostic, not a scientific target to fit.')
add('gauge.max_canonical_weight_error',1.55e-15,cohort='gauge',metric='max_absolute_canonical_weight_error',group='all',source='supplement',section='S6',figure='S6',page=11,comparison='upper_bound_roundoff')
add('gauge.predictions_preserved',1.,cohort='gauge',metric='prediction_agreement_fraction',group='all',source='supplement',section='S6',figure='S6',page=11)
add('gauge.canonical_alpha.balanced.contrast',.15901,cohort='gauge',metric='canonical_alpha',group='contrast',view='balanced',alignment='canonical_alpha',ci=(.15535,.16266),source='supplement',section='S6',page=11)
add('gauge.C8.raw_alpha.contrast_ci',.1008,cohort='gauge',metric='alpha',group='contrast',view='balanced',alignment='raw_alpha',ci=(.0947,.1070),section='3.8',page=11,envelope=8)
for metric,step in [('one_minus_cka',2),('alpha',2),('euclidean',4),('snd',4),('canonical_alpha',4)]:
 add(f'early.onset.{metric}',step,cohort='early',checkpoint=None,metric='onset_update',group='contrast',view='hidden' if metric=='one_minus_cka' else 'balanced',source='supplement',section='S7',figure='S7',table='S6',page=13,base_metric=metric,notes='First post-initialization checkpoint with complete normal 95% run-jackknife CI above zero.')
for group,step in [('all',64),('mnist',32),('fashion_mnist',48),('kmnist',96)]:
 add(f'early.progress90.{group}',step,cohort='early',checkpoint=None,metric='accuracy_progress90_update',group=group,section='3.10',figure='6',page=13,notes='Progress is relative to gain over updates 0 to 470, not eventual training gain.')
for checkpoint,group,value in [(0,'all',.1043),(2,'all',.4146),(2,'mnist',.5105)]:
 add(f'early.accuracy.s{checkpoint}.{group}',value,cohort='early',checkpoint=checkpoint,metric='test_accuracy_mean',group=group,section='3.9',figure='6',page=13)
for view,relative,rms in [('W0',.0815,.00168),('W1',.0588,.00423)]:
 for metric,value in [('relative_frobenius_displacement',relative),('rms_displacement',rms)]:
  add(f'early.s2.{metric}.{view}',value,cohort='early',checkpoint=2,metric=metric,group='all',view=view,section='3.9',figure='6',page=13)
for metric,step,value,lo,hi in [('one_minus_cka',2,.0860,.0655,.1066),('canonical_alpha',4,.1322,.1072,.1571)]:
 add(f'early.onset_contrast.{metric}',value,cohort='early',checkpoint=step,metric=metric,group='contrast',view='hidden' if metric=='one_minus_cka' else 'balanced',ci=(lo,hi),section='3.9',figure='6',page=13)
for metric,unmatched,pif,git in [('weight_cosine',.0288,.1774,.2885),('weight_rmse',.1402,.1288,.1195),('canonical_alpha',.4780,.2324,.1496),('activation_correlation',.0502,.1314,.2734)]:
 for alignment,value in [('identity',unmatched),('benchmark',pif),('git_rebasin',git)]:
  add(f'benchmark.{metric}.{alignment}.all',value,cohort='benchmark',metric=metric,group='all',view='balanced' if metric=='canonical_alpha' else None,alignment=alignment,source='supplement',section='S8',figure='S8',table='S7',page=14,notes='Benchmark PIF label denotes candidate selection, not the direct canonical assignment used for main Figs 1-2.')
for alignment,within,between in [('benchmark',.0789,.3015),('git_rebasin',.0409,.1985)]:
 for group,value in [('within',within),('between',between)]:
  add(f'benchmark.canonical_alpha.{alignment}.{group}',value,cohort='benchmark',metric='canonical_alpha',group=group,view='balanced',alignment=alignment,section='Figure 3',figure='3',page=7)
for alignment,within,between in [('identity',.0655,.0433),('benchmark',.3078,.0520),('git_rebasin',.4928,.1747)]:
 for group,value in [('within',within),('between',between)]:
  add(f'benchmark.activation.{alignment}.{group}',value,cohort='benchmark',metric='activation_correlation',group=group,alignment=alignment,section='3.12',figure='3',page=14)
for metric,value,ci in [('canonical_alpha_pif_minus_git',.0828,(.0703,.0953)),('activation_pif_minus_identity',.0812,(.0780,.0844)),('activation_git_minus_pif',.1421,(.1380,.1461))]:
 add(f'benchmark.advantage.{metric}',value,cohort='benchmark',metric=metric,group='all',ci=ci,section='3.11/3.12',page=14)
add('benchmark.assignment_overlap',.165,cohort='benchmark',metric='assignment_overlap_fraction',group='all',ci=(.157,.173),source='supplement',section='S8',figure='S8',page=14)
for alignment in ['benchmark','git_rebasin']:
 add(f'benchmark.renumber_recovery.{alignment}',1.,cohort='benchmark',metric='renumber_assignment_recovery_fraction',group='all',alignment=alignment,source='supplement',section='S8',figure='S8',page=15,notes='30 predetermined E20 pairs x 10 random renumberings; pair IDs and random seed not given.')
add('benchmark.permutation_logit_error',1.42e-14,cohort='benchmark',metric='max_permutation_logit_error',group='all',source='supplement',section='S8',figure='S8',page=15,comparison='upper_bound_roundoff')
for comp,group,value in [('pif_gt_identity','within',1.),('pif_gt_identity','between',.697),('git_gt_identity','all',1.),('git_gt_pif','all',1.)]:
 add(f'benchmark.fraction_improved.{comp}.{group}',value,cohort='benchmark',metric='activation_improvement_fraction',group=group,source='supplement',section='S9',figure='S9',page=16,comparison_pair=comp)
for dataset,w01,w020,w11,w120 in [('mnist',.7925,.6719,.7986,.7205),('fashion_mnist',.7227,.6075,.7631,.6900),('kmnist',.6996,.5627,.7913,.7234)]:
 for view,checkpoint,value in [('W0',1,w01),('W0',20,w020),('W1',1,w11),('W1',20,w120)]:
  add(f'sign.{dataset}.{view}.E{checkpoint}',value,metric='sign_retention',group=dataset,view=view,checkpoint=checkpoint,source='supplement',section='S10',figure='S10',table='S8',page=17,notes='Compare raw weight signs to the same run at E0; descriptive run means, no matching.')
for metric,before,after in [('euclidean',.40174,.08844),('snd',.70260,.26841),('alpha',.44991,.00008)]:
 for stage,value in [('before',before),('after',after)]:
  add(f'synthetic.{metric}.{stage}',value,cohort='synthetic',checkpoint=None,metric=metric,group=stage,view='whole_network',alignment=metric,source='supplement',section='S11',table='S9',page=19,notes='20-12-10-8-6 copied/permuted random network plus noise SD=0.3*weight SD. Random law, seed and aggregation not specified; compare qualitative reduction only.')
# Standardized dimensions align directly with newly computed analysis tables.
normal_metrics = {'canonical_alpha','raw_alpha','euclidean','snd','one_minus_cka',
                  'activation_correlation','weight_cosine','weight_rmse','assignment_agreement'}
for t in targets:
    t['layer'] = t.pop('view')
    t['statistic'] = 'mean'
    t['mapping_status'] = 'notautomapped'
    old_metric=t['metric']
    if old_metric in ('test_accuracy_mean','test_accuracy_sd','test_loss_mean'):
        t['metric'] = 'loss' if old_metric=='test_loss_mean' else 'accuracy'
        t['statistic'] = 'std' if old_metric=='test_accuracy_sd' else 'mean'
        t['alignment']='none'; t['layer']='total'
        t['mapping_status']='training'
    else:
        if t['metric']=='alpha': t['metric']='raw_alpha'
        if t['cohort']=='benchmark':
            t['source_cohort']='benchmark'; t['cohort']='main'
        if t['alignment'] in ('canonical_alpha','raw_alpha','euclidean','snd') and t['cohort'] in ('main','early'):
            t['alignment']='profile'
        if t['alignment'] and t['alignment'].startswith('incoming_'):
            t['alignment']='incoming'
        if t['metric']=='one_minus_cka':t['alignment']='none';t['layer']='hidden'
        if t['metric']=='activation_correlation':t['layer']='hidden'
        if t['metric'] in ('weight_cosine','weight_rmse'):t['layer']='total'
        if t['metric']=='assignment_overlap_fraction':
            t['metric']='assignment_agreement';t['alignment']='benchmark_vs_git';t['layer']='hidden'
        if t['cohort'] in ('main','task','early') and t['metric'] in normal_metrics and t['group'] in ('within','between','contrast','all'):
            t['mapping_status']='estimates'
    # Special records require an explicit custom adapter and must never silently
    # match a different scientific endpoint just because a numeric value exists.
    if t['alignment'] is None:t['alignment']='none'
    if t['layer'] is None:t['layer']='total'
assert len({t['key'] for t in targets})==len(targets)
source_files=[]
for path,url in [('fcogn-5-1960468.pdf','https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2026.1960468/full'),('references/supplement.pdf','https://public-pages-files-2025.frontiersin.org/articles/1960468/file/Data_Sheet_1.pdf/1960468_data-sheet_1/2')]:
 source_files.append({'path':path,'url':url,'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()})
result={'schema_version':1,'doi':'10.3389/fcogn.2026.1960468','purpose':'Reported estimates for transparent comparison only. All reproduction results must be computed from new training runs; never tune seeds or inject reported estimates into measured outputs. Exact agreement is not required by the user.','source_files':source_files,'checkpoint_convention':{'main':'half-epoch index E (0..20)','task':'half-epoch index E (0..20)','early':'optimizer update s','gauge':'main E20','benchmark':'main E20 unless otherwise specified'},'targets':targets}
(ROOT/'configs/paper_targets.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('Wrote',len(targets),'targets')

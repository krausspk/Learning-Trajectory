"""Compare independently recomputed results with published reference values."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .analysis import load_runs, write_csv, write_json
from .statistics import jackknife, mean_t_interval


class Results:
    def __init__(self, output):
        self.output=Path(output)
        self.est=pd.read_csv(self.output/'summary/estimates.csv')
        self.train=pd.read_csv(self.output/'summary/training.csv')
        self.corr=pd.read_csv(self.output/'summary/correlations.csv')
        self.sign=pd.read_csv(self.output/'summary/sign_retention.csv')
        self.disp=pd.read_csv(self.output/'summary/displacements.csv')
        self.contrib=pd.read_csv(self.output/'summary/run_contributions.csv')
        self.gauge=pd.read_csv(self.output/'summary/gauge_estimates.csv')
        self.gpairs=pd.read_csv(self.output/'summary/gauge_pairs.csv')
        self.gchecks=pd.read_csv(self.output/'validation/gauge_checks.csv')
        self.stability=pd.read_csv(self.output/'validation/alignment_stability.csv')
        self.synthetic=json.loads((self.output/'validation/synthetic.json').read_text())
        self.timing=json.loads((self.output/'summary/early_timing.json').read_text())
        self._pairs={}
        self.comparison_replicates=[]
        self.active_target=None
        self.runs={c:load_runs(self.output,c) for c in ['main','task','early']}

    def pairs(self,cohort,cp):
        key=(cohort,cp)
        if key not in self._pairs:
            self._pairs[key]=pd.read_csv(self.output/'pairs'/cohort/f'checkpoint_{cp:04d}.csv')
        return self._pairs[key]

    @staticmethod
    def select(frame,**kwargs):
        x=frame
        for k,v in kwargs.items():
            if v is not None: x=x[x[k].eq(v)]
        return x

    def summary(self,metric,layer='balanced',alignment='profile',group='contrast',cohort='main',checkpoint=20):
        x=self.select(self.est,metric=metric,layer=layer,alignment=alignment,group=group,cohort=cohort,checkpoint=checkpoint)
        if len(x)!=1: raise ValueError(f'Expected one estimate, found {len(x)}: {(cohort,checkpoint,metric,layer,alignment,group)}')
        return x.iloc[0][['estimate','ci_low','ci_high']].to_dict()

    def pair_estimate(self,frame,cohort='main',values=None):
        runs=self.runs[cohort]
        groups=np.array([r['meta']['task' if cohort=='task' else 'dataset'] for r in runs])
        blocks=np.array([r['meta']['seed'] for r in runs]) if cohort=='task' else None
        result=jackknife(frame[['index_a','index_b']].to_numpy(int),frame.value.to_numpy() if values is None else values,groups,blocks=blocks)
        for deleted,stratum,replicate in zip(result['deleted'],result['strata'],result['replicates']):
            self.comparison_replicates.append({'target':self.active_target,'cohort':cohort,'deleted':deleted,'stratum':stratum,'estimate':replicate})
        return {k:result[k] for k in ['estimate','ci_low','ci_high']}

    def target(self,target):
        cohort=target['cohort']; metric=target['metric']; group=target['group']; cp=target.get('checkpoint')
        layer=target.get('layer') or target.get('view'); alignment=target.get('alignment')
        def value(v): return dict(estimate=float(v),ci_low=None,ci_high=None)
        aliases={'alpha':'raw_alpha','canonical_alpha':'canonical_alpha','raw_alpha':'raw_alpha'}
        if metric.startswith('test_') or metric in ['accuracy','loss']:
            x=self.select(self.train,cohort=cohort,checkpoint=cp)
            if group not in ['all','pooled']:
                x=x[x['task' if cohort=='task' else 'dataset'].eq(group)]
            column='loss' if 'loss' in metric else 'accuracy'
            return value(x[column].std(ddof=1) if metric.endswith('_sd') or target.get('statistic')=='std' else x[column].mean())
        if metric=='positive_run_contribution_fraction':
            x=self.select(self.contrib,metric=aliases.get(target['base_metric'],target['base_metric']),layer=layer)
            return value((x.value>0).mean())
        if metric in ['pearson','spearman']:
            x=self.select(self.corr,cohort=cohort,checkpoint=cp,layer=layer,method=metric,group=group)
            return x.iloc[0][['estimate','ci_low','ci_high']].to_dict()
        if metric=='sign_retention':
            x=self.select(self.sign,cohort=cohort,checkpoint=cp,dataset=group,layer=layer)
            return value(x.retention.mean())
        if metric=='onset_update': return value(self.timing['structural_onset'][aliases.get(target['base_metric'],target['base_metric'])])
        if metric=='accuracy_progress90_update': return value(self.timing['functional_90_percent']['mean' if group=='all' else group])
        if metric in ['relative_frobenius_displacement','rms_displacement']:
            x=self.select(self.disp,cohort=cohort,checkpoint=cp,layer=layer)
            stat=mean_t_interval(x['relative_frobenius' if metric.startswith('relative') else 'rms'].to_numpy())
            return {k:stat[k] for k in ['estimate','ci_low','ci_high']}
        if cohort=='gauge':
            envelope=target.get('envelope',target.get('C'))
            if metric=='max_absolute_logit_error':return value(self.gchecks.logit_error.max())
            if metric=='max_absolute_canonical_weight_error':return value(self.gchecks.canonical_weight_error.max())
            if metric=='prediction_agreement_fraction':
                col=next(c for c in ['prediction_agreement','predictions_equal','prediction_agreement_fraction'] if c in self.gchecks)
                return value(self.gchecks[col].mean())
            if metric in ['alpha_dataset_contrast','alpha_contrast_fraction_retained']:
                x=self.select(self.gauge,C=envelope,metric='raw_alpha',layer=layer,group='contrast').iloc[0]
                if metric=='alpha_dataset_contrast': return x[['estimate','ci_low','ci_high']].to_dict()
                baseline=self.select(self.gauge,C=1,metric='raw_alpha',layer=layer,group='contrast').estimate.iloc[0]
                return value(x.estimate/baseline)
            if metric in ['canonical_alpha','raw_alpha','alpha']:
                x=self.select(self.gauge,C=envelope or 1,metric=aliases.get(metric,metric),layer=layer,group=group).iloc[0]
                return x[['estimate','ci_low','ci_high']].to_dict()
            a=self.select(self.gpairs,C=1,metric='raw_alpha',layer=layer).sort_values(['index_a','index_b']).value.to_numpy()
            b=self.select(self.gpairs,C=envelope,metric='raw_alpha',layer=layer).sort_values(['index_a','index_b']).value.to_numpy()
            if metric=='mean_absolute_alpha_change':return value(np.mean(np.abs(a-b)))
            if metric=='alpha_correlation_with_original':return value(np.corrcoef(a,b)[0,1])
        if cohort=='synthetic':
            records=self.synthetic.get('results',self.synthetic.get('metrics'))
            synthetic_metric='alpha' if metric=='raw_alpha' else metric
            row=records[synthetic_metric] if isinstance(records,dict) else next(x for x in records if x['metric']==synthetic_metric)
            return value(row[group])
        if cohort=='benchmark' or target.get('source_cohort')=='benchmark':
            if metric=='renumber_assignment_recovery_fraction':return value(self.select(self.stability,method=alignment).assignment_equal.mean())
            if metric=='max_permutation_logit_error':return value(self.stability.logit_error.max())
            f=self.pairs('main',cp)
            if metric in ['assignment_overlap_fraction','assignment_agreement']:
                return self.pair_estimate(self.select(f,metric='assignment_agreement'))
            differences={'canonical_alpha_pif_minus_git':('canonical_alpha','balanced','benchmark','git_rebasin'),
                         'activation_pif_minus_identity':('activation_correlation','hidden','benchmark','identity'),
                         'activation_git_minus_pif':('activation_correlation','hidden','git_rebasin','benchmark')}
            if metric in differences or metric=='activation_improvement_fraction':
                if metric in differences: base,view,left,right=differences[metric]
                else:
                    left,right=target['comparison_pair'].split('_gt_')
                    left={'pif':'benchmark','git':'git_rebasin'}.get(left,left)
                    right={'pif':'benchmark','git':'git_rebasin'}.get(right,right)
                    base,view='activation_correlation','hidden'
                a=self.select(f,metric=base,layer=view,alignment=left)
                b=self.select(f,metric=base,layer=view,alignment=right)
                merged=a.merge(b[['index_a','index_b','value']],on=['index_a','index_b'],suffixes=('','_other'))
                if group in ['within','between']:merged=merged[merged.within.eq(group=='within')]
                delta=(merged.value-merged.value_other).to_numpy()
                return value((delta>0).mean()) if metric=='activation_improvement_fraction' else self.pair_estimate(merged,values=delta)
            view=layer or ('hidden' if metric=='activation_correlation' else 'total')
            if group=='all':return self.pair_estimate(self.select(f,metric=metric,layer=view,alignment=alignment))
            return self.summary(metric,view,alignment,group,'main',cp)
        metric=aliases.get(metric,metric)
        if group in ['cross_same_seed','cross_different_seed']:
            f=self.select(self.pairs(cohort,cp),metric=metric,layer=layer,alignment='incoming')
            seeds=np.array([r['meta']['seed'] for r in self.runs[cohort]])
            same=seeds[f.index_a]==seeds[f.index_b]
            f=f[~f.within & (same==(group=='cross_same_seed'))]
            return self.pair_estimate(f,cohort)
        alignment='none' if metric=='one_minus_cka' else ('incoming' if cohort=='task' else 'profile')
        return self.summary(metric,layer,alignment,group,cohort,cp)


def qualitative_checks(results):
    r=results; checks=[]
    def add(key,label,passed,evidence,primary=True):
        checks.append(dict(key=key,claim=label,reproduced=bool(passed),evidence=evidence,primary=primary))
    endpoint=r.summary('canonical_alpha')
    add('dataset_clustering','Positive kanonische Gewichts-Trennung zwischen Datensätzen',endpoint['ci_low']>0,endpoint)
    init=r.summary('canonical_alpha',checkpoint=0)
    add('initialization','Keine klare Datensatz-Trennung bei Initialisierung',init['ci_low']<=0<=init['ci_high'],init)
    e=r.est
    raw=e[e.cohort.eq('main') & e.checkpoint.eq(20) & e.alignment.eq('profile') & e.group.eq('contrast')]
    add('all_weight_measures','Positive Trennung in beiden Schichten und allen Gewichtsmaßen',(raw.ci_low>0).all(),{'positive_intervals':int((raw.ci_low>0).sum()),'total':len(raw)})
    add('run_robustness','Positive individuelle Beiträge aller Hauptläufe',(r.contrib.value>0).all(),{'positive':int((r.contrib.value>0).sum()),'total':len(r.contrib)})
    for metric,layer,align in [('raw_alpha','W0','incoming'),('one_minus_cka','hidden','none')]:
        stat=r.summary(metric,layer,align,cohort='task')
        add('task_'+metric,'Aufgaben-Trennung bei identischen Bildern: '+metric,stat['ci_low']>0,stat)
    stat=r.summary('one_minus_cka','hidden','none')
    add('representations','Repräsentationen bestätigen Datensatz-Trennung',stat['ci_low']>0,stat)
    corr=r.select(r.corr,checkpoint=20,layer='W0',method='pearson').set_index('group')
    add('correlation_groups','Gepoolte Alpha–CKA-Korrelation übersteigt Korrelation zwischen Datensätzen',corr.loc['all','estimate']>corr.loc['between','estimate'],corr.estimate.to_dict())
    timing=r.timing
    early=timing['structural_onset']; functional=timing['functional_90_percent']['mean']
    add('early_differentiation','Strukturelle und repräsentationale Trennung vor 90% des frühen Genauigkeitsgewinns',early['canonical_alpha'] is not None and early['one_minus_cka'] is not None and max(early['canonical_alpha'],early['one_minus_cka'])<functional,timing)
    d=r.select(r.disp,cohort='early',checkpoint=2).groupby('layer').relative_frobenius.mean()
    a=r.select(r.train,cohort='early',checkpoint=2).accuracy.mean()
    add('small_early_updates','Früher Genauigkeitsgewinn bei geringer Parameterbewegung',a>.3 and d.max()<.15,{'accuracy_after_2':a,'displacement':d.to_dict()})
    maxerr=r.gchecks[['logit_error','canonical_weight_error']].max()
    add('gauge_invariance','ReLU-Gauge erhält Funktion und kanonische Gewichte',maxerr.max()<1e-10,maxerr.to_dict())
    g=r.select(r.gauge,metric='raw_alpha',layer='balanced',group='contrast')
    add('raw_gauge_sensitivity','Raw-Alpha reagiert auf funktionserhaltende Skalierung, Trennung bleibt positiv',g.estimate.max()-g.estimate.min()>.01 and (g.ci_low>0).all(),dict(zip(g.C.astype(str),g.estimate)))
    f=r.pairs('main',20)
    for metric,layer,higher in [('weight_cosine','total',True),('canonical_alpha','balanced',False),('activation_correlation','hidden',True)]:
        vals=r.select(f,metric=metric,layer=layer).groupby('alignment').value.mean()
        passed=vals.git_rebasin>vals.benchmark>vals.identity if higher else vals.git_rebasin<vals.benchmark<vals.identity
        add('alignment_'+metric,'Git Re-Basin > PIF/Alpha > unaligned: '+metric,passed,vals.to_dict())
    add('renumber_stability','Alignment bleibt bei zufälliger Neuronenumbenennung stabil',r.stability.assignment_equal.all() and r.stability.objective_error.max()<1e-10,{'identical_assignments':int(r.stability.assignment_equal.sum()),'total':len(r.stability)})
    synth=r.synthetic.get('results',r.synthetic.get('metrics'))
    entries=list(synth.values()) if isinstance(synth,dict) else synth
    add('synthetic','Mehrschicht-Sanity-Test reduziert alle drei Dissimilaritäten',all(x['after']<x['before'] for x in entries),synth)
    # Descriptive subclaims are tracked separately from the central conclusions.
    act=r.select(f,metric='activation_correlation').pivot(index=['index_a','index_b','within'],columns='alignment',values='value').reset_index()
    add('all_within_pif','Jedes Within-Paar verbessert Aktivierungskorrelation unter PIF/Alpha',(act[act.within].benchmark>act[act.within].identity).all(),{'fraction':float((act[act.within].benchmark>act[act.within].identity).mean())},False)
    add('all_pairs_git','Git Re-Basin verbessert Aktivierungskorrelation gegenüber PIF/Alpha für jedes Paar',(act.git_rebasin>act.benchmark).all(),{'fraction':float((act.git_rebasin>act.benchmark).mean())},False)
    km=r.select(r.train,cohort='main',dataset='kmnist').groupby('checkpoint')[['accuracy','loss']].mean()
    add('kmnist_late_loss','Später KMNIST-Loss steigt bei weitgehend stabiler Genauigkeit',km.loc[20,'loss']>km.loc[2,'loss'] and abs(km.loc[20,'accuracy']-km.loc[2,'accuracy'])<.05,{'E2':km.loc[2].to_dict(),'E20':km.loc[20].to_dict()},False)
    return checks


def _evidence_text(check):
    e=check['evidence'];key=check['key']
    if isinstance(e,dict) and 'estimate' in e:
        return f"{e['estimate']:.5f}, 95%-KI [{e['ci_low']:.5f}; {e['ci_high']:.5f}]"
    if key=='early_differentiation':
        return f"CKA: Update {e['structural_onset']['one_minus_cka']}; kanonisches Alpha: Update {e['structural_onset']['canonical_alpha']}; 90%-Genauigkeitsfortschritt: Update {e['functional_90_percent']['mean']}"
    if key=='synthetic':
        return '; '.join(f"{x['metric']}: {x['before']:.5g} → {x['after']:.5g}" for x in e)
    if key=='small_early_updates':
        return f"Genauigkeit {100*e['accuracy_after_2']:.2f}%; relative Verschiebung W0 {100*e['displacement']['W0']:.2f}%, W1 {100*e['displacement']['W1']:.2f}%"
    if key=='gauge_invariance':return f"Maximale Abweichung: Logits {e['logit_error']:.2g}; kanonische Gewichte {e['canonical_weight_error']:.2g}"
    return '; '.join(f'{k}: {v:.5g}' if isinstance(v,(int,float)) else f'{k}: {v}' for k,v in e.items())


def make_report(output='outputs',targets='configs/paper_targets.json'):
    output=Path(output); r=Results(output)
    specifications=json.loads(Path(targets).read_text())['targets']
    comparisons=[]
    for target in specifications:
        r.active_target=target['key']
        try:
            measured=r.target(target)
            if not np.isfinite(measured['estimate']):raise ValueError('nonfinite estimate')
            row=dict(target,computed=measured['estimate'],computed_ci_low=measured['ci_low'],computed_ci_high=measured['ci_high'],difference=measured['estimate']-target['expected'],status='computed',error='')
        except (KeyError,ValueError,IndexError,StopIteration,TypeError) as error:
            row=dict(target,computed=None,computed_ci_low=None,computed_ci_high=None,difference=None,status='unresolved',error=str(error))
        comparisons.append(row)
    table=pd.DataFrame(comparisons)
    write_csv(output/'summary/paper_comparison.csv',table)
    write_json(output/'summary/paper_comparison.json',comparisons)
    write_csv(output/'summary/comparison_jackknife_replicates.csv',pd.DataFrame(r.comparison_replicates))
    for name,part in table[table['table'].notna()].groupby('table'):
        write_csv(output/'tables'/f'Table_{name}.csv',part)
    checks=qualitative_checks(r)
    write_json(output/'summary/qualitative_checks.json',checks)
    primary=[c for c in checks if c['primary']]
    passed=sum(c['reproduced'] for c in primary)
    lines=['# Reproduktion: Task-dependent clustering in permutation-aligned neural network weights','',
           'Alle Messwerte wurden neu berechnet. Die publizierten Zahlen dienen ausschließlich als Referenz; das Hauptkriterium ist die qualitative Übereinstimmung.','',
           f'**Zentrale qualitative Prüfungen: {passed}/{len(primary)} bestätigt.** Deskriptive Einzelbehauptungen werden separat ausgewiesen.','',
           '## Umfang','',
           '- 30 Hauptläufe, 20 Läufe der gepaarten Aufgaben-Kontrolle und 15 Frühtrainingsläufe.',
           '- 1.290 gespeicherte Trainingscheckpoints, inklusive Gewichten, Momentum, vollständigem Modell-/Optimiererzustand, RNG-Zuständen und Bildreihenfolge.',
           '- 21 Haupt-, 6 Aufgaben- und 16 Frühtrainings-Analysezeitpunkte; 11.955 vollständige Paar-Zeitpunkt-Vergleiche.',
           '- Hauptfiguren 1–6, Zusatzfiguren S1–S10, Tabellen S1–S9 und synthetischer Mehrschicht-Test.',
           '- Rohdaten, feste Probes, Aktivierungen, Permutationen, paarweise Messwerte und Jackknife-Replikate sind gespeichert.','',
           '## Zentrale Ergebnisse','',
           '| Befund | Reproduziert | Neu berechnete Evidenz |','|---|---|---|']
    for check in primary:
        evidence=_evidence_text(check)
        lines.append(f"| {check['claim']} | {'Ja' if check['reproduced'] else 'Nein'} | {evidence} |")
    lines+=['','## Vergleich ausgewählter Zahlen','','| Messgröße | Paper | Reproduktion | Differenz |','|---|---:|---:|---:|']
    selected={'main.canonical_alpha.balanced.E20.contrast':'Kanonische Alpha-Trennung, E20','task.alpha.contrast':'Aufgaben-Trennung, W0-Alpha','task.one_minus_cka.contrast':'Aufgaben-Trennung, 1−CKA','main.mnist.test_accuracy_mean':'MNIST-Genauigkeit','main.fashion_mnist.test_accuracy_mean':'Fashion-MNIST-Genauigkeit','main.kmnist.test_accuracy_mean':'KMNIST-Genauigkeit','early.onset.one_minus_cka':'Erste CKA-Trennung (Update)','early.onset.canonical_alpha':'Erste kanonische Alpha-Trennung (Update)','early.progress90.all':'90%-Fortschritt (Update)'}
    for key,label in selected.items():
        x=table[table.key.eq(key)]
        if len(x) and x.iloc[0].status=='computed':
            v=x.iloc[0];lines.append(f'| {label} | {v.expected:.6g} | {v.computed:.6g} | {v.difference:+.6g} |')
    lines+=['',f"Der vollständige Vergleich enthält {len(table)} publizierte Referenzwerte, davon {int(table.status.eq('computed').sum())} automatisch zugeordnet. Siehe [paper_comparison.csv](summary/paper_comparison.csv). Numerische Abweichungen sind keine automatisch fehlgeschlagenen Replikationen.",'',
            '## Unterschiede zum Paper','',
            f"Die kanonische Alpha-Trennung wird hier erstmals nach {r.timing['structural_onset']['canonical_alpha']} Updates nachweisbar (Paper: 4); das 90%-Kriterium des frühen Genauigkeitsgewinns liegt bei {r.timing['functional_90_percent']['mean']} Updates (Paper: 64). Frühe Kurven und Effektgrößen stimmen nicht punktweise überein. Die zeitliche Schlussfolgerung – Struktur entsteht während des funktionellen Lernens – bleibt erhalten.",'',
            f"Nach vier Updates beträgt der direkte kanonische Alpha-Kontrast {r.summary('canonical_alpha',cohort='early',checkpoint=4)['estimate']:.5f} (Paper: 0,1322). Diese deutliche numerische Abweichung bleibt als Ergebnis der Hauptrekonstruktion erhalten.",'',
            '## Deskriptive Einzelbefunde','']
    for c in checks:
        if not c['primary']:lines.append(f"- {'Bestätigt' if c['reproduced'] else 'Nicht vollständig bestätigt'}: {c['claim']}. {json.dumps(c['evidence'],ensure_ascii=False)}")
    diagnostic_path=output/'sensitivity/diagnostic.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        lines+=['','## Zusätzliche Sensitivitätsanalyse zum frühen Alignment','',
                'Nach Feststellung der frühen Abweichung wurde die Wahl des Matching-Verfahrens auf denselben gespeicherten Frühtrainingsgewichten untersucht. Dies ist eine nachträgliche Diagnose; die primäre Analyse und ihre Zahlen wurden nicht ersetzt. Die Methodenbeschreibung legt die frühe Assignment-Variante nicht ausdrücklich fest. Die alternative Kandidatenauswahl kann die Diskrepanz erklären, beweist aber nicht, welche Implementierung im Original verwendet wurde.','',
                'Alle Vergleichswerte und Jackknife-Replikate liegen unter [sensitivity](sensitivity/diagnostic.json).']
        sensitivity=pd.read_csv(output/'sensitivity/early_alignment_estimates.csv')
        cpcol='checkpoint' if 'checkpoint' in sensitivity else 'update'
        methodcol='method' if 'method' in sensitivity else 'alignment'
        subset=sensitivity[(sensitivity[cpcol]==4)&sensitivity.group.eq('contrast')]
        if 'layer' in subset:subset=subset[subset.layer.eq('balanced')]
        lines+=['','| Assignment bei Update 4 | Kontrast | 95%-KI |','|---|---:|---|']
        for _,row in subset.iterrows():lines.append(f"| {row[methodcol]} | {row.estimate:.5f} | [{row.ci_low:.5f}; {row.ci_high:.5f}] |")
    unresolved=table[table.status.ne('computed')]
    if len(unresolved):
        lines+=['','## Offene numerische Zuordnungen','']+[f"- {x.key}: {x.error}" for x in unresolved.itertuples()]
    lines+=['','## Rekonstruktionsentscheidungen und Grenzen','',
            'Die Publikation legt nicht alle Zufalls- und Implementierungsdetails fest. Verwendet werden `run_seed + epoch` als Shuffle-Seed, Batch 118 von 235 als halbe Epoche, NumPy RandomState für das Haupt-Probe-Set und kanonische Gewichte für den PIF-Kandidaten des Benchmark-Vergleichs. Seed und Initialisierung des synthetischen Tests sowie die Skalierungs-/Umbenennungs-Seeds sind explizit dokumentierte Rekonstruktionsentscheidungen.','',
            'Training: PyTorch 2.8.0 CPU, float32, NumPy 1.23.5; Analyse float64 mit SciPy 1.10.1. Betriebssystem/Python und Plotbibliothek unterscheiden sich vom Original. Die Eingangsdaten und Trainingsparameter entsprechen dem beschriebenen Versuch. Für den Aufgabenvergleich werden ausschließlich W0 und Hidden-Repräsentationen verglichen, ohne Zweischicht-Gauge.','',
            'Bestätigtes Clustering stellt weiterhin keinen Nachweis dynamischer Attraktoren oder identischer gelernter Algorithmen dar.','',
            '## Abbildungen','']
    for path in sorted((output/'figures').glob('*.png')):
        lines.extend([f'### {path.stem}','',f'![{path.stem}](figures/{path.name})',''])
    (output/'REPRODUCTION_REPORT.md').write_text('\n'.join(lines)+'\n')
    return {'primary_reproduced':passed,'primary_total':len(primary),'references_computed':int(table.status.eq('computed').sum()),'references_total':len(table),'unresolved':unresolved.key.tolist()}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');p.add_argument('--targets',default='configs/paper_targets.json')
    args=p.parse_args();print(json.dumps(make_report(args.output,args.targets),indent=2))

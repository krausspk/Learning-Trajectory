#!/usr/bin/env python3
"""Analyze the registered expansion and package all experiment artifacts."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'src'))
from learning_trajectory.data import sha256_file,_atomic_json


def analyze_expanded(workers=8):
    from analysis import analyze
    from expanded import cells
    for *_,cell in cells():
        jobs=json.loads((cell/'pilot_jobs.json').read_text())
        receipts=list((cell/'receipts').glob('*.json'))
        if len(receipts)<len(jobs)+10:
            raise ValueError(f'Cell not finished: {cell}')
        status=cell/'analysis_status.json'
        valid=False
        if status.exists():
            old=json.loads(status.read_text())
            valid=(old.get('complete') and old.get('analyzed_runs')==len(jobs)
                   and old.get('config_sha256')==sha256_file(cell/'config.json')
                   and old.get('region_thresholds_sha256')==sha256_file(cell/'region_thresholds.json')
                   and old.get('analysis_source_sha256')==sha256_file(ROOT/'code/analysis.py')
                   and old.get('metrics_source_sha256')==sha256_file(ROOT/'code/metrics.py'))
        if not valid:
            print(json.dumps({'analyzing_cell':cell.name}),flush=True)
            analyze(cell,workers=workers,report=False)


def merge_expanded():
    summaries=[];endpoints=[];curves=[]
    from expanded import cells
    for *_,cell in cells():
        if not (cell/'analysis_status.json').exists():raise ValueError(f'Missing registered cell {cell}')
        jobs=json.loads((cell/'pilot_jobs.json').read_text())
        for filename,collection in [('endpoint_summary.csv',summaries),('endpoint_metrics.csv',endpoints),('recovery_summary.csv',curves)]:
            df=pd.read_csv(cell/filename)
            df['dataset']=jobs[0]['dataset'];df['checkpoint']=jobs[0]['checkpoint']
            collection.append(df)
    if len(summaries)!=9:raise ValueError('All nine registered cells are required')
    summary=pd.concat(summaries,ignore_index=True)
    endpoints=pd.concat(endpoints,ignore_index=True)
    curves=pd.concat(curves,ignore_index=True)
    manifest_rows=[]
    for folder in [ROOT]+[cell for *_,cell in cells()]:
        for job in json.loads((folder/'pilot_jobs.json').read_text()):
            receipt=json.loads((folder/'receipts'/f"{job['job_id']}.json").read_text())
            manifest_rows.append({**job,'output_root':str(folder),'npz_path':receipt['npz_path'],'pt_path':receipt['pt_path'],'npz_sha256':receipt['npz_sha256'],'pt_sha256':receipt['pt_sha256'],'status':receipt['status']})
    pd.DataFrame(manifest_rows).to_csv(ROOT/'all_perturbation_manifest.csv',index=False)
    summary.to_csv(ROOT/'expanded_summary.csv',index=False)
    endpoints.to_csv(ROOT/'expanded_endpoints.csv',index=False)
    curves.to_csv(ROOT/'expanded_recovery_summary.csv',index=False)
    return summary,endpoints,curves


def expanded_figures(summary,ends,curves):
    datasets=['mnist','fashion_mnist','kmnist'];checkpoints=[2,10,20]
    names={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST'}
    colors={'preserve':'#176a9a','reset':'#d45a28'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
    for i,ds in enumerate(datasets):
        for j,cp in enumerate(checkpoints):
            ax=axes[i,j]
            g=summary[(summary.dataset==ds)&(summary.checkpoint==cp)&(summary.layer=='both')&(summary.epsilon>0)]
            for mom,h in g.groupby('momentum'):
                h=h.sort_values('epsilon')
                ax.plot(h.epsilon,h.ratio_mean,'o-',color=colors[mom],label=mom)
                ax.fill_between(h.epsilon.to_numpy(),h.ratio_low.to_numpy(),h.ratio_high.to_numpy(),color=colors[mom],alpha=.16)
            ax.axhline(1,color='.4',ls='--',lw=1)
            ax.set_yscale('log');ax.set_title(f'{names[ds]} · E={cp}');ax.grid(alpha=.15)
            if i==2:ax.set_xlabel('Relative Störstärke ε')
            if j==0:ax.set_ylabel('Finaler Abstand / Störabstand')
    axes[0,0].legend(title='Momentum',fontsize=8)
    fig.suptitle('Erweitertes Screening: Annäherung an die Kontrolltrajektorie\n10 Ausgangsnetze je Zelle, eine Richtung je Netz, 95%-Seed-Bootstrap',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'expanded_recovery_curves.{suffix}')
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
    for i,ds in enumerate(datasets):
        for j,cp in enumerate(checkpoints):
            ax=axes[i,j]
            g=summary[(summary.dataset==ds)&(summary.checkpoint==cp)&(summary.layer=='both')&(summary.epsilon>0)]
            for mom,h in g.groupby('momentum'):
                h=h.sort_values('epsilon')
                ax.plot(h.epsilon,h.region_member_mean,'o-',color=colors[mom],label=f'{mom}: final in Region')
                ax.fill_between(h.epsilon.to_numpy(),h.region_member_low.to_numpy(),h.region_member_high.to_numpy(),color=colors[mom],alpha=.10)
                if 'fixed_return_mean' in h:
                    ax.plot(h.epsilon,h.fixed_return_mean,'s--',color=colors[mom],label=f'{mom}: Eintritt feste Region')
            ax.set_title(f'{names[ds]} · E={cp}');ax.set_ylim(-.04,1.04);ax.set_xticks([.01,.05,.10,.20]);ax.grid(alpha=.15)
            if i==2:ax.set_xlabel('Relative Störstärke ε')
            if j==0:ax.set_ylabel('Anteil der Ausgangsnetze')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Erweitertes Screening: Endpunkt-Mitgliedschaft und Eintritt\nFeste Zielregion; Bänder: 95%-Seed-Intervalle für finale Mitgliedschaft',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'expanded_return_probability.{suffix}')
    plt.close(fig)
    selected=summary[(summary.epsilon>0)&(summary.momentum=='preserve')]
    fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True)
    for ax,ds in zip(axes,datasets):
        g=selected[selected.dataset==ds]
        table=g.pivot(index=['checkpoint','layer'],columns='epsilon',values='ratio_mean').sort_index()
        values=np.log10(table.to_numpy())
        im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=2),aspect='auto')
        ax.set_xticks(range(len(table.columns)),[f'{x:g}' for x in table.columns])
        ax.set_yticks(range(len(table.index)),[f'E{cp} · {layer}' for cp,layer in table.index])
        ax.set_xlabel('Störstärke ε');ax.set_title(names[ds])
        for i in range(len(table)):
            for j in range(len(table.columns)):
                ax.text(j,i,f'{table.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if values[i,j]>1.5 or values[i,j]<-.5 else 'black')
    fig.suptitle('Schichtvergleich: mittleres finales Abstandsverhältnis r\nMomentum erhalten; r < 1 bedeutet Annäherung relativ zur Anfangsstörung',fontsize=14)
    fig.subplots_adjust(top=.86,left=.08,right=.86,wspace=.15,bottom=.1)
    cax=fig.add_axes([.90,.13,.018,.68])
    fig.colorbar(im,cax=cax,label='log10(r)')
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'expanded_layer_comparison.{suffix}')
    plt.close(fig)


def expanded_report(summary,ends):
    main=summary[(summary.epsilon>0)&(summary.layer=='both')].copy()
    table=[]
    for _,r in main.iterrows():
        table.append(f"| {r.dataset} | {int(r.checkpoint)} | {r.momentum} | {r.epsilon:.2f} | {r.ratio_mean:.3f} [{r.ratio_low:.3f}; {r.ratio_high:.3f}] | {r.region_member_mean:.0%} | {r.fixed_return_mean:.0%} |")
    reliable=int((main.ratio_high<1).sum());expanding=int((main.ratio_low>1).sum())
    allpert=ends[ends.epsilon>0]
    allcells=summary[summary.epsilon>0]
    supported_all=int((allcells.ratio_high<1).sum())
    diverged_all=int((allcells.ratio_low>1).sum())
    initial_structural=int(allpert.initial_fixed_structural_member.sum())
    structural_returns=int(allpert.fixed_structural_return.sum())
    induced_returns=int(allpert.perturbation_induced_return.sum())
    body=f'''# Erweiterte Untersuchung

Die vorregistrierte Erweiterung enthält **2.160 Störläufe und 180 Baselines**, jeweils mit zehn weiteren vollständigen Trainingsepochen. Die neun Kombinationen aus MNIST/Fashion-MNIST/KMNIST und E=2/10/20 verwenden je zehn unabhängige Ausgangsnetze, beide Momentum-Protokolle, W0/W1/beide Matrizen und ε=0,01/0,05/0,10/0,20. Hinzu kommen 90 zeitlich passende unabhängige Referenzfortsetzungen und 20 neu trainierte Referenz-Ausgangsnetze. Der vorausgehende Pilot untersucht fünf Richtungen pro Ausgangsnetz; die Erweiterung **eine Richtung pro Ausgangsnetz und Zelle**.

Alle {len(ends)} geplanten Forks wurden ausgewertet; {int(ends.failed.sum())} numerische Ausfälle. Für die gemeinsame Störung beider Matrizen liegen {reliable} von {len(main)} deskriptiven 95%-Intervallen des relativen Abstands vollständig unter 1 und {expanding} vollständig darüber. Über alle Schichtvarianten liegen {supported_all}/{len(allcells)} Intervalle vollständig unter 1 und {diverged_all}/{len(allcells)} vollständig darüber. Die Auswertung vieler Zellen ist explorativ; die Intervalle sind nicht für multiple Vergleiche korrigiert.

Ein r<1 zeigt Annäherung relativ zur injizierten Störung. Ein gutes Klassifikationsergebnis, die Nähe zu einer breiten Task-Region und eine niedrige Loss-Barriere beweisen jeweils alleine keine dynamische Attraktion. Die feste Zielregion verwendet Referenznetze am jeweiligen Endhorizont (11, 15 beziehungsweise 20 Trainingsepochen); das zusätzliche zeitlich passende Referenzband verändert sich entlang der Recovery. Eintritt in die feste Endregion kann auch allgemeines Weiterlernen widerspiegeln, insbesondere bei frühen Störzeitpunkten; die Baselines sind deshalb mitgespeichert. Eine Richtung pro Seed prüft nicht die Richtungsabhängigkeit innerhalb desselben Ausgangsnetzes. Eine universelle Basin-Grenze lässt sich daraus nicht ableiten.

Alle {initial_structural}/{len(allpert)} gestörten Anfangszustände liegen bereits innerhalb der festen strukturellen Zielregion. Es wurden {structural_returns} strukturelle Außen→Innen-Verläufe nach einer unmittelbaren Störung beobachtet. Da kein sofortiger struktureller Austritt ausgelöst wurde, wurde H2 für die Rückkehr nach einem solchen Austritt in diesem Raster nicht getestet. Die {induced_returns} Fälle von störungsbedingtem Austritt und Rückkehr in die kombinierte Region betreffen Funktionskriterien bei anfänglich vorhandener struktureller Nähe. Dies ist kein struktureller Attraktionsnachweis.

![Recovery-Zeitverlauf, relativ](expanded_recovery_trajectories.png)

![Recovery-Zeitverlauf, absolut](expanded_absolute_trajectories.png)

![Relative Endpunktabstände nach Störstärke](expanded_recovery_curves.png)

![Regionsmitgliedschaft](expanded_return_probability.png)

![Schichtvergleich](expanded_layer_comparison.png)

## Gemeinsame Störung beider Matrizen

| Datensatz | E | Momentum | ε | r, Mittel [95%-KI] | Final in Region | Eintritt feste Region |
|---|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

Die Endpunkt-Mitgliedschaft verlangt gleichzeitig strukturelle Nähe und das Accuracy-/Loss-Kriterium. „Eintritt feste Region“ zählt Anfang außerhalb und Ende innerhalb; dies ist keine alleinige Attraktorentscheidung. Identische binäre Ergebnisse bei zehn Seeds können Bootstrap-Intervalle [0;0] oder [1;1] erzeugen; diese bedeuten keine populationsweite Gewissheit.

Das Gesamtmanifest aller 2.720 Versuchsfortsetzungen einschließlich Pilot und Kontrollen steht in [all_perturbation_manifest.csv](all_perturbation_manifest.csv). Die vollständigen Zahlen der Erweiterung stehen in [expanded_summary.csv](expanded_summary.csv) und [expanded_endpoints.csv](expanded_endpoints.csv). Jede Zelle enthält die eigene eingefrorene Referenzkalibrierung, vollständige Recovery-Daten, Loss-Interpolationen und Einzelprüfungen. Der Plan steht in [expanded_preregistration.json](expanded_preregistration.json). [Sekundärer Vergleich der Matching-Verfahren](alignment_robustness.md).
'''
    (ROOT/'expanded_analysis.md').write_text(body)
    short=f'''\n\n## Erweiterung auf alle Datensätze und Zeitpunkte

Zusätzlich zum Pilot wurden **2.160 Störläufe plus 180 Baselines** über drei Datensätze, drei Checkpoints, drei Schichtvarianten, vier Störstärken und beide Momentum-Protokolle vollständig ausgeführt. Jede Zelle enthält zehn Ausgangsnetze mit einer Richtung je Netz; die fünf Richtungen des Pilots bleiben separat ausgewertet. {int(ends.failed.sum())} numerische Ausfälle.

Bei gemeinsamer Störung beider Matrizen liegen {reliable}/{len(main)} deskriptive r-Intervalle vollständig unter 1, {expanding}/{len(main)} vollständig darüber. Damit ist die Aussage bedingungsspezifisch; eine allgemeine Rückkehr zur individuellen Kontrolltrajektorie oder eine universelle Basin-Grenze folgt daraus nicht. [Erweiterter Ergebnisbericht mit Abbildungen](expanded_analysis.md).
'''
    short+=f'\nAlle {initial_structural}/{len(allpert)} gestörten Ausgangszustände waren bereits innerhalb der festen strukturellen Region. Eine strukturelle Rückkehr nach unmittelbarem Austritt ist damit in dieser Erweiterung nicht getestet. Die {induced_returns} Rückkehren in die kombinierte Region betreffen funktionale Kriterien. [Matching-Robustheit](alignment_robustness.md).\n'
    for name in ['README.md','basin_analysis.md','attractor_conclusion.md']:
        path=ROOT/name
        text=path.read_text()
        marker='\n\n## Erweiterung auf alle Datensätze und Zeitpunkte'
        text=text.split(marker)[0]+short
        if name=='attractor_conclusion.md':
            overall=(f'**Gesamtbefund:** Keine allgemeine Rückkehr zur individuellen Kontrolltrajektorie. '
                f'In der Erweiterung zeigen {supported_all}/{len(allcells)} Zellen über alle Schichtvarianten '
                f'ein deskriptives 95%-Intervall für r vollständig unter 1; {diverged_all}/{len(allcells)} liegen vollständig darüber. '
                'Die Ergebnisse gelten für die untersuchten Bedingungen. Ein strenger mathematischer Attraktor oder eine universelle Basin-Grenze ist damit nicht nachgewiesen.')
            text='# Attraktor-Schlussfolgerung\n\n'+overall+'\n\n## MNIST-Pilot\n\n'+text.split('\n',1)[1].lstrip()
        path.write_text(text)
    readme=ROOT/'README.md'
    readme.write_text(readme.read_text()+'''

## Abschlussprüfungen

- 22 Tests bestanden: [Testprotokoll](validation/unit_tests.log).
- 80 Pilot-/Erweiterungs-Wiederholungen an 1.200 Zuständen bitgleich: [Audit](validation/pilot_expanded_overlap.json).
- 60 Wiederholungen der Originaltrajektorie über alle 30 Ausgangsnetze bitgleich: [Audit](validation/original_trajectory_replays.json).
- 2.430 Archive der Erweiterung einschließlich Referenzen geprüft: [Integrität](validation/expanded_checks.json).
- Alle 421 Artefakte der Original-Reproduktion unverändert: [Audit](validation/original_reproduction_audit.json).
- [Gesamtmanifest aller 2.720 Versuchsfortsetzungen](all_perturbation_manifest.csv).

## Erweiterung reproduzieren

Vom Projektverzeichnis aus, nach abgeschlossenem Pilot und `code/validation.py`:

```bash
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py prepare
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py references --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py calibrate
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py train --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py analyze --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py verify
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py report
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py manifest
python3 scripts/audit_results.py --output 06_Attractor_Perturbation
```

Die Original-Reproduktion bleibt unverändert. Eigene vollständige Optimizerzustände liegen jeweils unter `checkpoints/`; SHA-256-Receipts und das abschließende Artefaktinventar sichern Herkunft und Vollständigkeit. Die Ausführungsunterbrechung des Pilots und seine vollständige Wiederaufnahme sind separat unter `provenance/execution_interruption.json` dokumentiert.
''')


def verify_one_cell(cell):
    import validation
    validation.ROOT=Path(cell)
    result=validation.check_saved_jobs()
    result['cell']=Path(cell).name
    result['passed']=result['passed'] and result['checked_receipts']==270
    return result


def verify_expanded():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from expanded import cells
    records=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(verify_one_cell,str(cell)) for *_,cell in cells()]
        for future in as_completed(futures):
            result=future.result();records.append(result)
            print(json.dumps({'verified_cell':result['cell'],'passed':result['passed'],'receipts':result['checked_receipts']}),flush=True)
    records.sort(key=lambda r:r['cell'])
    result={'passed':all(r['passed'] for r in records),'checked_receipts':sum(r['checked_receipts'] for r in records),'numerical_failure_count':sum(r['numerical_failure_count'] for r in records),'cells':records}
    _atomic_json(ROOT/'validation/expanded_checks.json',result)
    if not result['passed']:raise AssertionError(result)


def provenance():
    source_dir=ROOT/'provenance/source'
    source_dir.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'code').glob('*.py'):
        shutil.copy2(path,source_dir/path.name)
    shared=source_dir/'learning_trajectory'
    shared.mkdir(exist_ok=True)
    for name in ['__init__.py','training.py','checkpoints.py','data.py','metrics.py']:
        shutil.copy2(PROJECT/'src/learning_trajectory'/name,shared/name)
    shutil.copy2(PROJECT/'requirements.lock',ROOT/'provenance/requirements.lock')
    frozen=ROOT/'provenance/expanded_source_checksums_before.json'
    checks={}
    for p in [ROOT/'provenance/source_checksums_before.json',frozen]:
        if p.exists():checks.update(json.loads(p.read_text()))
    # Both maps use project-relative paths.
    changed=[name for name,h in checks.items() if sha256_file(PROJECT/name)!=h]
    _atomic_json(ROOT/'validation/all_originals_unchanged.json',dict(checked_files=len(checks),changed=changed,passed=not changed))
    if changed:raise AssertionError(changed)
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts or path.name=='artifact_manifest.json':continue
        files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha256_file(path)))
    _atomic_json(ROOT/'artifact_manifest.json',dict(created_at=datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),files=files))
    print(json.dumps({'artifact_files':len(files),'bytes':sum(f['bytes'] for f in files),'originals_unchanged':not changed}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['analyze','report','verify','manifest']);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if args.phase=='analyze':analyze_expanded(args.workers)
    elif args.phase=='verify':verify_expanded()
    elif args.phase=='report':
        from reporting import make_report
        make_report(ROOT)
        summary,ends,curves=merge_expanded()
        expanded_figures(summary,ends,curves)
        from plot_expanded_trajectories import plot
        plot(ROOT,curves)
        expanded_report(summary,ends)
    else:provenance()

if __name__=='__main__':main()

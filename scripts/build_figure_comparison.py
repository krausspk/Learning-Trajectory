#!/usr/bin/env python3
"""Rebuild published figure excerpts and analogous plots from reproduced data."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

CAPTIONS={
 'figure1':('Datensatzabhängige Gewichtsstruktur','Versuchsaufbau und finaler Within-/Between-Vergleich. Direkte Hungarian-Zuordnung auf kanonischen Alpha-Profilen; beide Gewichtsschichten gleich gewichtet.'),
 'figure2':('Trainingsverlauf und Schichten','Kanonische Alpha-Verläufe, getrennte Schichtkontraste sowie euklidische und SND-Vergleiche mit jeweils eigener Zuordnung.'),
 'figure3':('Vergleich der Alignment-Verfahren','Unaligned, PIF/Alpha-Kandidatenauswahl und Git Re-Basin: Gewichtskosinus, kanonische Alpha-Dissimilarität und unabhängig gemessene Aktivierungskorrelation.'),
 'figure4':('Gleiche Bilder, verschiedene Aufgaben','Ziffernidentität versus Parität bei identischen Eingangsbildern, initialem W0 und Bildreihenfolge. Verglichen werden rohes W0-Alpha und Hidden-CKA; Unsicherheit über gepaarte Seed-Blöcke.'),
 'figure5':('Gewichte und Repräsentationen','Rohes W0-Alpha und Hidden-CKA: Gruppenkontraste, finale Werte sowie Paarstreuung und Pearson-Korrelation innerhalb und zwischen Datensätzen.'),
 'figure6':('Die ersten SGD-Schritte','Genauigkeit, relative Gewichtsverschiebung und frühe strukturelle Differenzierung. Markierungen werden aus unseren Läufen bestimmt. Die kanonische Alpha-Kurve weicht vom Original ab; diese Abweichung bleibt sichtbar.'),
 'figureS1':('Trainingsdiagnostik','Alle 30 Hauptläufe: Genauigkeit, Cross-Entropy-Loss und finale Verteilungen über unabhängige Läufe.'),
 'figureS2':('Alle Schichten und Gewichtsmaße','Euklidische Dissimilarität, SND und rohes Alpha, jeweils für W0, W1 sowie die schicht- und parameterzahlgewichtete Zusammenfassung.'),
 'figureS3':('Robustheit auf Run-Ebene','Individuelle Between-minus-Within-Beiträge aller Hauptläufe. Die dargestellten Punkte bleiben vollständige Trainingsläufe.'),
 'figureS4':('Aufgabenkontrolle und Seed-Paare','Genauigkeitsverläufe, ergänzende Gewichtsmaße und Vergleiche innerhalb von Aufgaben sowie zwischen Aufgaben bei gleichen oder verschiedenen Seeds.'),
 'figureS5':('Detaillierter Alpha–CKA-Vergleich','Schichtabhängige Korrelationsverläufe und getrennte Within-/Between-Streudiagramme. Gepoolte und bedingte Korrelationen werden unterschieden.'),
 'figureS6':('ReLU-Skalierung','Funktionserhaltende neuronweise Skalierung verändert rohes Alpha. Kanonische Gewichtsdarstellung und Netzfunktion werden separat auf numerische Invarianz geprüft.'),
 'figureS7':('Vollständige frühe Trainingsverläufe','Alle 16 gespeicherten Frühtrainingszeitpunkte bis Update 470, mit allen Gewichtsmaßen und Hidden-CKA. Die Zeitachse nennt die tatsächlichen SGD-Updates.'),
 'figureS8':('Matching-Verläufe und Validierung','Verlauf von Gewichtsübereinstimmung, Alpha und Aktivierungskorrelation sowie Assignment-Übereinstimmung und kontrollierte Neuronenumbenennung.'),
 'figureS9':('Aktivierungsbasierte Validierung','Verteilungen der Neuronenkorrelationen, Anteile verbesserter Netzwerkpaare und Verlauf der gepoolten Aktivierungskorrelation. Violinformen zeigen die neu gemessenen Verteilungen.'),
 'figureS10':('Vorzeichenstabilität','Vorzeichenretention relativ zur Initialisierung und finale Vorzeichenänderungen je Schicht und Datensatz; deskriptive Analyse ohne Neuronenmatching.'),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_originals(directory):
    manifest_path=directory/'original_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for entry in manifest['figures']:
        source=Path(entry['source_pdf'])
        if not source.exists():
            source=ROOT/'references'/'supplement.pdf' if 'supplement' in source.name else ROOT/source.name
        if sha(source)!=entry['source_sha256']:raise ValueError(f'Original PDF changed: {source}')
        x,y,w,h=entry['crop_pixels_top_left_xywh']
        cmd=['pdftoppm','-f',str(entry['page']),'-l',str(entry['page']),'-r',str(entry['dpi']),
             '-x',str(x),'-y',str(y),'-W',str(w),'-H',str(h),'-singlefile','-png',str(source),str(directory/Path(entry['file']).stem)]
        subprocess.run(cmd,check=True)
        entry.update(source_pdf=str(source.resolve()),render_command=cmd,sha256=sha(directory/entry['file']))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


def build_gallery(directory,output):
    originals=json.loads((directory/'original_manifest.json').read_text())
    reconstructed=json.loads((directory/'manifest.json').read_text())
    known={Path(row['png']).stem.removesuffix('_recon'):row for row in reconstructed['figures']}
    table=['# Original und Reproduktion','',
           'Alle Originale sind unveränderte PDF-Ausschnitte einschließlich Bildunterschrift. Sämtliche Rekonstruktionen verwenden ausschließlich unsere neu berechneten Daten.','',
           '[Vergleich nebeneinander öffnen](index.html)','',
           '| Abbildung | Inhalt | Original | Neue Daten | Vektorgrafik |','|---|---|---|---|---|']
    articles=[];navigation=[];records=[]
    for original in originals['figures']:
        name=original['id'];title,caption=CAPTIONS[name];recon=known[name]
        label='Fig. '+name.removeprefix('figure')
        source=Path(original['source_pdf']);kind='Supplement' if 'supplement' in source.name else 'Hauptpaper'
        for filename in [original['file'],recon['png'],recon['svg']]:
            if not (directory/filename).is_file():raise FileNotFoundError(filename)
        navigation.append(f'<a href="#{name}">{label}</a>')
        note=''
        if name in ['figure6','figureS7']:
            note='<p class="difference">Frühes kanonisches Alpha: Unsere direkte Zuordnung ergibt bei Update 4 einen kleineren Kontrast als im Paper. Die <a href="../outputs/REPRODUCTION_REPORT.md">Dokumentation</a> enthält die separate Sensitivitätsanalyse; die Hauptgrafik wurde nicht auf den Originalverlauf angepasst.</p>'
        articles.append(f'''<section id="{name}"><header class="figure-heading"><div><p class="eyebrow">{label} · {kind}</p><h2>{escape(title)}</h2></div><a class="vector" href="{recon['svg']}" target="_blank">Rekonstruktion als SVG ↗</a></header>
<p class="caption">{escape(caption)}</p>{note}
<div class="pair"><figure><figcaption><b>Original</b><span>{kind}, PDF-Seite {original['page']} · mit Originalcaption</span></figcaption><a href="{original['file']}" target="_blank"><img src="{original['file']}" loading="lazy" alt="{label}: Originalausschnitt aus {kind}"></a></figure>
<figure><figcaption><b>Rekonstruktion</b><span>Neu berechnete Daten · keine übernommenen Paper-Werte</span></figcaption><a href="{recon['png']}" target="_blank"><img src="{recon['png']}" loading="lazy" alt="{label}: analoge Auswertung unserer Reproduktion"></a></figure></div></section>''')
        table.append(f"| {label} | {title} | [{name}_orig]({original['file']}) | [{name}_recon]({recon['png']}) | [SVG]({recon['svg']}) |")
        records.append({'id':name,'title':title,'description':caption,'source_page':original['page'],'source_pdf':str(source),
                        'files':{filename:sha(directory/filename) for filename in [original['file'],recon['png'],recon['svg']]}})
    html='''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper und Reproduktion – Figurenvergleich</title>
<style> *{box-sizing:border-box} html{scroll-behavior:smooth;scroll-padding-top:85px} body{margin:0;background:#f2f4f7;color:#16283d;font:16px/1.5 system-ui,sans-serif} a{color:#245f9a} main{max-width:1800px;margin:auto;padding:36px 24px 60px} .intro{max-width:1050px;margin-bottom:28px} h1{font-size:clamp(28px,3vw,44px);line-height:1.15;margin:8px 0 20px} h2{font-size:23px;margin:0} .eyebrow{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#60758c;margin:0 0 6px} .intro p{color:#42566d}.notice{background:#e7eff8;padding:14px 18px;border-left:4px solid #4477aa} nav{position:sticky;top:0;z-index:10;background:#16283d;display:flex;flex-wrap:wrap;justify-content:center;gap:6px;padding:11px 16px} nav a{color:white;text-decoration:none;padding:4px 11px;border-radius:4px;background:#28415d;font-size:13px}nav a:hover{background:#4477aa} section{background:white;border:1px solid #dce3eb;border-radius:10px;margin-bottom:26px;padding:24px;box-shadow:0 3px 12px #16283d08}.figure-heading{display:flex;justify-content:space-between;align-items:center;gap:20px}.vector{font-size:13px;white-space:nowrap}.caption{max-width:1250px;color:#485c72;margin:14px 0 22px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}figure{margin:0;min-width:0;border:1px solid #dce3eb;background:white}figcaption{display:flex;flex-direction:column;padding:12px 16px;background:#f6f8fa;border-bottom:1px solid #dce3eb}figcaption b{font-size:16px}figcaption span{font-size:12px;color:#62758b}figure img{display:block;width:100%;height:auto;padding:6px} .difference{font-size:14px;padding:12px 16px;border-left:3px solid #b18430;background:#fbf7ed;color:#70551f}footer{font-size:13px;color:#60758c;margin-top:30px} @media(max-width:900px){.pair{grid-template-columns:1fr}main{padding:22px 12px}.figure-heading{display:block}.vector{display:inline-block;margin-top:8px}section{padding:16px}nav{position:static}} @media print{nav{display:none}body{background:white}main{max-width:none;padding:0}section{break-inside:avoid;box-shadow:none}h1{font-size:26px}.pair{grid-template-columns:1fr 1fr}}</style></head><body>
<nav aria-label="Figuren">NAVIGATION</nav><main><div class="intro"><p class="eyebrow">Learning Trajectory · 16 direkte Vergleiche</p><h1>Originalpaper und neu berechnete Ergebnisse</h1><p>Links stehen die Abbildungen aus dem Originalpaper und Supplement mit ihren Bildunterschriften. Rechts stehen dieselben wissenschaftlichen Auswertungen auf Basis unserer unabhängig neu trainierten Netzwerke.</p><p class="notice">Alle Kurven, Balken, Punkte, Fehlerintervalle und Zahlen rechts stammen aus unseren neuen Daten. Abweichungen bleiben sichtbar. Ein Klick auf eine Abbildung öffnet die volle Auflösung.</p><p>Gewichts- und Repräsentationsvergleiche verwenden Run-Jackknife beziehungsweise gepaarte Seed-Blöcke; frühe Genauigkeits- und Verschiebungsbänder verwenden Student-t-Intervalle. Die konkreten Vergleichsvarianten sind jeweils erläutert.</p></div>ARTICLES
<footer>Originale: Krauss et al. (2026), <i>Task-dependent clustering in permutation-aligned neural network weights</i>, Frontiers in Cognition 5:1960468. <a href="https://doi.org/10.3389/fcogn.2026.1960468">DOI: 10.3389/fcogn.2026.1960468</a>. Artikel unter CC BY. Originale direkt aus den vorhandenen PDFs gerendert; Quellen, Ausschnitte und Hashes in <a href="original_manifest.json">original_manifest.json</a>. Rekonstruktionsdaten und Hashes in <a href="manifest.json">manifest.json</a>. <a href="comparison_manifest.json">Vergleichsinventar</a>.</footer></main></body></html>'''
    numerical_links='<p><a href="../outputs/summary/paper_comparison.csv">Alle 170 Zahlenwerte: Original vs. Reproduktion (CSV)</a> · <a href="../outputs/REPRODUCTION_REPORT.md">Vollständiger Ergebnisbericht</a></p>'
    (directory/'index.html').write_text(html.replace('NAVIGATION',''.join(navigation)).replace('ARTICLES',numerical_links+'\n'+'\n'.join(articles)))
    table+=['','Originalquelle: Krauss et al. (2026), DOI [10.3389/fcogn.2026.1960468](https://doi.org/10.3389/fcogn.2026.1960468), CC BY.','',
            'Die sechs Hauptfiguren und zehn Supplementfiguren werden vollständig abgedeckt. Supplement S11 enthält nur Tabelle S9.','',
            'Neu erzeugen: `.venv/bin/python scripts/build_figure_comparison.py`. Nur die Übersicht aktualisieren: denselben Befehl mit `--gallery-only` ausführen.','',
            'Original-PDF-Ausschnitte enthalten die Originalcaptions. Die Rekonstruktionen sind analoge Auswertungen der neuen Daten; sie übernehmen keine Messwerte aus den Originalgrafiken.']
    (directory/'README.md').write_text('\n'.join(table)+'\n')
    comparison={'count':len(records),'data_directory':str(output.resolve()),'figures':records,'gallery':'index.html',
                'original_manifest_sha256':sha(directory/'original_manifest.json'),'reconstruction_manifest_sha256':sha(directory/'manifest.json')}
    (directory/'comparison_manifest.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+'\n')
    print(f'Created {len(records)} original/reconstruction pairs: {directory/"index.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'outputs'))
    parser.add_argument('--figures-dir',default=str(ROOT/'figures'))
    parser.add_argument('--gallery-only',action='store_true')
    args=parser.parse_args();directory=Path(args.figures_dir).resolve();output=Path(args.output).resolve()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    if not args.gallery_only:
        from learning_trajectory.figures import make_figures
        render_originals(directory)
        make_figures(output,directory,dpi=240,reconstruction_names=True)
    build_gallery(directory,output)


if __name__=='__main__':main()

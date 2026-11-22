"""Assemble German experiment reports with all original vector result figures.

Run with the project environment (reportlab + pypdf). Content is explicitly
specified in each experiment's results_content.json; no result is invented.
Existing result figures are embedded at native size to preserve their labels.
"""
from __future__ import annotations
import argparse,hashlib,io,json,html
from pathlib import Path
from datetime import datetime,timezone

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether
from pypdf import PdfReader,PdfWriter,PageObject,Transformation

ROOT=Path(__file__).resolve().parents[1]
FONT=Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DV',str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-Bold',str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
INK=colors.HexColor('#20344a'); BLUE=colors.HexColor('#315d85'); MUTED=colors.HexColor('#526375')
STYLES={
 'title':ParagraphStyle('title',fontName='DV-Bold',fontSize=23,leading=29,textColor=INK,spaceAfter=12),
 'subtitle':ParagraphStyle('subtitle',fontName='DV',fontSize=11,leading=16,textColor=MUTED,spaceAfter=15),
 'heading':ParagraphStyle('heading',fontName='DV-Bold',fontSize=12,leading=16,textColor=BLUE,spaceBefore=13,spaceAfter=7,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='DV',fontSize=9.6,leading=14,textColor=INK,spaceAfter=8),
 'callout':ParagraphStyle('callout',fontName='DV-Bold',fontSize=11,leading=16,textColor=INK),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=8.1,leading=11,textColor=INK),
 'headercell':ParagraphStyle('headercell',fontName='DV-Bold',fontSize=8.1,leading=11,textColor=INK),
 'small':ParagraphStyle('small',fontName='DV',fontSize=8,leading=11,textColor=MUTED,spaceAfter=6),
 'caption':ParagraphStyle('caption',fontName='DV',fontSize=10,leading=14,textColor=INK),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(text):
    return str(text).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')


def para(text,style='body'):
    return Paragraph(html.escape(clean(text)).replace('\n','<br/>'),STYLES[style])


def add_table(story,spec,width):
    headers=spec['headers'];data=[headers]+spec['rows']
    cells=[[para(value,'headercell' if i==0 else 'cell') for value in row] for i,row in enumerate(data)]
    fractions=spec.get('widths',[1/len(headers)]*len(headers))
    weights=sum(fractions)
    table=Table(cells,colWidths=[width*x/weights for x in fractions],repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#b9c9d6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#ced8e1'))]))
    story.extend([table,Spacer(1,10)])
    if spec.get('note'):story.append(para(spec['note'],'small'))


def make_intro(model):
    stream=io.BytesIO();w,h=A4;margin=44
    doc=SimpleDocTemplate(stream,pagesize=A4,rightMargin=margin,leftMargin=margin,topMargin=47,bottomMargin=43,title=model['title'],author='Learning Trajectory - experiment analysis')
    story=[para(model.get('experiment','Learning Trajectory'),'small'),para(model['title'],'title'),para(model['subtitle'],'subtitle')]
    box=Table([[para(model['finding'],'callout')]],colWidths=[w-2*margin])
    box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eaf2f8')),('BOX',(0,0),(-1,-1),.6,colors.HexColor('#bdcfdf')),('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12)]))
    story.extend([box,Spacer(1,12)])
    for section in model['sections']:
        story.append(para(section['title'],'heading'))
        for text in section.get('paragraphs',[]):story.append(para(text))
        for spec in section.get('tables',[]):add_table(story,spec,w-2*margin)
    story.append(para('Abbildungen und Nachvollziehbarkeit','heading'))
    for i,fig in enumerate(model['figures'],1):story.append(para(f"Abbildung {i}: {fig['title']}",'small'))
    story.append(para('Alle nachfolgenden Abbildungen werden als Vektorseiten in ihrer ursprünglichen Größe eingebettet. Dadurch bleiben auch umfangreiche Kontrollübersichten lesbar. Die zusätzliche deutsche Erläuterung beschreibt die jeweils gezeigte Auswertung.','small'))
    story.append(para(model.get('provenance','Rohdaten, statistische Tabellen, Seeds, Prüfsummen und Reproduktionscode liegen im selben Experimentordner. Ergebnisse werden ausschließlich aus den gespeicherten eigenen Läufen berechnet.'),'small'))
    def decorate(c,d):
        c.setFont('DV',8);c.setFillColor(MUTED)
        c.drawString(margin,h-25,clean(model.get('experiment','Learning Trajectory')))
        c.drawRightString(w-margin,h-25,'Ergebnisbericht')
        c.setStrokeColor(colors.HexColor('#d4dee7'));c.line(margin,h-32,w-margin,h-32)
        c.drawString(margin,22,'Learning Trajectory | Eigene Reproduktion und Folgeanalysen')
        c.drawRightString(w-margin,22,str(d.page))
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    stream.seek(0);return stream


def figure_page(source,model,figure,index,page_number):
    reader=PdfReader(source)
    if len(reader.pages)!=1:raise ValueError(f'Expected a single figure page: {source}')
    src=reader.pages[0];w,h=float(src.mediabox.width),float(src.mediabox.height)
    caption=para(figure['caption'],'caption');_,capheight=caption.wrap(w-40,10000)
    bottom=capheight+44;top=47;totalh=h+bottom+top
    page=PageObject.create_blank_page(width=w,height=totalh)
    page.merge_transformed_page(src,Transformation().translate(0,bottom))
    stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=(w,totalh))
    c.setFont('DV-Bold',11);c.setFillColor(INK)
    c.drawString(20,totalh-23,clean(f"Abbildung {index}: {figure['title']}"))
    c.setFont('DV',8);c.setFillColor(MUTED)
    c.drawString(20,totalh-38,clean(model.get('experiment','Learning Trajectory')))
    caption.drawOn(c,20,31)
    c.setFont('DV',8);c.setFillColor(MUTED)
    c.drawRightString(w-20,12,str(page_number))
    c.drawString(20,12,'Eigene Daten | Vollständige Definitionen und Tabellen im Experimentordner')
    c.save();stream.seek(0);page.merge_page(PdfReader(stream).pages[0]);return page


def build(out):
    out=Path(out).resolve();content=out/'results_content.json';model=json.loads(content.read_text())
    writer=PdfWriter();intro=make_intro(model);intro_reader=PdfReader(intro)
    for page in intro_reader.pages:writer.add_page(page)
    writer.add_outline_item('Ergebnisse, Methoden und Interpretation',0)
    figure_records=[]
    for i,fig in enumerate(model['figures'],1):
        source=out/fig['pdf'];pageindex=len(writer.pages)
        writer.add_page(figure_page(source,model,fig,i,pageindex+1))
        writer.add_outline_item(f"Abbildung {i}: {clean(fig['title'])}",pageindex)
        figure_records.append({'source':fig['pdf'],'sha256':sha(source),'report_page':pageindex+1})
    writer.add_metadata({'/Title':model['title'],'/Author':'Learning Trajectory - experiment analysis','/Subject':model['subtitle'],'/Creator':'ReportLab and pypdf; source-linked scientific result report'})
    target=out/'results.pdf'
    with target.open('wb') as stream:writer.write(stream)
    actual=PdfReader(target)
    assert len(actual.pages)==len(intro_reader.pages)+len(model['figures'])
    assert all((page.extract_text() or '').strip() for page in actual.pages)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'report':'results.pdf','report_sha256':sha(target),'pages':len(actual.pages),'intro_pages':len(intro_reader.pages),'figure_count':len(model['figures']),'content_sha256':sha(content),'generator':str(Path(__file__).resolve()),'generator_sha256':sha(__file__),'figure_sources':figure_records,'data_sources':{p:sha(out/p) for p in model.get('source_files',[])},'visual_qa':'pending'}
    (out/'results_manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'directory':out.name,'pages':len(actual.pages),'figures':len(model['figures'])}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directories',nargs='+',type=Path)
    for path in parser.parse_args().directories:build(path)

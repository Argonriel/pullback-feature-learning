"""Render the final English manuscript to a reviewable PDF.

Requires reportlab. The experiments themselves use the project NumPy runtime.
"""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

ROOT=Path(__file__).resolve().parent
FONTROOT=Path('C:/Windows/Fonts')
for alias,filename in [('Body','calibri.ttf'),('BodyBold','calibrib.ttf'),('BodyItalic','calibrii.ttf'),('Display','georgia.ttf')]:
    pdfmetrics.registerFont(TTFont(alias,str(FONTROOT/filename)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic')
INK=colors.HexColor('#16333C');TEAL=colors.HexColor('#0F6C73');GRAY=colors.HexColor('#52656B')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('Text',fontName='Body',fontSize=10.5,leading=14,textColor=INK,spaceAfter=7,allowWidows=0,allowOrphans=0))
styles.add(ParagraphStyle('SmallText',parent=styles['Text'],fontSize=8.8,leading=11.5,spaceAfter=4))
styles.add(ParagraphStyle('TitleX',fontName='Display',fontSize=24,leading=29,textColor=INK,spaceAfter=15))
styles.add(ParagraphStyle('HeadingX',fontName='BodyBold',fontSize=14,leading=18,textColor=TEAL,spaceBefore=13,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle('SubheadingX',fontName='BodyBold',fontSize=11.5,leading=15,textColor=INK,spaceBefore=8,spaceAfter=6,keepWithNext=True))

def inline(s):
    s=escape(s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<link href="\2" color="#0F6C73"><u>\1</u></link>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'\1',s)
    return s

def page(c,doc):
    c.saveState()
    c.setStrokeColor(colors.HexColor('#D7E1E2'));c.setLineWidth(.6)
    c.line(44,748,568,748);c.line(44,38,568,38)
    c.setFillColor(TEAL);c.setFont('BodyBold',8)
    c.drawString(44,760,'PULLBACK FEATURE LEARNING | INDEPENDENT RESEARCH EXTENSION')
    c.setFillColor(GRAY);c.setFont('Body',8)
    c.drawString(44,25,'Same training progress, prospective predictions, and symmetry-aware inference')
    c.drawRightString(568,25,str(doc.page))
    c.restoreState()

def build():
    lines=(ROOT/'FINAL_MANUSCRIPT.md').read_text(encoding='utf8').splitlines();story=[];i=0
    while i<len(lines):
        s=lines[i].strip();i+=1
        if not s:continue
        if s.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',s)
            if match:
                p=ROOT/match.group(2);w,h=ImageReader(str(p)).getSize();width=510; height=width*h/w
                story.extend([Spacer(1,5),Image(str(p),width=width,height=height),Paragraph(inline(match.group(1)),styles['SmallText']),Spacer(1,6)])
            continue
        if s.startswith('|'):
            raw=[s]
            while i<len(lines) and lines[i].strip().startswith('|'):raw.append(lines[i].strip());i+=1
            parsed=[[c.strip() for c in row.strip('|').split('|')] for row in raw if not re.fullmatch(r'[|:\-\s]+',row)]
            data=[[Paragraph(inline(c),styles['SmallText']) for c in row] for row in parsed]
            cols=len(data[0]);widths=[510/cols]*cols
            if cols>=5:widths=[110]+[400/(cols-1)]*(cols-1)
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF3F3')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.8,TEAL),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#CADBDD'))]))
            story.append(KeepTogether([t,Spacer(1,8)]));continue
        if s.startswith('# '):style='TitleX';s=s[2:]
        elif s.startswith('## '):style='HeadingX';s=s[3:]
        elif s.startswith('### '):style='SubheadingX';s=s[4:]
        else:
            style='Text'
            if s.startswith('- '):s='• '+s[2:]
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ')):
                s+=' '+lines[i].strip();i+=1
        story.append(Paragraph(inline(s),styles[style]))
    doc=SimpleDocTemplate(str(ROOT/'final_insert.pdf'),pagesize=letter,leftMargin=44,rightMargin=44,topMargin=60,bottomMargin=52,
        title='Predicting loss-matched updates and selecting a second view',author='277 final research')
    doc.build(story,onFirstPage=page,onLaterPages=page)

if __name__=='__main__':build()

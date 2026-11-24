"""Build the two-page course-project insert from the audited seed summaries."""
import csv
import json
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = ROOT / 'final_insert.pdf'
W,H = letter
L,R = 46,566
INK, MUTED, TEAL = '#182D36','#4B606A','#08675D'
PALE, GRID, GRAY = '#EDF6F3','#DAE5E2','#7F9498'


def register_fonts():
    sources = [('Text','calibri.ttf'),('Bold','calibrib.ttf'),('Title','georgia.ttf'),
               ('Italic','calibrii.ttf')]
    for name, file in sources:
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
    pdfmetrics.registerFontFamily('Text', normal='Text', bold='Bold', italic='Italic')


def color(value):
    return colors.HexColor(value)


def txt(c,x,t,s,font='Text',size=11,ink=INK,align='left'):
    c.setFillColor(color(ink))
    c.setFont(font,size)
    draw = {'left': c.drawString, 'right': c.drawRightString, 'center':c.drawCentredString}[align]
    draw(x,H-t,s)


def block(c,x,t,w,s,size=11.5,leading=15.5,ink=INK):
    style = ParagraphStyle('block',fontName='Text',fontSize=size,leading=leading,
                           alignment=TA_LEFT,textColor=color(ink),spaceBefore=0,spaceAfter=0)
    p = Paragraph(s,style)
    _,height = p.wrap(w,700)
    p.drawOn(c,x,H-t-height)
    return t+height


def rule(c,x1,x2,t,ink=GRID,width=.7):
    c.setStrokeColor(color(ink)); c.setLineWidth(width)
    c.line(x1,H-t,x2,H-t)


def frame(c,page):
    txt(c,L,34,'SYMMETRY AND TEST-TIME COMPUTATION','Bold',9,TEAL)
    txt(c,R,34,'RESEARCH NOTE','Text',9,MUTED,'right')
    rule(c,L,R,44)
    rule(c,L,R,752)
    txt(c,L,770,'Conditional half-turn averaging on noisy concentric rings','Text',9,MUTED)
    txt(c,R,770,f'{page} / 2','Bold',9,MUTED,'right')


def page1(c,summary):
    frame(c,1)
    pb = summary['models']['pb']
    arms = pb['arms']
    share = arms['folded']['fraction_of_full_gain']
    d = pb['folded_advantage']['folded_vs_random']
    txt(c,L,91,'Which inputs deserve a second view?','Title',26)
    end=block(c,L,109,R-L,'A noisy ring keeps its class after a half-turn. A trained network can still give the two orientations different predictions. With one extra prediction available for only a quarter of inputs, can we identify which ones benefit most?',13.2,17.5)
    assert end<=166
    c.setFillColor(color(PALE)); c.roundRect(L,H-256,R-L,78,8,fill=1,stroke=0)
    txt(c,L+15,227,f'{share["estimate"]:.1%}','Title',34,TEAL)
    txt(c,L+150,209,'of the full two-view gain recovered','Bold',15)
    end=block(c,L+150,218,348,f'Pullback at 500 steps. The 95% seed-bootstrap interval was {share["ci95"][0]:.1%} to {share["ci95"][1]:.1%}.',11,14,MUTED)
    assert end<=253
    txt(c,L,286,'How much of the full improvement did each rule capture?','Bold',15)
    txt(c,L,306,'Pullback models, fixed second-view budget of 64 per 256 inputs','Text',10.5,MUTED)
    x,bar_w = 210,282
    for t,method,label in [(334,'folded','Radius + folded angle'),(362,'radius','Radius only'),
                           (390,'boundary','Nearest Bayes boundary'),(418,'random','Uniform random')]:
        v = arms[method]['fraction_of_full_gain']['estimate']
        txt(c,L,t,label,'Bold' if method=='folded' else 'Text',11,TEAL if method=='folded' else INK)
        c.setFillColor(color('#EDF1F1')); c.rect(x,H-t+1,bar_w,7,stroke=0,fill=1)
        c.setFillColor(color(TEAL if method=='folded' else '#87A4A1')); c.rect(x,H-t+1,bar_w*v,7,stroke=0,fill=1)
        txt(c,R,t,f'{v:.1%}','Bold',11,TEAL if method=='folded' else INK,'right')
    end=block(c,L,435,R-L,'The denominator is the mean loss reduction when every input receives a half-turn. Shares use a ratio of mean gains over the 16 trained models.',10.2,13,MUTED)
    assert end<=465
    rule(c,L,R,478)
    txt(c,L,508,'The effect depended on how the network was trained','Bold',15)
    cols = [L,278,365,463,R]
    for col,label in zip(cols,['Training schedule','One view','Targeted','All inputs','Gain kept']):
        txt(c,col,532,label,'Bold',10,MUTED,'left' if col==L else 'right')
    rule(c,L,R,539)
    for t,m,name in [(556,'gd','Gradient descent'),(578,'pb','Pullback'),(600,'hybrid','Pullback then gradient descent')]:
        a = summary['models'][m]['arms']
        txt(c,L,t,name,'Text',10.8)
        for i,key in enumerate(('raw','folded','full')):
            txt(c,cols[i+1],t,f'{a[key]["loss"]["mean"]:.4f}','Text',10.8,INK,'right')
        txt(c,R,t,f'{a["folded"]["fraction_of_full_gain"]["estimate"]:.1%}','Bold',10.8,TEAL,'right')
    rule(c,L,R,609)
    end=block(c,L,620,R-L,'Entries are mean population log-loss estimates in nats per example. “Targeted” uses radius plus folded angle. Every model trained for 500 steps. The hybrid switched after 100 pullback steps.',10.4,13.6,MUTED)
    assert end<=657
    lo,hi = d['ci95']
    end=block(c,L,656,R-L,f'<b>Primary comparison.</b> Targeted allocation beat random allocation on all {d["positive_pairs"]} pullback models, saving {d["mean"]:.5f} nats per input (95% paired interval {lo:.5f} to {hi:.5f}).',11.2,14)
    assert end<=686
    end=block(c,L,692,R-L,'<b>A stronger alternative.</b> On a separate test draw, selecting the least confident first predictions beat the coordinate rule on all 16 pullback models (loss 0.09860 versus 0.10113, paired difference 0.00253 [0.00198, 0.00317]). Confidence also won on all 16 GD models. The coordinate rule won on 12 of 16 hybrid models.',11.2,14)
    assert end<=749
    c.showPage()


def page2(c,summary):
    frame(c,2)
    txt(c,L,89,'Why the rule can learn without labels','Title',23.5)
    end=block(c,L,108,R-L,'Let <b>a</b> and <b>b</b> be the logits of the original and half-turned inputs. Binary logistic loss equals B(a) − ya, where B(a) = log(1 + exp(a)). Averaging the loss over both orientations and subtracting the loss of their mean logit cancels the label term. The remaining gain is',11.9,15.5)
    assert end<=158
    c.setFillColor(color(PALE)); c.roundRect(L,H-211,R-L,37,6,fill=1,stroke=0)
    txt(c,(L+R)/2,198,'d(a, b) = [B(a) + B(b)] / 2 − B((a + b) / 2) ≥ 0','Text',15.2,TEAL,'center')
    end=block(c,L,222,R-L,'Convexity gives the inequality. The two orientations have the same class law on these rings. Within any cell unchanged by the half-turn, the mean of d equals the expected loss saved by taking the second view. The optimal cell-based rule selects inputs with the greatest expected gains. First-view confidence can change with orientation and lies outside this result.',11.6,15)
    assert end<=288
    rule(c,L,R,298)
    txt(c,L,325,'What was fitted and tested','Bold',15)
    end=block(c,L,336,R-L,'For each model, we used 8,192 unlabeled calibration inputs to estimate the mean gain in each of 16 radius bins crossed with eight folded-angle bins. Folding the angle modulo π puts opposite orientations in the same cell. At test time, coordinates give the cell score. Among 256 inputs, the top 64 receive the second view and average the two logits.',11.4,15)
    assert end<=402
    end=block(c,L,408,R-L,'The evaluation used 16 independent training and initialization pairs, three training schedules per pair, and 32,768 fresh inputs per pair. Policies shared the same examples. The evaluator used the generator’s conditional class probabilities and averaged the two possible base orientations. Intervals resample whole training pairs.',11.4,15)
    assert end<=471
    rule(c,L,R,481)
    txt(c,L,507,'Fewer predictions and higher measured latency','Bold',15)
    with (RESULTS/'timing.csv').open(newline='') as f:
        times = [r for r in csv.DictReader(f) if r['model']=='pb']
    median = lambda k: statistics.median(float(r['ms_per_256']) for r in times if r['policy']==k)
    be = summary['models']['pb']['amortization']['break_even_future_examples_for_mean_gains']
    end=block(c,L,519,R-L,f'The targeted rule uses 320 network example-forwards per batch of 256, versus 512 for two views everywhere. But on the tested CPU, its median latency was <b>{median("folded"):.3f} ms</b> per batch, versus <b>{median("full"):.3f} ms</b> for two views everywhere. Calibration also used 16,384 example-forwards per model. By forward count, its estimated break-even against equal-cost random allocation was about {be:,.0f} future inputs.',11.5,15)
    assert end<=604
    rule(c,L,R,615)
    txt(c,L,641,'Where this fits in prior work','Bold',15)
    end=block(c,L,652,R-L,'<link href="https://arxiv.org/abs/2005.00178" color="#08675D"><u>Lyle et al. (2020)</u></link> studied group averaging under convex loss. <link href="https://www.jmlr.org/papers/volume24/23-0041/23-0041.pdf" color="#08675D"><u>Wood et al. (2023)</u></link> derived exact log-loss decompositions for ensembles. <link href="https://proceedings.neurips.cc/paper/2020/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html" color="#08675D"><u>Kim et al. (2020)</u></link> learned to rank test-time transformations, and <link href="https://doi.org/10.1109/VLSI-SoC53125.2021.9606980" color="#08675D"><u>Mocerino et al. (2021)</u></link> adapted the number of views per input. Here we turn the half-turn loss identity into an unlabeled rule for spending a fixed batch budget and test it against a known population distribution.',11.2,14.6)
    assert end<=740
    c.showPage()


if __name__ == '__main__':
    register_fonts()
    s = json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))
    c = canvas.Canvas(str(OUT),pagesize=letter)
    c.setTitle('Which inputs deserve a second view?')
    c.setSubject('Budgeted half-turn averaging on noisy concentric rings')
    page1(c,s)
    page2(c,s)
    c.save()

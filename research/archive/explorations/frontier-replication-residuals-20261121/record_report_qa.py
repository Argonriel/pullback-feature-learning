"""Record final PDF checks after completed human-style visual inspection."""
from pathlib import Path
import datetime,hashlib,json
from pypdf import PdfReader
from PIL import Image
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reader=PdfReader(ROOT/'research_report.pdf');assert len(reader.pages)==9
texts=[p.extract_text() for p in reader.pages];assert all(len(s)>1200 for s in texts)
assert 'verification comparisons' in texts[4]
assert '0<=epsilon<=1' in texts[5]
assert 'in the recovery check' in texts[7]
assert 'no confidence interval' in texts[1].lower()
links=[]
for page in reader.pages:
    for ref in page.get('/Annots',[]):
        obj=ref.get_object();action=obj.get('/A')
        if action and '/URI' in action:links.append(str(action['/URI']))
assert len(links)>=5
pages=[]
for n in range(1,10):
    p=ROOT/'pdf_preview'/f'page-{n}.png'
    with Image.open(p) as im:width,height=im.size
    assert height==1500 and width>1000
    pages.append(dict(page=n,preview_sha256=sha(p),width=width,height=height,text_characters=len(texts[n-1]),visual_status='PASS'))
reviews={}
for name in ('ALLOCATION_REPORT_REVIEW.md','IMAGE_REPORT_REVIEW.md','THEORY_REPORT_REVIEW.md'):
    path=ROOT/'root_audit'/name;assert path.exists();reviews[name]=sha(path)
out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_pages=9,
    pdf_sha256=sha(ROOT/'research_report.pdf'),manuscript_sha256=sha(ROOT/'REPORT.md'),pages=pages,external_hyperlinks=links,
    visual_review='Root inspected all nine initial pages and final affected theory/timing pages; independent branch reviewers checked final pages2-9. No clipping, overlap, missing glyphs or illegible figures remain.',
    resolved=['explicit epsilon range and IID same-P sampling','un-clipped counterexample equality','roundoff maximum distinguished from initial failure','complete timing exclusions and hardware/criterion','baseline identity inference and source/category confounding'],
    independent_review_hashes=reviews)
(ROOT/'REPORT_QA.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(dict(status='PASS',pages=9,pdf_sha256=out['pdf_sha256'],links=len(links))))

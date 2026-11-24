"""Record completed visual/semantic review; this does not perform the review."""
from pathlib import Path
import datetime,hashlib,json
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pdf=ROOT/'research_report.pdf';reader=PdfReader(pdf);assert len(reader.pages)==9
texts=[p.extract_text() for p in reader.pages]
assert all(len(t)>1200 for t in texts)
for needle,page in [('0.553049',1),('validation - flop_1280',2),('9.813%',3),('43.511%',4),('1.197',5),('[0,.047928]',6),('(1+.6 cos(3phi))/(2 pi)',7),('19,890,176',8)]:assert needle in texts[page],needle
out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pages=9,pdf_sha256=sha(pdf),source_sha256=sha(ROOT/'REPORT.md'),
    render_command='pdftoppm -scale-to 1500 -png research_report.pdf pdf_preview/page',
    pages_visually_inspected=list(range(1,10)),final_modified_pages_reinspected=[3,5,7,8],
    independent_review='Structured-branch researcher inspected all nine pages against source tables; root applied comparator, normalized-density and model-dependence clarifications, then re-rendered and reviewed changed pages.',
    checks=['No clipped text, overlaps, missing glyphs or unreadable plots','Nine planned pages remain nine pages','Tables use source results and correct contrast directions','Negative quality, serving and shift results retained','Known/estimated-family guarantees explicitly conditional','No claim of equal elapsed time from nominal arithmetic equality','Five live external source links'],
    page_render_sha256={str(i):sha(ROOT/'pdf_preview'/f'page-{i}.png') for i in range(1,10)},
    character_counts=[len(t) for t in texts],limitations='Visual/semantic human-style review and extraction checks, not a formal verification of scientific claims.')
(ROOT/'REPORT_QA.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k not in ['page_render_sha256','checks']},indent=2))

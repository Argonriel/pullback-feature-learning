"""Record completed visual and independent report reviews for this final artifact."""
from pathlib import Path
import datetime
import hashlib
import json
from pypdf import PdfReader

root = Path(__file__).resolve().parent
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reader = PdfReader(root / 'research_report.pdf')
assert len(reader.pages) == 6
assert all(len(page.extract_text()) > 1500 for page in reader.pages)
for relative in ['root_audit/TRAINING_RESULTS.json', 'root_audit/ROBUSTNESS_RESULTS.json',
                 'root_audit/REALDATA_RESULTS.json', 'root_audit/REALDATA_TRANSFER_CHECK.json',
                 'SOURCE_INTEGRITY.json']:
    assert json.loads((root / relative).read_text(encoding='utf8'))['status'] == 'PASS', relative
result = dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='PASS',
    report='research_report.pdf', pdf_sha256=digest(root / 'research_report.pdf'),
    markdown_sha256=digest(root / 'REPORT.md'),
    pages=6, visually_inspected_pages=[1, 2, 3, 4, 5, 6],
    rendering='pdftoppm PNG, 1400-pixel long side; all final pages inspected directly',
    visual_findings='Readable text, figures and tables; no clipped content, overlapping elements or unintended overflow pages.',
    independent_report_reviews={
        'loss_matched': 'Training/calibration claims, primary intervals, costs and real-image replay accurate.',
        'fourier_views': 'Required coordinatewise mask invariance added after review; corrected final theorem wording confirmed accurate.',
        'independent_audit': 'Real-image metrics, timing, targets and cost counts accurate; optional 15,000 diagnostic forwards disclosure added.'
    },
    final_clarifications=[
        'Artificial paired-law loss improvement is distinguished from accuracy; paired ridge error increases from 20.36% to 22.84%.',
        'All five frozen scientific source sets and three internal release hash commitments pass verification.',
        'No experiments or fitted parameters changed during report review.'
    ]
)
(root / 'REPORT_QA.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(dict(status='PASS', pages=6, pdf_sha256=result['pdf_sha256'])))

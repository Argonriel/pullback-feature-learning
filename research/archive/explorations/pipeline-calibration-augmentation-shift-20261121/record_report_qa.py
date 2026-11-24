"""Record the completed direct visual inspection and independent report review."""
from pathlib import Path
import datetime,hashlib,json
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reader=PdfReader(ROOT/'research_report.pdf')
assert len(reader.pages)==6 and all(len(p.extract_text())>1500 for p in reader.pages)
for relative in ['root_audit/PREFLIGHT.json','root_audit/CALIBRATION_RESULTS.json',
                 'root_audit/AUGMENTATION_RESULTS.json','root_audit/SHIFT_RESULTS.json',
                 'root_audit/CNN_RANGE_BOUNDS.json','SOURCE_INTEGRITY.json']:
    assert json.loads((ROOT/relative).read_text(encoding='utf8'))['status']=='PASS',relative
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',pages=6,
    pdf_sha256=sha(ROOT/'research_report.pdf'),markdown_sha256=sha(ROOT/'REPORT.md'),
    visual_review=dict(pages=[1,2,3,4,5,6],render='pdftoppm PNG, 1400-pixel long side',
        findings='All pages directly inspected: readable figures/tables, consistent layout, no clipping, overlap, or overflow pages.',
        final_revision='Page2 selection-field chronology corrected and visually rechecked. Rendered pages1,3,4,5,6 remain byte-identical to their immediately prior inspected versions.'),
    independent_reviews=dict(
        calibration='Four primary intervals, equal-label versus unequal-compute distinction and audit costs verified; field-versus-future-mask chronology corrected.',
        augmentation='Quality, primary intervals, historical exposure, augmented-only timing and pooled comparisons, and separate cost ledgers verified.',
        shift='Multiclass assumptions, same-input-law moments, bound versus measured gain, TV obstruction and non-formal CNN numerical range qualification verified.'),
    scientific_source_changes_after_freeze=0,
    data_boundary='Image holdout excluded from this round fitting, but historically exposed; no pristine external replication claimed.',
    no_real_image_certificate=True)
(ROOT/'REPORT_QA.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(dict(status='PASS',pages=6,pdf_sha256=result['pdf_sha256'])))

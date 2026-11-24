"""Verify compact archive reproducibility from the 1,500 selected original JPEGs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
from PIL import Image
from prepare_external import preprocess

P=Path(__file__).resolve().parent
def main():
    rows=json.loads((P/'external'/'population_manifest.json').read_text())
    stored=np.load(P/'external'/'population.npz');images=[]
    for row in rows:
        p=P/'external'/'selected_raw'/(row['id']+'.jpg')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
        with Image.open(p) as im:a=preprocess(im)
        assert hashlib.sha256(a.tobytes()).hexdigest()==row['pixels_sha256']
        images.append(a)
    rebuilt=np.stack(images)[:,None]
    assert np.array_equal(rebuilt,stored['images'])
    assert np.array_equal(np.array([r['label'] for r in rows]),stored['labels'])
    assert np.array_equal(np.array([r['id'] for r in rows]),stored['ids'])
    result=dict(utc=datetime.now(timezone.utc).isoformat(),images=len(rows),
        every_original_jpeg_sha256_matches=True,every_preprocessed_pixel_sha256_matches=True,
        all_pixels_labels_and_ids_identical=True,full_source_zip_required=False,
        evaluated_population_rebuilt_without_predictions=True)
    (P/'EXTERNAL_REBUILD_AUDIT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

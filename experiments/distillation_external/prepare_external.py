"""Population definition and provenance; no model is imported or evaluated."""
from pathlib import Path
from datetime import datetime, timezone
import csv, gzip, hashlib, io, json, zipfile
import numpy as np
from PIL import Image, ImageOps

HERE=Path(__file__).resolve().parent
MAPPING={'Tshirts':0,'Trousers':1,'Dresses':3,'Sandals':5,'Shirts':6,'Handbags':8}
PER_CLASS=250
SELECTION_SALT='external-population-2321000'
def sha(b):return hashlib.sha256(b).hexdigest()
def preprocess(image):
    image=ImageOps.invert(image.convert('L'))
    image.thumbnail((24,24),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(28,28),0)
    canvas.paste(image,((28-image.width)//2,(28-image.height)//2))
    return np.asarray(canvas,dtype=np.uint8).copy()

def main():
    ex=HERE/'external'; z=zipfile.ZipFile(ex/'fashion-product-images-small-v1.zip')
    rows=list(csv.DictReader(io.StringIO((ex/'styles.csv').read_text(encoding='utf-8-sig'))))
    # Only the canonical images/ tree, not the redundant myntradataset/images tree.
    available=set(z.namelist()); fm=set()
    for prefix in ['train','t10k']:
        x=np.frombuffer(gzip.decompress((HERE/'data'/f'{prefix}-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,28,28)
        fm.update(sha(a.tobytes()) for a in x)
    raw=ex/'selected_raw';raw.mkdir(exist_ok=False)
    manifest=[];exclusions=[];images=[];rawseen=set();pixseen=set();counts={}
    for category,label in MAPPING.items():
        candidates=[r for r in rows if r['articleType']==category]
        candidates.sort(key=lambda r:sha((SELECTION_SALT+':'+r['id']).encode()))
        counts[category]=len(candidates);accepted=0
        for r in candidates:
            if accepted==PER_CLASS:break
            name='images/'+r['id']+'.jpg';reason=None
            if name not in available:reason='missing_file'
            else:
                b=z.read(name);bh=sha(b)
                try:
                    im=Image.open(io.BytesIO(b));im.load();a=preprocess(im);ph=sha(a.tobytes())
                except Exception as e:reason='decode_failure:'+repr(e)
                if reason is None:
                    if bh in rawseen:reason='internal_raw_exact_duplicate'
                    elif ph in pixseen:reason='internal_preprocessed_exact_duplicate'
                    elif ph in fm:reason='fashion_mnist_exact_pixel_duplicate'
            if reason:
                exclusions.append(dict(id=r['id'],category=category,reason=reason));continue
            rawseen.add(bh);pixseen.add(ph);(raw/(r['id']+'.jpg')).write_bytes(b)
            manifest.append(dict(index=len(images),id=r['id'],category=category,label=label,
                source_member=name,raw_sha256=bh,pixels_sha256=ph,width=im.width,height=im.height,
                productDisplayName=r['productDisplayName'],gender=r['gender'],
                ranking_hash=sha((SELECTION_SALT+':'+r['id']).encode())))
            images.append(a);accepted+=1
        assert accepted==PER_CLASS,(category,accepted)
    np.savez_compressed(ex/'population.npz',images=np.stack(images)[:,None],
        labels=np.array([r['label'] for r in manifest]),ids=np.array([r['id'] for r in manifest]))
    (ex/'population_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    audit=dict(utc=datetime.now(timezone.utc).isoformat(),mapping=MAPPING,per_class=PER_CLASS,
        selection_salt=SELECTION_SALT,candidate_counts=counts,selected=len(manifest),exclusions=exclusions,
        fashion_mnist_unique_pixel_hashes=len(fm),exact_overlap_with_fashion_mnist=0,
        selected_internal_exact_raw_duplicates=0,selected_internal_exact_preprocessed_duplicates=0,
        near_duplicates_or_same_product_families_not_ruled_out=True,models_loaded=False,
        preprocessing='Pillow L grayscale; invert; LANCZOS fit inside 24x24 preserving aspect; centered black 28x28 pad; uint8 / 255 at model entry',
        source_metadata_sha256=sha((HERE/'source_metadata.json').read_bytes()),
        population_sha256=sha((ex/'population.npz').read_bytes()),
        raw_archive_sha256=sha((ex/'fashion-product-images-small-v1.zip').read_bytes()),
        license='MIT as declared by dataset owner in Kaggle API metadata; independent ownership clearance not audited')
    (ex/'POPULATION_FREEZE.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    # A deterministic provenance contact sheet, not an outcome-based quality screen.
    sheet=Image.new('L',(28*12,28*6),0)
    for i in range(6):
        for j in range(12):sheet.paste(Image.fromarray(images[i*PER_CLASS+j]),(28*j,28*i))
    sheet.resize((672,336),Image.Resampling.NEAREST).save(ex/'population_contact_sheet.png')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()

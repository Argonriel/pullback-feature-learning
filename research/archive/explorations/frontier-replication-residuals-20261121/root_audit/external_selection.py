"""Independently reconstruct UT Zappos selection without experiment modules."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,datetime,gzip,hashlib,io,json,zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'distillation';E=B/'external'
digest=lambda b:hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pixels(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im.load();size=im.size;a=np.array(im.convert('L'),dtype=np.uint8)
    image=Image.fromarray(255-a);image.thumbnail((24,24),Image.Resampling.LANCZOS)
    small=np.array(image);h,w=small.shape;canvas=np.zeros((28,28),dtype=np.uint8)
    canvas[(28-h)//2:(28-h)//2+h,(28-w)//2:(28-w)//2+w]=small
    return canvas,size
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,default=E/'ut-zap50k-images.zip');ap.add_argument('--record',action='store_true');args=ap.parse_args()
    frozen=json.loads((E/'POPULATION_FREEZE.json').read_text());saved=dict(np.load(E/'population.npz'))
    manifest=json.loads((E/'population_manifest.json').read_text());download=json.loads((E/'DOWNLOAD.json').read_text())
    assert sha(args.zip)==frozen['source_archive_sha256']==download['sha256']
    assert sha(E/'population.npz')==frozen['population_sha256']
    metadata=list(csv.DictReader((E/'meta-data.csv').open()))
    groups=defaultdict(list);classes=defaultdict(set)
    for row in metadata:
        cat,sub=row['Category'],row['SubCategory']
        label=5 if cat=='Sandals' else 7 if (cat,sub)==('Shoes','Sneakers and Athletic Shoes') else 9 if (cat,sub)==('Boots','Ankle') else None
        if label is None:continue
        pid=row['CID'].split('-')[0];groups[label,pid].append(row['CID']);classes[pid].add(label)
    conflicts=sorted(pid for pid,labels in classes.items() if len(labels)>1)
    assert len(conflicts)==frozen['metadata_conflicting_product_labels']==7
    historical=set()
    for name,count in [('train',60000),('t10k',10000)]:
        a=np.frombuffer(gzip.decompress((B/'data'/f'{name}-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,28,28)
        assert len(a)==count;historical.update(digest(v.tobytes()) for v in a)
    old=dict(np.load(B/'legacy_external/population.npz'));historical.update(digest(v.tobytes()) for v in old['images'])
    oldraw={r['raw_sha256'] for r in json.loads((B/'legacy_external/population_manifest.json').read_text())}
    z=zipfile.ZipFile(args.zip);names=[n for n in z.namelist() if n.lower().endswith('.jpg')]
    members={Path(n).stem.replace('.','-'):n for n in names};assert len(members)==len(names)
    by_class={label:sorted((pid for lab,pid in groups if lab==label and len(classes[pid])==1),
        key=lambda pid:digest(f'external-products-2421000:{label}:{pid}'.encode())) for label in (5,7,9)}
    quota=min(500,min(map(len,by_class.values())));assert quota==frozen['per_class']==500
    rawseen=set();pixseen=set();brands=set();index=0;exclusions=[];counts={}
    for label,products in by_class.items():
        counts[str(label)]=len(products);accepted=0
        for pid in products:
            if accepted==quota:break
            for cid in sorted(groups[label,pid],key=lambda v:digest(('external-colors-2421001:'+v).encode())):
                reason=None
                if cid not in members:reason='missing_image'
                else:
                    raw=z.read(members[cid]);rh=digest(raw)
                    try:a,size=pixels(raw);ph=digest(a.tobytes())
                    except Exception:reason='decode_error'
                    if reason is None:
                        if rh in rawseen:reason='selected_raw_duplicate'
                        elif ph in pixseen:reason='selected_processed_duplicate'
                        elif rh in oldraw:reason='prior_myntra_raw_duplicate'
                        elif ph in historical:reason='historical_processed_duplicate'
                if reason:
                    exclusions.append(dict(cid=cid,product_id=pid,label=label,reason=reason));continue
                row=manifest[index]
                assert (row['cid'],row['product_id'],row['label'])==(cid,pid,label)
                assert row['source_member']==members[cid]
                assert row['raw_sha256']==rh==sha(E/'selected_raw'/f'{cid}.jpg')
                assert row['pixels_sha256']==ph and (row['width'],row['height'])==size
                assert np.array_equal(a,saved['images'][index,0])
                assert str(saved['ids'][index])==cid and str(saved['groups'][index])==pid and saved['labels'][index]==label
                assert row['product_ranking_sha256']==digest(f'external-products-2421000:{label}:{pid}'.encode())
                assert row['color_ranking_sha256']==digest(('external-colors-2421001:'+cid).encode())
                rawseen.add(rh);pixseen.add(ph);brands.add(Path(members[cid]).parent.name)
                accepted+=1;index+=1;break
        assert accepted==quota
    assert counts==frozen['eligible_unique_products']
    assert exclusions==frozen['excluded_candidates']==[]
    assert index==len(manifest)==1500 and len(set(saved['groups']))==1500
    assert len(brands)==frozen['selected_brands']==435
    for row in json.loads((E/'ACQUISITION_METADATA.json').read_text()):assert sha(E/row['name'])==row['sha256']
    out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive_sha256=sha(args.zip),
        selected_images=index,unique_products=1500,eligible_counts=counts,conflicting_product_ids=conflicts,
        examined_selection_prefix=index,exclusions=exclusions,exact_pixel_overlap=0,brands=len(brands),
        independently_rebuilt_selection_order=True,independently_rebuilt_pixels=True,
        limitations=['Metadata taxonomy may be imperfect.','Product grouping does not remove near-design or brand dependence.',
                    'Replaying skipped candidates requires original source ZIP; selected originals suffice for pixel replay.'])
    if args.record:
        (ROOT/'root_audit/EXTERNAL_SELECTION.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()

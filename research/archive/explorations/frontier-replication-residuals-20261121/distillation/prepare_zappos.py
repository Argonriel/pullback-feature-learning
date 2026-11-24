"""Freeze an independent product-grouped footwear population before inference."""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,io,json,zipfile
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageOps

P=Path(__file__).resolve().parent
MAPPING={5:('Sandals',None),7:('Shoes','Sneakers and Athletic Shoes'),9:('Boots','Ankle')}
PER_CLASS=500
def sha(b):return hashlib.sha256(b).hexdigest()
def preprocess(image):
    image=ImageOps.invert(image.convert('L'))
    image.thumbnail((24,24),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(28,28),0)
    canvas.paste(image,((28-image.width)//2,(28-image.height)//2))
    return np.asarray(canvas,dtype=np.uint8).copy()

def main():
    e=P/'external';rows=list(csv.DictReader((e/'meta-data.csv').open()))
    z=zipfile.ZipFile(e/'ut-zap50k-images.zip')
    members={Path(n).stem.replace('.','-'):n for n in z.namelist() if n.lower().endswith('.jpg')}
    historic=set()
    for prefix in ['train','t10k']:
        a=np.frombuffer(gzip.decompress((P/'data'/f'{prefix}-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,28,28)
        historic.update(sha(v.tobytes()) for v in a)
    legacy=np.load(P/'legacy_external'/'population.npz')
    historic.update(sha(v.tobytes()) for v in legacy['images'])
    legacy_raw={r['raw_sha256'] for r in json.loads((P/'legacy_external'/'population_manifest.json').read_text())}
    groups=defaultdict(list);product_labels=defaultdict(set)
    for r in rows:
        r['product_id']=r['CID'].split('-')[0]
        for label,(cat,sub) in MAPPING.items():
            if r['Category']==cat and (sub is None or r['SubCategory']==sub):
                groups[label,r['product_id']].append(r);product_labels[r['product_id']].add(label)
    eligible={c:sorted([pid for cc,pid in groups if cc==c and len(product_labels[pid])==1],
        key=lambda pid:sha(f'external-products-2421000:{c}:{pid}'.encode())) for c in MAPPING}
    counts={str(c):len(ids) for c,ids in eligible.items()}
    # Common-size fallback is determined exclusively by metadata before image selection.
    target=min(PER_CLASS,min(map(len,eligible.values())))
    assert target>=400,'Fewer than 1200 products: a new pre-outcome source review would be required.'
    selected=[];arrays=[];exclusions=[];rawseen=set();pixseen=set();out=e/'selected_raw';out.mkdir(exist_ok=False)
    for label in MAPPING:
        accepted=0
        for pid in eligible[label]:
            if accepted==target:break
            variants=sorted(groups[label,pid],key=lambda r:sha(('external-colors-2421001:'+r['CID']).encode()))
            for r in variants:
                cid=r['CID'];reason=None
                if cid not in members:reason='missing_image'
                else:
                    b=z.read(members[cid]);rh=sha(b)
                    try:
                        im=Image.open(io.BytesIO(b));im.load();a=preprocess(im);ph=sha(a.tobytes())
                    except Exception as exc:reason='decode_error:'+repr(exc)
                    if reason is None:
                        if rh in rawseen:reason='selected_raw_duplicate'
                        elif ph in pixseen:reason='selected_processed_duplicate'
                        elif rh in legacy_raw:reason='prior_myntra_raw_duplicate'
                        elif ph in historic:reason='historical_processed_duplicate'
                if reason:
                    exclusions.append(dict(cid=cid,product_id=pid,label=label,reason=reason));continue
                rawseen.add(rh);pixseen.add(ph);(out/(cid+'.jpg')).write_bytes(b)
                row=dict(index=len(selected),cid=cid,product_id=pid,label=label,
                    category=r['Category'],subcategory=r['SubCategory'],brand=Path(members[cid]).parent.name,
                    gender=r['Gender'],source_member=members[cid],width=im.width,height=im.height,
                    raw_sha256=rh,pixels_sha256=ph,product_ranking_sha256=sha(f'external-products-2421000:{label}:{pid}'.encode()),
                    color_ranking_sha256=sha(('external-colors-2421001:'+cid).encode()))
                selected.append(row);arrays.append(a);accepted+=1;break
        assert accepted==target,(label,accepted)
    assert len({r['product_id'] for r in selected})==len(selected)
    np.savez_compressed(e/'population.npz',images=np.stack(arrays)[:,None],labels=np.array([r['label'] for r in selected]),
        ids=np.array([r['cid'] for r in selected]),groups=np.array([r['product_id'] for r in selected]))
    (e/'population_manifest.json').write_text(json.dumps(selected,indent=2))
    record=dict(utc=datetime.now(timezone.utc).isoformat(),source='UT Zappos50K, official UT Austin source; README version1.2; site version2.1',
        license_terms='Academic, non-commercial use only; retain official README and cite Yu and Grauman (CVPR2014, ICCV2017).',
        mapping={str(k):list(v) for k,v in MAPPING.items()},eligible_unique_products=counts,per_class=target,
        selected_images=len(selected),selected_unique_products=len(selected),selected_brands=len({r['brand'] for r in selected}),
        metadata_conflicting_product_labels=sum(len(v)>1 for v in product_labels.values()),
        ambiguous_product_exclusions=[dict(product_id=pid,mapped_labels=sorted(vals)) for pid,vals in sorted(product_labels.items()) if len(vals)>1],excluded_candidates=exclusions,
        selected_raw_exact_duplicates=0,selected_pixel_exact_duplicates=0,exact_pixel_overlap_with_fashion_mnist_and_prior_myntra=0,
        product_group_rule='One deterministically selected color image per unique ProductID; sample groups separately by mapped category.',
        population_sha256=sha((e/'population.npz').read_bytes()),
        source_archive_sha256=json.loads((e/'DOWNLOAD.json').read_text())['sha256'],
        preprocessing_unchanged_from_prior=True,models_loaded=False,outcomes_observed=False,
        near_duplicate_designs_and_brand_dependencies_not_ruled_out=True)
    (e/'POPULATION_FREEZE.json').write_text(json.dumps(record,indent=2))
    sheet=Image.new('L',(28*16,28*3),0)
    for i in range(3):
        for j in range(16):sheet.paste(Image.fromarray(arrays[i*target+j]),(j*28,i*28))
    sheet.resize((896,168),Image.Resampling.NEAREST).save(e/'contact_sheet.png')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()

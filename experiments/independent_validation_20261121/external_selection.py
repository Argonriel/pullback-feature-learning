"""Independently replay the complete ranked-prefix external population selection.

No experiment-author module or model is imported. The full original ZIP is
required to verify first-eligible selection, including rejected candidates.
For an archive which retains only selected_raw/, supply --zip /path/to/the.zip.
Those selected bytes suffice for preprocessing/prediction replay, but cannot
alone establish why higher-ranked unselected candidates were excluded.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,datetime,gzip,hashlib,io,json,struct,zipfile
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
BRANCH=HERE.parent/'distillation'
EXTERNAL=BRANCH/'external'
MAPPING=(('Tshirts',0),('Trousers',1),('Dresses',3),('Sandals',5),('Shirts',6),('Handbags',8))
SALT='external-population-2321000'
QUOTA=250

def digest(data):return hashlib.sha256(data).hexdigest()
def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()
def canonical_digest(value):
    return digest(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8'))
def rank(identifier):return digest((SALT+':'+identifier).encode('ascii'))
def preprocess_independent(raw):
    with Image.open(io.BytesIO(raw)) as original:
        original.load();width,height=original.size
        # Independent implementation: explicit uint8 inversion and array padding.
        luminance=np.array(original.convert('L'),dtype=np.uint8)
    inverted=Image.fromarray(255-luminance)
    inverted.thumbnail((24,24),Image.Resampling.LANCZOS)
    resized=np.array(inverted,dtype=np.uint8)
    height_small,width_small=resized.shape
    canvas=np.zeros((28,28),dtype=np.uint8)
    y0=(28-height_small)//2;x0=(28-width_small)//2
    canvas[y0:y0+height_small,x0:x0+width_small]=resized
    return canvas,width,height

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=EXTERNAL/'fashion-product-images-small-v1.zip',dest='zip_path')
    parser.add_argument('--output',type=Path,default=HERE/'EXTERNAL_SELECTION.json')
    args=parser.parse_args();archive=args.zip_path.resolve()
    if not archive.is_file():
        raise FileNotFoundError('The full original ZIP is needed to audit skipped candidates; use --zip. Selected raw images alone cannot verify first-eligible selection.')
    assert not args.output.exists(),'Use a fresh output file; an earlier audit is never overwritten.'
    frozen=json.loads((EXTERNAL/'POPULATION_FREEZE.json').read_text(encoding='utf-8'))
    source=json.loads((BRANCH/'SOURCE_FREEZE.json').read_text(encoding='utf-8'))
    metadata=json.loads((BRANCH/'source_metadata.json').read_text(encoding='utf-8'))
    download=json.loads((EXTERNAL/'DOWNLOAD.json').read_text(encoding='utf-8'))
    recorded=json.loads((EXTERNAL/'population_manifest.json').read_text(encoding='utf-8'))
    saved=dict(np.load(EXTERNAL/'population.npz'))
    assert frozen['mapping']==dict(MAPPING) and frozen['per_class']==QUOTA and frozen['selection_salt']==SALT
    assert metadata['ref']=='paramaggarwal/fashion-product-images-small' and metadata['currentVersionNumber']==1
    assert metadata['licenseName']==metadata['licenseNameNullable']=='MIT'
    assert file_digest(BRANCH/'source_metadata.json')==frozen['source_metadata_sha256']
    assert file_digest(EXTERNAL/'population.npz')==frozen['population_sha256']
    assert file_digest(archive)==frozen['raw_archive_sha256']==download['sha256']
    assert archive.stat().st_size==download['bytes']
    for relative in ['source_metadata.json','external/POPULATION_FREEZE.json','external/population_manifest.json',
                     'external/population.npz','external/styles.csv','external/DOWNLOAD.json']:
        assert file_digest(BRANCH/relative)==source['hashes'][relative],relative

    fashion_hashes=set();fashion_records=[]
    for prefix,expected_count in [('train',60000),('t10k',10000)]:
        path=BRANCH/'data'/f'{prefix}-images-idx3-ubyte.gz'
        raw=gzip.decompress(path.read_bytes());magic,count,height,width=struct.unpack('>IIII',raw[:16])
        assert (magic,count,height,width)==(2051,expected_count,28,28)
        assert len(raw)==16+count*28*28
        data=np.frombuffer(raw,np.uint8,offset=16).reshape(count,28,28)
        before=len(fashion_hashes)
        fashion_hashes.update(digest(image.tobytes()) for image in data)
        compressed_sha=file_digest(path)
        assert compressed_sha==source['hashes'][f'data/{path.name}']
        fashion_records.append(dict(split=prefix,images=count,new_unique_hashes=len(fashion_hashes)-before,
                                    compressed_sha256=compressed_sha,idx_header=[magic,count,height,width]))
    assert len(fashion_hashes)==70000==frozen['fashion_mnist_unique_pixel_hashes']

    style_bytes=(EXTERNAL/'styles.csv').read_bytes()
    rows=list(csv.DictReader(io.StringIO(style_bytes.decode('utf-8-sig'))))
    assert len({row['id'] for row in rows})==len(rows),'Repeated CSV IDs require a separate tie policy.'
    expected_count=QUOTA*len(MAPPING)
    assert saved['images'].shape==(expected_count,1,28,28) and saved['images'].dtype==np.uint8
    assert len(recorded)==len(saved['labels'])==len(saved['ids'])==expected_count
    selected_ids=[];accepted_raw={};accepted_pixels={};excluded=[];class_records=[];candidate_total=0
    inspected=0;decoded=0;raw_bytes_read=0;pixel_differences=0;raw_exact_checks=0;manifest_fields_checked=0
    selection_digest=hashlib.sha256()
    with zipfile.ZipFile(archive,'r') as zip_file:
        members=zip_file.namelist();member_counts=Counter(members)
        assert zip_file.read('styles.csv')==style_bytes
        assert zip_file.read('myntradataset/styles.csv')==style_bytes
        canonical_images={name for name in members if name.startswith('images/') and name.endswith('.jpg')}
        assert all(member_counts[name]==1 for name in canonical_images)
        for category,label in MAPPING:
            # Rank every eligible metadata record, not merely retained IDs.
            candidates=[row for row in rows if row['articleType']==category]
            candidate_total+=len(candidates)
            ordered=sorted(candidates,key=lambda row:rank(row['id']))
            assert len({rank(row['id']) for row in ordered})==len(ordered)
            assert len(ordered)==frozen['candidate_counts'][category]
            accepted=0;class_exclusions=[];last_rank=0
            for position,row in enumerate(ordered,start=1):
                if accepted==QUOTA:break
                inspected+=1;last_rank=position;identifier=row['id'];name=f'images/{identifier}.jpg';reason=None;duplicate_of=None
                if name not in canonical_images:
                    reason='missing_file'
                else:
                    raw=zip_file.read(name);raw_bytes_read+=len(raw);raw_hash=digest(raw)
                    try:
                        pixels,width,height=preprocess_independent(raw);decoded+=1;pixel_hash=digest(pixels.tobytes())
                    except Exception as exc:
                        reason='decode_failure:'+repr(exc)
                    if reason is None:
                        if raw_hash in accepted_raw:
                            reason='internal_raw_exact_duplicate';duplicate_of=accepted_raw[raw_hash]
                        elif pixel_hash in accepted_pixels:
                            reason='internal_preprocessed_exact_duplicate';duplicate_of=accepted_pixels[pixel_hash]
                        elif pixel_hash in fashion_hashes:
                            reason='fashion_mnist_exact_pixel_duplicate'
                if reason is not None:
                    exclusion=dict(id=identifier,category=category,reason=reason)
                    excluded.append(exclusion)
                    class_exclusions.append(dict(**exclusion,candidate_rank=position,ranking_hash=rank(identifier),duplicate_of_selected_id=duplicate_of))
                    continue
                # Every preceding candidate has just been checked; this is the next eligible ID.
                index=len(selected_ids);assert identifier==recorded[index]['id']==str(saved['ids'][index])
                disk_raw=EXTERNAL/'selected_raw'/f'{identifier}.jpg'
                assert disk_raw.read_bytes()==raw;raw_exact_checks+=1
                assert file_digest(disk_raw)==source['hashes'][f'external/selected_raw/{identifier}.jpg']
                differences=int(np.count_nonzero(pixels!=saved['images'][index,0]));pixel_differences+=differences;assert differences==0
                expected=dict(index=index,id=identifier,category=category,label=label,source_member=name,
                    raw_sha256=raw_hash,pixels_sha256=pixel_hash,width=width,height=height,
                    productDisplayName=row['productDisplayName'],gender=row['gender'],ranking_hash=rank(identifier))
                assert recorded[index]==expected,(index,identifier);manifest_fields_checked+=len(expected)
                assert int(saved['labels'][index])==label
                selected_ids.append(identifier);accepted_raw[raw_hash]=identifier;accepted_pixels[pixel_hash]=identifier;accepted+=1
                selection_digest.update(json.dumps(expected,sort_keys=True,separators=(',',':')).encode('utf-8'))
            assert accepted==QUOTA
            class_records.append(dict(category=category,label=label,all_candidates_ranked=len(ordered),
                ranked_candidate_ids_sha256=canonical_digest([row['id'] for row in ordered]),
                inspected_prefix_length=last_rank,selected=accepted,excluded_count=len(class_exclusions),exclusions=class_exclusions,
                last_selected_id=selected_ids[-1],last_selected_ranking_hash=rank(selected_ids[-1])))
    assert excluded==frozen['exclusions'] and len(excluded)==15
    assert Counter(row['reason'] for row in excluded)=={'internal_raw_exact_duplicate':15}
    assert len(selected_ids)==len(set(selected_ids))==len(accepted_raw)==len(accepted_pixels)==expected_count
    assert set(path.name for path in (EXTERNAL/'selected_raw').iterdir() if path.is_file())=={identifier+'.jpg' for identifier in selected_ids}
    assert not (set(accepted_pixels)&fashion_hashes)
    assert np.array_equal(np.unique(saved['labels'],return_counts=True)[1],np.full(6,QUOTA))
    result=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Independent full ranked-prefix selection replay, not just reconstruction of already selected images.',
        auditor_sha256=file_digest(Path(__file__)),source_freeze_sha256=file_digest(BRANCH/'SOURCE_FREEZE.json'),
        population_freeze_sha256=file_digest(EXTERNAL/'POPULATION_FREEZE.json'),population_npz_sha256=file_digest(EXTERNAL/'population.npz'),
        population_manifest_sha256=file_digest(EXTERNAL/'population_manifest.json'),metadata_sha256=file_digest(BRANCH/'source_metadata.json'),
        zip_path=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=download['sha256'],
        zip_members=len(members),canonical_jpeg_members=len(canonical_images),styles_csv_rows=len(rows),
        styles_csv_sha256=digest(style_bytes),styles_csv_matches_both_zip_members=True,
        mapping=dict(MAPPING),selection_salt=SALT,per_class_quota=QUOTA,
        full_relevant_candidates_ranked=candidate_total,ranked_prefix_candidates_inspected=inspected,
        images_decoded=decoded,zip_image_bytes_read=raw_bytes_read,
        selected_ids_exactly_reproduced=len(selected_ids),selected_raw_byte_equality_checks=raw_exact_checks,
        selected_pixel_array_equality_checks=len(selected_ids),pixel_value_discrepancies=pixel_differences,
        independently_rebuilt_manifest_fields=manifest_fields_checked,
        rebuilt_manifest_sequence_sha256=selection_digest.hexdigest(),selected_ids=selected_ids,
        exact_excluded_id_reason_sequence_reproduced=excluded,class_details=class_records,
        fashion_mnist_splits=fashion_records,fashion_mnist_unique_pixel_hashes=len(fashion_hashes),
        selected_fashion_mnist_exact_overlap=0,selected_internal_raw_duplicates=0,selected_internal_pixel_duplicates=0,
        model_imports=0,model_forward_examples=0,population_or_source_mutations=0,
        compact_archive_reproducibility=dict(
            retained='styles.csv, metadata, 1500 selected original JPEGs, processed population, manifests and freeze hashes',
            sufficient_for='Reconstruct all retained pixels and labels, verify selected bytes, and replay model evaluation.',
            insufficient_without_full_zip='Independently prove that omitted higher-ranked candidates fail eligibility; the 15 rejected original JPEGs are not in selected_raw.',
            replay='python external_selection.py --zip /path/to/fashion-product-images-small-v1.zip --output /fresh/path/EXTERNAL_SELECTION.json',
            required_full_zip_sha256=download['sha256'],download_url=download['url']),
        license=dict(declared_license=metadata['licenseName'],declared_by='Dataset owner in retained Kaggle API metadata',
            dataset_ref=metadata['ref'],dataset_version=metadata['currentVersionNumber'],
            independent_copyright_ownership_clearance=False,claim='Declared MIT metadata verified; this is not independent ownership or redistribution-rights proof.'),
        limitations=['Exact hashing does not rule out near duplicates or related product families.',
            'Six metadata categories are mapped to six Fashion-MNIST labels; semantic mapping is not independent manual label validation.',
            'This fixed balanced product-photo population is not representative of all natural images or all ten Fashion-MNIST classes.',
            'No model fitting, predictive scoring, statistical resampling or network access is part of this audit.'])
    with args.output.open('x',encoding='utf-8') as output:json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ['status','full_relevant_candidates_ranked','ranked_prefix_candidates_inspected',
        'selected_ids_exactly_reproduced','selected_raw_byte_equality_checks','pixel_value_discrepancies',
        'fashion_mnist_unique_pixel_hashes','selected_fashion_mnist_exact_overlap','model_forward_examples']},indent=2))

if __name__=='__main__':main()

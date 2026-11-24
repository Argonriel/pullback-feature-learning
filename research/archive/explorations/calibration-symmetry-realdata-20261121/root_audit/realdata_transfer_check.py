"""Independent post-timing IDX/order/flip/CNN and classwise transfer audit.

No training, fitting, or policy modification. Direct GPU examples are charged.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json,os,struct,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REAL=ROOT/'realdata'
EXPECTED={
 'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
 'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
 't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
 't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
CLASSES=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def cw(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def decode(prefix):
    image=gzip.decompress((REAL/'data'/f'{prefix}-images-idx3-ubyte.gz').read_bytes())
    label=gzip.decompress((REAL/'data'/f'{prefix}-labels-idx1-ubyte.gz').read_bytes())
    magic,n,r,c=struct.unpack('>IIII',image[:16]);lm,ln=struct.unpack('>II',label[:8])
    assert magic==2051 and lm==2049 and n==ln and r==28 and c==28
    assert len(image)==16+n*r*c and len(label)==8+n
    return np.frombuffer(image,dtype='u1',offset=16).reshape(n,1,r,c).copy(),np.frombuffer(label,dtype='u1',offset=8).astype('int64')

class IndependentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x):return self.head(self.features(x))

def metrics(logits,labels):
    z=np.asarray(logits,dtype='float64');shift=z.max(1,keepdims=True);exp=np.exp(z-shift);prob=exp/exp.sum(1,keepdims=True)
    loss=np.log(exp.sum(1))+shift[:,0]-z[np.arange(len(z)),labels]
    return loss,z.argmax(1),prob.max(1)

@torch.inference_mode()
def run():
    timing=REAL/'TIMING_COMPLETE.json';assert timing.exists(),'Wait until benchmark completion'
    out=HERE/'realdata_transfer_v2';out.mkdir(exist_ok=False)
    start_time=now();timing_record=json.loads(timing.read_text())
    manifest=json.loads((REAL/'TEST_COMPLETE.json').read_text())['manifest']
    model_commit=json.loads((REAL/'MODEL_COMMIT.json').read_text())['manifest']
    paths=[REAL/name for name in ('data.py','model.py','policy.py','study.py','SOURCE_FREEZE.json','MODEL_COMMIT.json','POLICY_COMMIT.json','TEST_COMPLETE.json','temperatures.json')]
    frozen_before={p.name:sha(p) for p in paths}
    file_rows=[]
    for name,md5 in EXPECTED.items():
        path=REAL/'data'/name;actual=hashlib.md5(path.read_bytes()).hexdigest();assert actual==md5,(name,actual)
        file_rows.append(dict(file=name,md5=actual,sha256=sha(path),published_md5_matches=True))
    images,labels=decode('t10k');assert images.shape==(10000,1,28,28) and np.array_equal(np.bincount(labels),np.full(10,1000))
    order=np.random.default_rng(2136000).permutation(10000);ordered=images[order];y=labels[order]
    # Independent CPU row-wise reversal confirms exactly the intended image width action.
    partner=ordered[:,:,:,::-1].copy();assert np.array_equal(partner[:,:,:,::-1],ordered)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    x=torch.tensor(ordered,dtype=torch.float32,device='cuda')/255.
    reverse=torch.arange(27,-1,-1,device='cuda');tx=x.index_select(3,reverse)
    normalization_difference=float(np.max(abs(tx.cpu().numpy()-partner.astype('float32')/255.)))
    assert normalization_difference<=np.finfo(np.float32).eps,normalization_difference
    assert torch.equal(torch.flip(x,[-1]),tx) and torch.equal(tx.index_select(3,reverse),x)
    temperatures={r['seed']:r['temperature'] for r in json.loads((REAL/'temperatures.json').read_text())['rows']}
    replay=[];classwise=[];confusions=[];overall=[];total_examples=0;invocations=0
    pixelstats=[]
    cols=np.linspace(-1,1,28)[None,None,:]
    for label in range(10):
        xx=images[labels==label,0].astype('float64')/255.;meanimage=xx.mean(0)
        moment=(xx*cols).sum((1,2))/np.maximum(xx.sum((1,2)),1e-30)
        pixelstats.append(dict(class_id=label,class_name=CLASSES[label],n=len(xx),mean_horizontal_centroid=float(moment.mean()),centroid_se=float(moment.std(ddof=1)/np.sqrt(len(xx))),
          class_mean_image_flip_l1=float(np.mean(abs(meanimage-meanimage[:,::-1])))))
    for seed in (1,2,3):
        checkpoint=REAL/'models'/f'{seed}.pt';assert sha(checkpoint)==model_commit[checkpoint.name]
        net=IndependentCNN();state=torch.load(checkpoint,map_location='cpu',weights_only=True);net.load_state_dict(state['state_dict']);net.to('cuda').eval()
        temp=temperatures[seed];predpath=REAL/'predictions'/f'{seed}_cuda.npz';assert sha(predpath)==manifest[predpath.name]
        saved=np.load(predpath);assert np.array_equal(saved['original_indices'],order) and np.array_equal(saved['labels'],y)
        # One separate-view batch per model checks the exact original deployment path.
        raw_batch=net(x[:256])/temp;partner_batch=net(tx[:256])/temp;total_examples+=512;invocations+=2
        raw=np.empty((10000,10),np.float32);flipped=np.empty_like(raw)
        torch.cuda.synchronize();begin_time=time.perf_counter()
        for begin in range(0,10000,256):
            end=min(begin+256,10000);n=end-begin
            both=net(torch.cat([x[begin:end],tx[begin:end]],dim=0))/temp
            raw[begin:end]=both[:n].cpu().numpy();flipped[begin:end]=both[n:].cpu().numpy();total_examples+=2*n;invocations+=1
        torch.cuda.synchronize();seconds=time.perf_counter()-begin_time
        names=list(saved['names']);sr=saved['logits'][names.index('raw')];sf=saved['logits'][names.index('full_fused')]
        reconstructed=2*sf-sr;full=.5*(raw+flipped)
        checks=dict(original_batch_max_error=float(np.max(abs(raw_batch.cpu().numpy()-sr[:256]))),
            partner_batch_reconstruction_max_error=float(np.max(abs(partner_batch.cpu().numpy()-reconstructed[:256]))),
            original_all_max_error=float(np.max(abs(raw-sr))),full_all_max_error=float(np.max(abs(full-sf))),
            partner_all_reconstruction_max_error=float(np.max(abs(flipped-reconstructed))),
            reconstructed_partner_changed_classes=int(np.sum(flipped.argmax(1)!=reconstructed.argmax(1))))
        assert checks['original_batch_max_error']==0.,checks
        assert max(checks[k] for k in checks if k.endswith('max_error'))<1e-4,checks
        assert checks['reconstructed_partner_changed_classes']==0,checks
        trueg=.5*(np.logaddexp.reduce(raw.astype('float64'),axis=1)+np.logaddexp.reduce(flipped.astype('float64'),axis=1))-np.logaddexp.reduce(full.astype('float64'),axis=1)
        rawloss,_,_=metrics(raw,y);fliploss,_,_=metrics(flipped,y);fullloss,_,_=metrics(full,y)
        pairgain=.5*(rawloss+fliploss)-fullloss
        identity_error=float(np.max(abs(pairgain-trueg)));assert identity_error<1e-5
        replay.append(dict(seed=seed,temperature=temp,full_direct_forward_seconds=seconds,additional_forward_examples=20512,additional_forward_invocations=42,
          model_sha256=sha(checkpoint),saved_prediction_sha256=sha(predpath),paired_jensen_identity_error=identity_error,**checks))
        np.savez_compressed(out/f'{seed}_direct_logits.npz',original_indices=order,labels=y,raw=raw,partner=flipped,full=full)
        for view,z in [('original',raw),('flipped_only',flipped),('full_average',full)]:
            loss,pred,confidence=metrics(z,y);cm=np.zeros((10,10),dtype=int);np.add.at(cm,(y,pred),1)
            overall.append(dict(seed=seed,view=view,loss=float(loss.mean()),error=float(np.mean(pred!=y)),confidence=float(confidence.mean()),n=10000))
            for label in range(10):
                selected=y==label
                classwise.append(dict(seed=seed,view=view,class_id=label,class_name=CLASSES[label],n=int(selected.sum()),loss=float(loss[selected].mean()),
                    error=float(np.mean(pred[selected]!=y[selected])),confidence=float(confidence[selected].mean())))
                for guess in range(10):confusions.append(dict(seed=seed,view=view,true_class=label,predicted_class=guess,count=int(cm[label,guess])))
        # The CPU saved outputs share exact image/label order as well; no CPU forward calls added.
        cp=REAL/'predictions'/f'{seed}_cpu.npz';assert sha(cp)==manifest[cp.name];cs=np.load(cp)
        assert np.array_equal(cs['original_indices'],order) and np.array_equal(cs['labels'],y)
    assert total_examples==61536 and invocations==126
    assert frozen_before=={p.name:sha(p) for p in paths},'Frozen source changed during audit'
    cw(out/'classwise.csv',classwise);cw(out/'confusions.csv',confusions);cw(out/'overall.csv',overall);cw(out/'pixel_asymmetry.csv',pixelstats)
    result=dict(timestamp_utc=now(),audit_started_utc=start_time,timing_completed_utc=timing_record['utc'],status='PASS',
      conclusion='No image-order, label-order, preprocessing-axis, half-turn implementation, checkpoint, or saved-forward discrepancy was found in these checks. Negative original-distribution transfer is reproduced.',
      official_source='https://github.com/zalandoresearch/fashion-mnist#Get-the-Data',official_files=file_rows,
      original_image_array_sha256=ah(images),original_labels_sha256=ah(labels),permutation_sha256=ah(order),ordered_images_sha256=ah(ordered),ordered_labels_sha256=ah(y),
      partner_images_sha256=ah(partner),all_cpu_cuda_orders_match=True,all10classes1000examples=True,numpy_cuda_normalization_max_error=normalization_difference,
      extra_forward_examples=total_examples,extra_forward_invocations=invocations,models=3,replays=replay,overall=overall,
      frozen_source_hashes=frozen_before,source_modified=False,retuning=False)
    (HERE/'REALDATA_TRANSFER_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    note='''# Independent real-data negative-transfer check

The negative transfer on the original Fashion-MNIST test distribution was reproduced. This audit found no image/label ordering error, preprocessing-axis error, wrong flip dimension, checkpoint mismatch, or materially inconsistent saved CNN output. It ran only after the serving benchmarks completed and did not train, refit, or tune anything.

The four compressed IDX files match the MD5 values published by the [official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data). An independent IDX decoder checked magic values, byte lengths, 28×28 shape, and all 10,000 test labels, with 1,000 examples per class. The frozen permutation (seed 2136000) exactly matches all six CPU/GPU prediction archives, including all image-label pairings. Input normalization is uint8-to-float32 division by 255. The independently computed width-index reversal equals `torch.flip(x,[-1])` exactly and is an exact involution; rows, channels, and labels are unchanged.

An independently instantiated CNN with the saved state dictionaries replayed all original and partner inputs for all three seeds. The first 256 original examples per seed also replayed the exact separate-view deployment path. The complete audit added **61,536 model-example forwards in 126 invocations** (60,000 for full paired replay and 1,536 for separate-view checks); these are evaluation-only audit costs, not part of deployment or the timing benchmark. Direct logits are retained under `root_audit/realdata_transfer_v2/`. The original saved partner diagnostic was reconstructed as twice the averaged logit minus the first logit; this reconstruction can differ slightly in floating point and batch-shape kernel arithmetic, so it was checked against actual partner forwards rather than silently treated as a new exact forward result. The first audit attempt stopped before any CNN forwards because an exact NumPy-versus-CUDA normalization assertion was too strict; its source and failure record are preserved. The corrected check records the normalization difference and allows one float32 epsilon, while the two torch width-reversal implementations must still agree exactly.

| Model seed | Original loss / error | Flipped-only loss / error | Full-average loss / error |
|---|---:|---:|---:|
'''
    for seed in (1,2,3):
        rr={r['view']:r for r in overall if r['seed']==seed}
        note+=f"| {seed} | {rr['original']['loss']:.6f} / {rr['original']['error']:.2%} | {rr['flipped_only']['loss']:.6f} / {rr['flipped_only']['error']:.2%} | {rr['full_average']['loss']:.6f} / {rr['full_average']['error']:.2%} |\n"
    note+='''
`realdata_transfer_v2/classwise.csv` contains all 90 seed/view/class rows; `confusions.csv` contains every 10×10 confusion table, including zero cells. `pixel_asymmetry.csv` gives class-conditioned horizontal brightness-centroid means and class-mean-image flip differences as descriptive input-distribution diagnostics. They can support the plausibility of orientation imbalance but are not a proof identifying the causal source of failure or a population symmetry-defect bound.

The full-average Jensen identity holds for the equal-weight mixture of an original image and its flipped partner carrying the same class label. That paired construction changes the evaluated input distribution. A semantic intuition that horizontal reflection preserves clothing class does not establish that the original test distribution is reflection invariant. Therefore positive paired-set gain and harmful original-set transfer are compatible. The paired diagnostic is useful algebraic evidence, not a substitute for reporting the negative original-distribution outcome.

All source hashes, numerical discrepancies, ordering hashes, explicit extra query counts, and aggregate metrics are in `REALDATA_TRANSFER_CHECK.json`. This audit rules out the implementation mistakes it actually checks; it does not prove that no other modeling limitation exists.
'''
    (HERE/'REALDATA_TRANSFER_CHECK.md').write_text(note,encoding='utf-8')
    print(json.dumps(dict(status='PASS',overall=overall,replays=replay,extra_forward_examples=total_examples),indent=2))
if __name__=='__main__':run()

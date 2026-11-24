"""Post-fit descriptive diagnostics requested by the training protocol.

No temperature, checkpoint, or policy is changed. Temperature-fit logits are
reused; validation logits cost 5,000 additional forward examples per model.
The checkpoint used validation for selection, so these are diagnostics rather
than a fresh unbiased model-selection evaluation.
"""
from datetime import datetime,timezone
import csv,json
import numpy as np
import torch
from data import HERE,arrays,sha
from model import runtime,load_model,predict_chunks


@torch.inference_mode()
def main():
    runtime();images,labels=arrays('train');ids=np.load(HERE/'splits.npz')['validation']
    target=HERE/'TEMPERATURE_DIAGNOSTICS_START.json'
    with target.open('x') as f:json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        descriptive_after_test=True,new_forward_examples=15000,changes_to_choices=False),f,indent=2)
    x=torch.as_tensor(images[ids],device='cuda',dtype=torch.float32)/255;rows=[]
    for seed in (1,2,3):
        saved=np.load(HERE/'models'/f'{seed}_temperature.npz');t=float(saved['temperature'])
        net=load_model(HERE/'models'/f'{seed}.pt','cuda');v=predict_chunks(net,x).cpu()
        for split,z,y,calls in [('temperature_fit',torch.as_tensor(saved['logits']),torch.as_tensor(saved['labels']),0),
                                ('checkpoint_validation',v,torch.as_tensor(labels[ids]),5000)]:
            assert torch.equal(z.argmax(1),(z/t).argmax(1))
            rows.append(dict(seed=seed,split=split,temperature=t,n=len(y),raw_loss=float(torch.nn.functional.cross_entropy(z,y)),
                calibrated_loss=float(torch.nn.functional.cross_entropy(z/t,y)),error=float((z.argmax(1)!=y).float().mean()),additional_forward_examples=calls))
    with (HERE/'temperature_diagnostics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

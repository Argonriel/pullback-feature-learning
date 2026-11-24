"""Fixed bounded training and independent positive-temperature calibration."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json, time
import numpy as np
import torch
from model import CNN, runtime, predict_chunks
from data import HERE, arrays, splits, sha
CONFIG = dict(seeds=[1,2,3], init_base=2132000, order_base=2133000, split_seed=2131000,
    epochs=12, validation_epochs=[3,6,9,12], batch=256, optimizer='Adam', learning_rate=.001,
    weight_decay=.0001, augmentation=False, dtype='float32', tf32=False,
    temperature_min=.25, temperature_max=4., temperature_iterations=80)


def now(): return datetime.now(timezone.utc).isoformat()
def write_json(path, payload):
    with Path(path).open('x') as stream: json.dump(payload, stream, indent=2)
def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def freeze():
    names = ['data.py','model.py','train.py','TRAINING_PROTOCOL.md']
    write_json(HERE/'TRAINING_FREEZE.json', dict(utc=now(), config=CONFIG,
        source_hashes={name:sha(HERE/name) for name in names}, test_accessed=False))


def temperature_fit(logits, labels):
    z = np.asarray(logits, dtype=np.float64); labels=np.asarray(labels)
    zy=z[np.arange(len(z)),labels]
    def derivative(beta):
        v=beta*z;v-=v.max(1,keepdims=True);w=np.exp(v);w/=w.sum(1,keepdims=True)
        return float(np.mean(np.sum(w*z,axis=1)-zy))
    lower, upper = .25,4.
    if derivative(lower)>=0: beta=lower
    elif derivative(upper)<=0: beta=upper
    else:
        for _ in range(80):
            mid=(lower+upper)/2
            if derivative(mid)>0:upper=mid
            else:lower=mid
        beta=(lower+upper)/2
    return 1/beta


def train():
    frozen=json.loads((HERE/'TRAINING_FREEZE.json').read_text())
    assert frozen['config']==CONFIG
    for name,digest in frozen['source_hashes'].items():assert sha(HERE/name)==digest
    assert not (HERE/'TEST_APPROVAL.json').exists()
    runtime(); assert torch.cuda.is_available()
    images,labels=arrays('train');section=splits(labels)
    output=HERE/'models';output.mkdir(exist_ok=False)
    np.savez_compressed(HERE/'splits.npz',**section)
    device='cuda';all_x=torch.as_tensor(images,device=device,dtype=torch.float32)/255
    all_y=torch.as_tensor(labels,device=device)
    train_ids=torch.as_tensor(section['train'],device=device)
    val_ids=torch.as_tensor(section['validation'],device=device)
    temp_ids=torch.as_tensor(section['temperature'],device=device)
    records=[];traces=[];temperatures=[];manifest={}
    write_json(HERE/'TRAINING_START.json',dict(utc=now(),freeze_sha256=sha(HERE/'TRAINING_FREEZE.json'),
        torch_version=torch.__version__,device=torch.cuda.get_device_name(0),cpu_threads=1,
        parameters=sum(p.numel() for p in CNN().parameters()),split_sha256=sha(HERE/'splits.npz')))
    for seed in CONFIG['seeds']:
        torch.manual_seed(CONFIG['init_base']+seed);torch.cuda.manual_seed_all(CONFIG['init_base']+seed)
        net=CNN().to(device);optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(CONFIG['order_base']+seed);best=float('inf');best_epoch=None
        torch.cuda.synchronize();start=time.perf_counter();seed_traces=[]
        try:
            for epoch in range(1,13):
                net.train();order=torch.as_tensor(rng.permutation(len(train_ids)),device=device)
                train_loss=0.
                for begin in range(0,len(order),256):
                    ids=train_ids[order[begin:begin+256]];optimizer.zero_grad(set_to_none=True)
                    z=net(all_x[ids]);loss=torch.nn.functional.cross_entropy(z,all_y[ids])
                    loss.backward();optimizer.step();train_loss+=float(loss.detach())*len(ids)
                row=dict(seed=seed,epoch=epoch,training_loss=train_loss/len(train_ids),validation_loss='',validation_error='')
                if epoch in CONFIG['validation_epochs']:
                    net.eval();z=predict_chunks(net,all_x[val_ids]);vl=float(torch.nn.functional.cross_entropy(z,all_y[val_ids]))
                    row['validation_loss']=vl;row['validation_error']=float((z.argmax(1)!=all_y[val_ids]).float().mean())
                    checkpoint=output/f'{seed}_epoch{epoch}.pt'
                    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},epoch=epoch,seed=seed),checkpoint)
                    manifest[checkpoint.name]=sha(checkpoint)
                    if vl<best:best=vl;best_epoch=epoch
                seed_traces.append(row);print('CNN',seed,'epoch',epoch,'train',round(row['training_loss'],5),'val',row['validation_loss'],flush=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            payload=torch.load(output/f'{seed}_epoch{best_epoch}.pt',map_location='cpu',weights_only=True)
            selected=output/f'{seed}.pt';torch.save(payload,selected);manifest[selected.name]=sha(selected)
            net.load_state_dict(payload['state_dict']);net.eval();temp_start=time.perf_counter()
            z=predict_chunks(net,all_x[temp_ids]).cpu().numpy();t=temperature_fit(z,labels[section['temperature']])
            temperature_seconds=time.perf_counter()-temp_start
            np.savez_compressed(output/f'{seed}_temperature.npz',logits=z,labels=labels[section['temperature']],temperature=t)
            manifest[f'{seed}_temperature.npz']=sha(output/f'{seed}_temperature.npz')
            temperatures.append(dict(seed=seed,temperature=t,forward_examples=5000,seconds=temperature_seconds))
            records.append(dict(seed=seed,best_epoch=best_epoch,validation_loss=best,training_seconds=elapsed,
                training_forward_backward_examples=480000,validation_forward_examples=20000,
                temperature_forward_examples=5000,temperature=t))
            traces.extend(seed_traces)
        except Exception as exc:
            torch.save(dict(state_dict=net.state_dict(),seed=seed,epoch=epoch),output/f'{seed}_FAILED.pt')
            write_json(output/f'{seed}_FAILED.json',dict(utc=now(),exception=repr(exc),trace=seed_traces,replaced=False))
            raise
    write_csv(HERE/'training.csv',records);write_csv(HERE/'training_trace.csv',traces)
    write_json(HERE/'temperatures.json',dict(utc=now(),rows=temperatures))
    write_json(HERE/'MODEL_COMMIT.json',dict(utc=now(),manifest=manifest,source_freeze_sha256=sha(HERE/'TRAINING_FREEZE.json'),
        split_sha256=sha(HERE/'splits.npz'),test_accessed=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train']);args=parser.parse_args()
    globals()[args.stage]()

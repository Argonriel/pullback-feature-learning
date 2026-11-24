"""Exactly flip-even, input-only selectors and counted CNN deployment."""
import numpy as np
import torch
import torch.nn.functional as F
RIDGE_ALPHAS = (1e-4, .01, 1.)
NEIGHBORS = (1, 5, 25)
HIST_BINS = (8, 16, 32)
NAMES = ('raw','full_fused','full_sequential','confidence','ridge','histogram','knn','oracle_jensen')


def features(x):
    partner=torch.flip(x,[-1])
    symmetric=(x+partner)*.5
    difference=torch.abs(x-partner)*.5
    return torch.cat((F.avg_pool2d(symmetric,4).flatten(1),F.avg_pool2d(difference,4).flatten(1)),dim=1)


def gap(first,second):
    return torch.clamp(.5*(torch.logsumexp(first,1)+torch.logsumexp(second,1))-torch.logsumexp(.5*(first+second),1),min=0.)


def top_quarter(score,ties):
    assert len(score)%4==0
    by_tie=torch.argsort(ties,stable=True)
    order=torch.argsort(score[by_tie],descending=True,stable=True)
    return by_tie[order[:len(score)//4]]


def fit_tables(fit_features,fit_gap):
    f=np.asarray(fit_features,dtype=np.float64);y=np.asarray(fit_gap,dtype=np.float64)
    mean=f.mean(0);std=np.maximum(f.std(0),1e-4);z=(f-mean)/std;target=y.mean()
    result=dict(mean=mean.astype('float32'),std=std.astype('float32'),fit_z=z.astype('float32'),fit_y=y.astype('float32'))
    for alpha in RIDGE_ALPHAS:
        coef=np.linalg.solve(z.T@z/len(z)+alpha*np.eye(z.shape[1]),z.T@(y-target)/len(z))
        result[f'ridge_{alpha:g}']=np.r_[coef,target].astype('float32')
    brightness=f[:,:49].mean(1);asymmetry=f[:,49:].mean(1)
    for bins in HIST_BINS:
        e0=np.quantile(brightness,np.arange(1,bins)/bins);e1=np.quantile(asymmetry,np.arange(1,bins)/bins)
        ids=np.searchsorted(e0,brightness)*bins+np.searchsorted(e1,asymmetry)
        count=np.bincount(ids,minlength=bins*bins)
        values=np.divide(np.bincount(ids,weights=y,minlength=bins*bins),count,out=np.full(bins*bins,target),where=count>0)
        result[f'hist_edges0_{bins}']=e0.astype('float32');result[f'hist_edges1_{bins}']=e1.astype('float32')
        result[f'hist_values_{bins}']=values.astype('float32');result[f'hist_counts_{bins}']=count
    return result


def device_tables(table,device):
    return {k:torch.as_tensor(v,device=device) for k,v in table.items()}


def score(x,table,name,choice):
    f=features(x)
    if name=='histogram':
        bins=int(choice['hist_bins'])
        brightness=f[:,:49].mean(1);asymmetry=f[:,49:].mean(1)
        ids=torch.bucketize(brightness,table[f'hist_edges0_{bins}'])*bins+torch.bucketize(asymmetry,table[f'hist_edges1_{bins}'])
        return table[f'hist_values_{bins}'][ids]
    z=(f-table['mean'])/table['std']
    if name=='ridge':
        coef=table[f"ridge_{choice['ridge_alpha']:g}"]
        return torch.clamp(z@coef[:-1]+coef[-1],min=0.)
    if name=='knn':
        # Exact nearest neighbors in the fixed standardized98-dimensional feature.
        ref=table['fit_z'];distance=(z*z).sum(1)[:,None]+(ref*ref).sum(1)[None,:]-2*z@ref.T
        indices=torch.topk(distance,int(choice['knn_k']),largest=False,sorted=False).indices
        return table['fit_y'][indices].mean(1)
    raise ValueError(name)


def deploy(net,x,ties,table,choice,temperature,name,calls=None):
    def predict(value):
        if calls is not None:calls.append(len(value))
        return net(value)/temperature
    n=len(x);indices=None
    if name=='raw':return predict(x),indices
    if name=='full_fused':
        both=predict(torch.cat((x,torch.flip(x,[-1])),0))
        return .5*(both[:n]+both[n:]),indices
    if name=='full_sequential':return .5*(predict(x)+predict(torch.flip(x,[-1]))),indices
    if name in ('ridge','histogram','knn'):
        indices=top_quarter(score(x,table,name,choice),ties)
        partner=torch.flip(torch.index_select(x,0,indices),[-1])
        both=predict(torch.cat((x,partner),0));out=both[:n].clone()
        out.index_copy_(0,indices,.5*(torch.index_select(out,0,indices)+both[n:]))
        return out,indices
    first=predict(x)
    if name=='confidence':
        confidence=torch.softmax(first,1).max(1).values
        indices=top_quarter(-confidence,ties)
        other=predict(torch.flip(torch.index_select(x,0,indices),[-1]))
    elif name=='oracle_jensen':
        other_all=predict(torch.flip(x,[-1]));indices=top_quarter(gap(first,other_all),ties)
        other=torch.index_select(other_all,0,indices)
    else:raise ValueError(name)
    out=first.clone();out.index_copy_(0,indices,.5*(torch.index_select(first,0,indices)+other))
    return out,indices


def expected_calls(name,n):
    if name=='raw':return [n]
    if name=='full_fused':return [2*n]
    if name in ('full_sequential','oracle_jensen'):return [n,n]
    if name=='confidence':return [n,n//4]
    return [n+n//4]

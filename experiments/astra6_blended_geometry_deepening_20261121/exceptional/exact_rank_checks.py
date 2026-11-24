"""Exact small-state checks of exceptional Lie ranks using Fraction arithmetic.
Independent coordinate Lyapunov solve; no spectral solve or symbolic dependency.
"""
from fractions import Fraction as Q
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np


def matrix(rows):
    return np.array([[Q(str(v)) for v in r] for r in rows], dtype=object)


def inverse(a):
    n=len(a)
    aug=[[Q(a[i,j]) for j in range(n)]+[Q(int(i==j)) for j in range(n)] for i in range(n)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if aug[i][j])
        aug[j],aug[pivot]=aug[pivot],aug[j]
        z=aug[j][j]
        aug[j]=[v/z for v in aug[j]]
        for i in range(n):
            if i!=j:
                z=aug[i][j]
                aug[i]=[v-z*w for v,w in zip(aug[i],aug[j])]
    return np.array([r[n:] for r in aug],dtype=object)


def rank(a):
    aa=[list(r) for r in a]
    n=len(aa); m=len(aa[0]); p=0
    for j in range(m):
        hit=next((i for i in range(p,n) if aa[i][j]),None)
        if hit is None: continue
        aa[p],aa[hit]=aa[hit],aa[p]
        z=aa[p][j]; aa[p]=[v/z for v in aa[p]]
        for i in range(p+1,n):
            z=aa[i][j]; aa[i]=[v-z*w for v,w in zip(aa[i],aa[p])]
        p+=1
        if p==n: break
    return p


class Geometry:
    def __init__(self,x,lam):
        self.x=x; self.lam=Q(lam); self.n=len(x)
        self.pairs=[(i,i) for i in range(self.n)]+[(i,j) for i in range(self.n) for j in range(i+1,self.n)]
        self.d=len(self.pairs)
        self.basis=[]
        for i,j in self.pairs:
            b=np.full_like(x,Q(0)); b[i,j]=b[j,i]=Q(1); self.basis.append(b)
        k=np.array([self.vec(x@b+b@x+self.lam*b) for b in self.basis],dtype=object).T
        self.ri=inverse(k)
        self.fields=np.array([self.basis[i]-self.lam*self.R(self.basis[i]) for i in range(self.n)],dtype=object)
        self.a=np.array([np.diag(f) for f in self.fields],dtype=object).T
        self.c=inverse(self.a)
        self.h=self.combine(self.fields,self.c)
    def vec(self,u): return np.array([u[i,j] for i,j in self.pairs],dtype=object)
    def mat(self,v):
        out=np.full_like(self.x,Q(0))
        for z,(i,j) in zip(v,self.pairs): out[i,j]=out[j,i]=z
        return out
    def R(self,u): return self.mat(self.ri@self.vec(u))
    def dm(self,v,u):
        w=self.R(u); return self.lam*self.R(v@w+w@v)
    def d2m(self,v,w,u):
        a=self.R(u)
        b=self.R(v@a+a@v); c=self.R(w@a+a@w)
        return -self.lam*self.R(w@b+b@w+v@c+c@v)
    def combine(self,fields,coeff):
        return np.array([sum((fields[k]*coeff[k,i] for k in range(self.n)),np.full_like(self.x,Q(0))) for i in range(self.n)],dtype=object)
    def df(self,v): return np.array([self.dm(v,self.basis[i]) for i in range(self.n)],dtype=object)
    def dh(self,v):
        df=self.df(v); da=np.array([np.diag(f) for f in df],dtype=object).T
        return self.combine(df,self.c)-self.combine(self.h,da@self.c)
    def d2h(self,v,w):
        d2f=np.array([self.d2m(v,w,self.basis[i]) for i in range(self.n)],dtype=object)
        d2a=np.array([np.diag(f) for f in d2f],dtype=object).T
        dav=np.array([np.diag(f) for f in self.df(v)],dtype=object).T
        daw=np.array([np.diag(f) for f in self.df(w)],dtype=object).T
        return self.combine(d2f,self.c)-self.combine(self.dh(v),daw@self.c)-self.combine(self.dh(w),dav@self.c)-self.combine(self.h,d2a@self.c)
    def bracket(self,i,j): return self.dh(self.h[i])[j]-self.dh(self.h[j])[i]
    def dbracket(self,i,j,v):
        dhv=self.dh(v)
        return self.d2h(v,self.h[i])[j]+self.dh(dhv[i])[j]-self.d2h(v,self.h[j])[i]-self.dh(dhv[j])[i]
    def nested(self,k,i,j):
        b=self.bracket(i,j)
        return self.dbracket(i,j,self.h[k])-self.dh(b)[k]
    def ranks(self):
        pairs=[(i,j) for i in range(self.n) for j in range(i+1,self.n)]
        bs=[self.bracket(i,j) for i,j in pairs]
        ns=[self.nested(k,i,j) for i,j in pairs for k in range(self.n)]
        return [rank(np.array([self.vec(v) for v in vs],dtype=object).T) for vs in (list(self.h),list(self.h)+bs,list(self.h)+bs+ns)],bs,ns


def rational(v):
    if isinstance(v,np.ndarray): return [rational(z) for z in v]
    return str(v)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for label,x,lam in [
        ('two_equal_diagonals',matrix([[2,'1/3'],['1/3',2]]),Q(3)),
        ('two_distinct_diagonals',matrix([[2,'1/3'],['1/3',3]]),Q(3)),
        ('two_diagonal',matrix([[2,0],[0,3]]),Q(3)),
        ('three_equicorrelated',matrix([[2,'1/3','1/3'],['1/3',2,'1/3'],['1/3','1/3',2]]),Q(3)),
        ('three_connected_path',matrix([[2,'1/4',0],['1/4',3,'1/5'],[0,'1/5',4]]),Q(1)),
        ('three_block_pair_single',matrix([[2,'1/3',0],['1/3',2,0],[0,0,4]]),Q(3)),
    ]:
        g=Geometry(x,lam); rs,bs,ns=g.ranks()
        row={'label':label,'x':rational(x),'lambda':str(lam),'ranks_through_length_1_2_3':rs,'first_brackets':rational(np.array(bs)),'third_brackets':rational(np.array(ns))}
        if label=='two_equal_diagonals':
            s=x[0,0]; c=x[0,1]; d=s*s-c*c; qq=2*s*d+lam*s*s
            predicted=-lam*c*d/(2*qq*qq)
            row['third_bracket_prediction']=str(predicted)
            row['prediction_exact_match']=ns[0][0,1]==predicted
            assert row['prediction_exact_match'] and rs==[2,2,3]
        if label=='two_distinct_diagonals': assert rs==[2,3,3]
        if label=='two_diagonal': assert rs==[2,2,2]
        if label=='three_connected_path':
            row['missing_edge_field_F2_13']=str(g.fields[1][0,2])
            assert g.fields[1][0,2]!=0
        if label=='three_block_pair_single':
            assert rs==[3,3,4]
            assert all(v[0,2]==0 and v[1,2]==0 for v in list(g.h)+bs+ns)
        rows.append(row)
    payload={'arithmetic':'exact fractions','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'numpy':np.__version__,'rows':rows}
    (args.output/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps([{'label':r['label'],'ranks':r['ranks_through_length_1_2_3'],'missing_edge':r.get('missing_edge_field_F2_13')} for r in rows],indent=2))
if __name__=='__main__':main()

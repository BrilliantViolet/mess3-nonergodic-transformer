import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import json,os
CONFIGS=[{"x":0.05,"alpha":0.85},{"x":0.05,"alpha":0.60},{"x":0.15,"alpha":0.85},{"x":0.15,"alpha":0.60}]
class Mess3:
    def __init__(self,x,alpha):
        self.x=x;self.alpha=alpha
        T=np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                pt=(1-3*x) if i==j else x
                for t in range(3):
                    T[t,i,j]=pt*(alpha if t==j else (1-alpha)/2)
        self.T=T;self.Tf=T.sum(0)
        ev,evec=linalg.eig(self.Tf.T)
        s=np.real(evec[:,np.argmax(np.real(ev))]);self.stat=s/s.sum()
    def gen(self,L,rng):
        tok=np.zeros(L,dtype=np.int64);st=rng.choice(3,p=self.stat)
        for t in range(L):
            p=self.T[:,st,:].flatten();p/=p.sum();c=rng.choice(9,p=p);tok[t]=c//3;st=c%3
        return tok
    def belief(self,seq):
        B=np.zeros((len(seq),3));b=self.stat.copy()
        for i,tk in enumerate(seq):
            nb=b@self.T[tk];nb/=nb.sum();B[i]=nb;b=nb
        return B
procs=[Mess3(**c) for c in CONFIGS]
rng=np.random.default_rng(42)
N=5000;L=16;seqs=[];cids=[]
for _ in range(N):
    c=int(rng.integers(4));seqs.append(procs[c].gen(L,rng));cids.append(c)
seqs=np.array(seqs,dtype=np.int64);cids=np.array(cids)
print("Dataset ok",seqs.shape)
class TF(nn.Module):
    def __init__(self):
        super().__init__()
        self.te=nn.Embedding(3,64);self.pe=nn.Embedding(15,64)
        el=nn.TransformerEncoderLayer(d_model=64,nhead=2,batch_first=True,dim_feedforward=128)
        self.tr=nn.TransformerEncoder(el,num_layers=2)
        self.ln=nn.LayerNorm(64);self.out=nn.Linear(64,3,bias=False)
        self.acts=None
    def forward(self,x):
        B,T=x.shape;h=self.te(x)+self.pe(torch.arange(T,device=x.device))
        mask=torch.triu(torch.ones(T,T,device=x.device),1).bool()
        h=self.tr(h,mask=mask);h=self.ln(h);self.acts=h.detach();return self.out(h)
device="cuda" if torch.cuda.is_available() else "cpu"
model=TF().to(device)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
xs=torch.tensor(seqs[:,:-1]);ys=torch.tensor(seqs[:,1:])
print("Training on",device)
for ep in range(20):
    idx=torch.randperm(N);tl=0;nb=0
    for i in range(0,N,256):
        b=idx[i:i+256];xb=xs[b].to(device);yb=ys[b].to(device)
        loss=F.cross_entropy(model(xb).reshape(-1,3),yb.reshape(-1))
        opt.zero_grad();loss.backward();opt.step();tl+=loss.item();nb+=1
    if (ep+1)%5==0:print(f"Ep{ep+1} loss={tl/nb:.4f}")
model.eval()
acts=[];comp=[];bels=[]
with torch.no_grad():
    for i in range(min(1000,N)):
        x=xs[i:i+1].to(device);_=model(x)
        acts.append(model.acts[0,-1].cpu().numpy())
        comp.append(cids[i])
        bels.append(procs[cids[i]].belief(seqs[i,:-1])[-1])
acts=np.array(acts);comp=np.array(comp);bels=np.array(bels)
Xb=np.hstack([acts,np.ones((len(acts),1))])
W,_,_,_=np.linalg.lstsq(Xb,bels,rcond=None);pred=Xb@W
r2=1-((bels-pred)**2).sum()/((bels-bels.mean(0))**2).sum()
_,S,_=np.linalg.svd(acts-acts.mean(0),full_matrices=False)
ev2=(S**2)/(S**2).sum();dim95=int(np.argmax(np.cumsum(ev2)>=0.95))+1
print(f"R2={r2:.4f} dim95={dim95}")
for i in range(4):
    m=comp==i
    if m.sum()>5:
        r2i=1-((bels[m]-pred[m])**2).sum()/((bels[m]-bels[m].mean(0))**2).sum()
        print(f"  C{i}: R2={r2i:.4f} n={m.sum()}")
os.makedirs("/workspace/outputs",exist_ok=True)
json.dump({"r2":float(r2),"dim95":dim95},open("/workspace/outputs/results.json","w"))
print("Done")

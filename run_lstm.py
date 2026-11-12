import argparse,collections,hashlib,json,random,re,time,platform
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent; out=root/('smoke' if a.smoke else 'results');out.mkdir(exist_ok=True)
seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
d=pd.read_csv(root/'data/train.csv'); test=pd.read_csv(root/'data/test.csv')
idx=np.arange(len(d));tr,va=train_test_split(idx,test_size=10000,stratify=d.label,random_state=seed)
if a.smoke:tr=tr[:2048];va=va[:512]
train=d.iloc[tr];valid=d.iloc[va]
def tok(s):return re.findall(r'\w+|[^\w\s]',str(s).replace('I','ı').replace('İ','i').lower())[:128]
counts=collections.Counter(w for s in train.comment for w in tok(s));vocab={w:i+2 for i,(w,n) in enumerate(counts.most_common(30000)) if n>=2}
def encode(frame):return [(torch.tensor([vocab.get(w,1) for w in tok(s)] or [1]),int(y)) for s,y in zip(frame.comment,frame.label)]
def collate(b):
 xs,ys=zip(*b);lens=torch.tensor([len(x) for x in xs]);return pad_sequence(xs,batch_first=True),lens,torch.tensor(ys)
def loader(frame,shuffle=False):return torch.utils.data.DataLoader(encode(frame),batch_size=128,shuffle=shuffle,collate_fn=collate)
trainloader=loader(train,True);validloader=loader(valid)
class LSTM(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(max(vocab.values())+1,64,padding_idx=0);self.lstm=nn.LSTM(64,64,batch_first=True);self.drop=nn.Dropout(.3);self.head=nn.Linear(64,3)
 def forward(self,x,l):
  _,(h,_)=self.lstm(pack_padded_sequence(self.emb(x),l.cpu(),batch_first=True,enforce_sorted=False));return self.head(self.drop(h[-1]))
model=LSTM();opt=torch.optim.Adam(model.parameters(),lr=.001);lossfn=nn.CrossEntropyLoss()
def evaluate(dl):
 model.eval();yy=[];pp=[];prob=[];total=0
 with torch.no_grad():
  for x,l,y in dl:
   z=model(x,l);total+=lossfn(z,y).item()*len(y);yy.extend(y.tolist());pp.extend(z.argmax(1).tolist());prob.extend(z.softmax(1).tolist())
 return total/len(yy),yy,pp,prob
config={'seed':seed,'train_n':len(train),'validation_n':len(valid),'test_n':len(test),'vocab_size':len(vocab),'embedding_dim':64,'hidden_dim':64,'max_tokens':128,'batch_size':128,'optimizer':'Adam','lr':.001,'dropout':.3,'max_epochs':10,'early_stopping_patience':2,'selection':'minimum validation cross-entropy','dataset_revision':'b79f242406e89b63d3cdb39db5fc4e99442a6932','torch':torch.__version__,'label_semantics':'numeric source labels; semantic mapping not established','data_policy':'source train/test preserved; validation stratified from train; no deduplication'}
(out/'config.json').write_text(json.dumps(config,indent=2));pd.DataFrame({'source_train_row':np.r_[tr,va],'partition':['train']*len(tr)+['validation']*len(va)}).to_csv(out/'split.csv',index=False)
start=time.time();history=[];best=float('inf');stale=0
for epoch in range(1,(1 if a.smoke else 10)+1):
 model.train();total=0
 for x,l,y in trainloader:
  opt.zero_grad();z=model(x,l);loss=lossfn(z,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(y)
 vl,yy,pp,_=evaluate(validloader);row={'epoch':epoch,'train_loss':total/len(train),'validation_loss':vl,'validation_accuracy':accuracy_score(yy,pp),'elapsed_seconds':time.time()-start};history.append(row);print(json.dumps(row),flush=True);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
 if vl<best:best=vl;stale=0;torch.save(model.state_dict(),out/'model.pt')
 else:stale+=1
 if stale>=2:break
if a.smoke:print('SMOKE_OK',flush=True);raise SystemExit
model.load_state_dict(torch.load(out/'model.pt',weights_only=True));_,yy,pp,probs=evaluate(loader(test))
metrics={'accuracy':accuracy_score(yy,pp),'classification_report':classification_report(yy,pp,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yy,pp,labels=[0,1,2]).tolist(),'elapsed_seconds':time.time()-start}
for avg in ['macro','weighted']:
 pr,rc,f1,_=precision_recall_fscore_support(yy,pp,average=avg,zero_division=0);metrics[avg]={'precision':pr,'recall':rc,'f1':f1}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame({'source_test_row':np.arange(len(test)),'true_label':yy,'predicted_label':pp,'p0':np.array(probs)[:,0],'p1':np.array(probs)[:,1],'p2':np.array(probs)[:,2]}).to_csv(out/'test_predictions.csv',index=False)
(out/'vocabulary.json').write_text(json.dumps(vocab,ensure_ascii=False));print('FINAL',json.dumps(metrics),flush=True)

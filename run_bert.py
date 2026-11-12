
import os,argparse,json,time,random
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_recall_fscore_support
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='auto');p.add_argument('--checkpoint-every',type=int,default=20);p.add_argument('--stop-after-batches',type=int);p.add_argument('--output');a=p.parse_args()
r=Path(__file__).resolve().parent;o=Path(a.output) if a.output else r/('bert_smoke' if a.smoke else 'bert_results');o.mkdir(exist_ok=True)
print(f'Training checkpoint interval: {a.checkpoint_every} batches.',flush=True)
required=[r/'data/train.csv',r/'data/test.csv',r/'results/split.csv']
missing=[str(p) for p in required if not p.is_file()]
if missing:raise FileNotFoundError('Missing package files: '+', '.join(missing))
print('Input files found. Loading BERT model...',flush=True)
seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
name='dbmdz/bert-base-turkish-uncased';rev='6cb8cd880acc6f7d9723161b573fce0dfd23b39b'
tokenizer=AutoTokenizer.from_pretrained(name,revision=rev)
model=AutoModelForSequenceClassification.from_pretrained(name,revision=rev,num_labels=3).to(device)
d=pd.read_csv(r/'data/train.csv');test=pd.read_csv(r/'data/test.csv');split=pd.read_csv(r/'results/split.csv');tr=split.loc[split.partition=='train','source_train_row'].to_numpy();va=split.loc[split.partition=='validation','source_train_row'].to_numpy()
train=d.iloc[tr[:64] if a.smoke else tr];valid=d.iloc[va[:32] if a.smoke else va]
class Texts(torch.utils.data.Dataset):
 def __init__(self,f):self.enc=tokenizer(f.comment.tolist(),truncation=True,max_length=128,padding=False);self.y=f.label.tolist()
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return {k:v[i] for k,v in self.enc.items()}|{'labels':self.y[i]}
def collate(batch):return tokenizer.pad(batch,padding=True,return_tensors='pt')
def loader(f,shuffle=False):return torch.utils.data.DataLoader(Texts(f),batch_size=16,shuffle=shuffle,collate_fn=collate)
print('Tokenizing training and validation data...',flush=True)
tl=loader(train,True);vl=loader(valid);print('Data ready. Starting training or resuming checkpoint.',flush=True);opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0)
epochs=1 if a.smoke else 4;accum=2;steps=(len(tl)+1)//2*epochs;sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=0,num_training_steps=steps)
config={'base_model':name,'revision':rev,'seed':seed,'train_n':len(train),'validation_n':len(valid),'test_n':len(test),'max_tokens':128,'epochs':epochs,'microbatch':16,'gradient_accumulation':2,'effective_batch_size':32,'optimizer':'AdamW','lr':5e-5,'weight_decay':0,'early_stopping_patience':2,'selection':'minimum validation cross-entropy','device':device,'same_split_as':'results/split.csv','fresh_fine_tuning':True}
(o/'config.json').write_text(json.dumps(config,indent=2));history=[];best=float('inf');stale=0;t0=time.time();start_epoch=1;start_batch=0;resume_loss=0;elapsed_before=0
resume_path=o/'resume.pt'
if resume_path.exists():
 state=torch.load(resume_path,map_location=device,weights_only=False)
 if state['config']!=config:raise ValueError('Checkpoint configuration differs from current experiment')
 model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
 start_epoch=state['epoch'];start_batch=state['next_batch'];resume_loss=state['loss'];history=state['history'];best=state['best'];stale=state['stale'];elapsed_before=state['elapsed_seconds']
 torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
 if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
 del state
 print(json.dumps({'event':'RESUMED','epoch':start_epoch,'next_batch':start_batch}),flush=True)
def save_resume(epoch,next_batch,loss):
 state={'config':config,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'next_batch':next_batch,'loss':loss,'history':history,'best':best,'stale':stale,'elapsed_seconds':elapsed_before+time.time()-t0,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None}
 tmp=o/'resume.tmp';torch.save(state,tmp);os.replace(tmp,resume_path)
 (o/'progress.json').write_text(json.dumps({'epoch':epoch,'next_batch':next_batch,'elapsed_seconds':state['elapsed_seconds'],'optimizer_updates':sched.last_epoch},indent=2))
 print(json.dumps({'event':'CHECKPOINT_SAVED','epoch':epoch,'next_batch':next_batch}),flush=True)
def evaluate(dl):
 model.eval();yy=[];pp=[];probs=[];loss=0
 with torch.no_grad():
  for b in dl:
   b={k:v.to(device) for k,v in b.items()};z=model(**b);loss+=z.loss.item()*len(b['labels']);yy.extend(b['labels'].cpu().tolist());pp.extend(z.logits.argmax(1).cpu().tolist());probs.extend(z.logits.softmax(1).cpu().tolist())
 return loss/len(yy),yy,pp,probs
for epoch in range(start_epoch,epochs+1):
 if stale>=2:break
 order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
 first=start_batch if epoch==start_epoch else 0
 epoch_loader=torch.utils.data.DataLoader(tl.dataset,batch_size=16,sampler=order[first*16:],collate_fn=collate,generator=torch.Generator().manual_seed(seed+epoch))
 model.train();opt.zero_grad();loss=resume_loss if epoch==start_epoch else 0
 for i,b in enumerate(epoch_loader,start=first):
  b={k:v.to(device) for k,v in b.items()};z=model(**b);loss+=z.loss.item()*len(b['labels']);(z.loss/accum).backward()
  if (i+1)%accum==0 or i+1==len(tl):torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step();opt.zero_grad()
  if (i+1)%10==0:print(json.dumps({'epoch':epoch,'batch':i+1,'total_batches':len(tl),'elapsed_seconds':elapsed_before+time.time()-t0}),flush=True)
  stopping=a.stop_after_batches is not None and i+1>=a.stop_after_batches
  if ((i+1)%a.checkpoint_every==0 or stopping) and ((i+1)%accum==0 or i+1==len(tl)):
   save_resume(epoch,i+1,loss)
   if stopping:print('REQUESTED_STOP',flush=True);raise SystemExit
 vloss,yy,pp,_=evaluate(vl);row={'epoch':epoch,'train_loss':loss/len(train),'validation_loss':vloss,'validation_accuracy':accuracy_score(yy,pp),'elapsed_seconds':time.time()-t0};history.append(row);pd.DataFrame(history).to_csv(o/'history.csv',index=False);print(json.dumps(row),flush=True)
 if vloss<best:
  best=vloss;stale=0;model.save_pretrained(o/'checkpoint');tokenizer.save_pretrained(o/'checkpoint')
 else:stale+=1
 save_resume(epoch+1,0,0)
 if stale>=2:break
if a.smoke:print('SMOKE_OK',flush=True);raise SystemExit
model=AutoModelForSequenceClassification.from_pretrained(o/'checkpoint').to(device);_,yy,pp,probs=evaluate(loader(test));m={'accuracy':accuracy_score(yy,pp),'classification_report':classification_report(yy,pp,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yy,pp,labels=[0,1,2]).tolist(),'elapsed_seconds':time.time()-t0}
for avg in ['macro','weighted']:
 pr,rc,f1,_=precision_recall_fscore_support(yy,pp,average=avg,zero_division=0);m[avg]={'precision':pr,'recall':rc,'f1':f1}
(o/'metrics.json').write_text(json.dumps(m,indent=2));pd.DataFrame({'source_test_row':np.arange(len(test)),'true_label':yy,'predicted_label':pp,'p0':np.array(probs)[:,0],'p1':np.array(probs)[:,1],'p2':np.array(probs)[:,2]}).to_csv(o/'test_predictions.csv',index=False);print('FINAL',json.dumps(m),flush=True)

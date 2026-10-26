import os,time,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
os.environ['HF_HUB_DISABLE_XET']='1'
torch.set_num_threads(4);torch.manual_seed(42)
name='dbmdz/bert-base-turkish-uncased';rev='6cb8cd880acc6f7d9723161b573fce0dfd23b39b'
print('LOADING',flush=True)
tokenizer=AutoTokenizer.from_pretrained(name,revision=rev)
model=AutoModelForSequenceClassification.from_pretrained(name,revision=rev,num_labels=3)
print('LOADED',flush=True)
import pandas as pd
s=pd.read_csv(Path(__file__).parent/'data/train.csv').comment.iloc[5000:5016].tolist()
x=tokenizer(s,truncation=True,max_length=128,padding=True,return_tensors='pt');y=torch.tensor([0,1,2,0]*4)
opt=torch.optim.AdamW(model.parameters(),lr=5e-5)
times=[]
for k in range(3):
 t=time.time();opt.zero_grad();z=model(**x,labels=y);z.loss.backward();opt.step();dt=time.time()-t;times.append(dt);print('BENCHMARK_STEP',k,dt,'tokens',x['input_ids'].shape,flush=True)
r={'batch_size':16,'padded_tokens':x['input_ids'].shape[1],'step_seconds':times,'estimated_four_epoch_training_hours':sum(times[1:])/2*50976/16*4/3600,'revision':rev}
(Path(__file__).parent/'bert_benchmark.json').write_text(json.dumps(r,indent=2));print('BENCHMARK',r,flush=True)

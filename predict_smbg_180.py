
import argparse,csv,hashlib,json,sys,time
from pathlib import Path

LABELS={0:'neutral',1:'positive',2:'negative'}
def main():
 root=Path(__file__).resolve().parent
 parser=argparse.ArgumentParser()
 parser.add_argument('--model',type=Path,default=root/'bert_results/checkpoint')
 parser.add_argument('--input',type=Path,default=root/'smbg_180_reviews.csv')
 parser.add_argument('--output',type=Path,default=root/'smbg_180_results')
 parser.add_argument('--batch-size',type=int,default=16)
 parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 parser.add_argument('--check-input',action='store_true')
 args=parser.parse_args()
 if args.batch_size<1:parser.error('batch-size must be positive')
 with args.input.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=180:raise ValueError(f'Expected 180 reviews, found {len(rows)}')
 ids=[int(r['review_id']) for r in rows]
 if sorted(ids)!=list(range(1,181)):raise ValueError('Expected unique review IDs 1..180')
 if any(not r['text'].strip() for r in rows):raise ValueError('Empty review text')
 print('SMBG180_V1: 180 reviews found. No training will be performed.',flush=True)
 if args.check_input:print('INPUT_CHECK_OK');return
 modeldir=args.model.resolve()
 if not (modeldir/'config.json').is_file():raise FileNotFoundError(f'Best checkpoint not found: {modeldir}. Copy the code and CSV beside run_bert.py, or specify --model.')
 if not any((modeldir/n).exists() for n in ['model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json']):raise FileNotFoundError('Checkpoint weights missing')
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.set_num_threads(4)
 device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
 print(f'Loading LOCAL best checkpoint: {modeldir}; device={device}',flush=True)
 tokenizer=AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(str(modeldir),local_files_only=True).to(device)
 if model.config.num_labels!=3:raise ValueError('Expected a three-class model')
 model.eval();outrows=[];start=time.time();lengths=[]
 with torch.inference_mode():
  for i in range(0,len(rows),args.batch_size):
   batch=rows[i:i+args.batch_size];texts=[r['text'] for r in batch]
   lengths.extend(len(x) for x in tokenizer(texts,truncation=False,padding=False)['input_ids'])
   enc=tokenizer(texts,truncation=True,max_length=128,padding=True,return_tensors='pt')
   logits=model(**{k:v.to(device) for k,v in enc.items()}).logits
   probs=logits.softmax(-1).cpu().tolist()
   for r,p in zip(batch,probs):
    cls=max(range(3),key=lambda k:p[k]);old=r.get('archived_sentiment','')
    outrows.append({'review_id':int(r['review_id']),'text':r['text'],'predicted_class_id':cls,'predicted_sentiment':LABELS[cls],'p_class_0':p[0],'p_class_1':p[1],'p_class_2':p[2],'p_neutral':p[0],'p_positive':p[1],'p_negative':p[2],'max_probability':p[cls],'archived_sentiment':old,'changed_from_archive':LABELS[cls]!=old if old else ''})
   print(f'Predicted {min(i+args.batch_size,180)}/180 reviews',flush=True)
 for x in outrows:
  assert abs(x['p_class_0']+x['p_class_1']+x['p_class_2']-1)<1e-5
 args.output.mkdir(parents=True,exist_ok=True)
 target=args.output/'smbg_180_predictions.csv'
 with target.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
 summary={'n':180,'counts':{label:sum(r['predicted_sentiment']==label for r in outrows) for label in LABELS.values()},'changed_from_archived_count':sum(r['changed_from_archive'] is True for r in outrows),'truncated_reviews':sum(n>128 for n in lengths),'max_tokens':128,'label_mapping':LABELS,'label_mapping_basis':'Source training dataset class examples; checkpoint numeric IDs are retained. Semantic mapping is not specified by generic LABEL_0/1/2 metadata.','human_validation_performed':False}
 (args.output/'smbg_180_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 config={'model_directory':str(modeldir),'model_config':model.config.to_dict(),'input_file':str(args.input.resolve()),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'weights_sha256':{},'batch_size':args.batch_size,'device':device,'torch_version':torch.__version__,'max_tokens':128,'text_policy':'No additional cleaning; archived processed review text preserved','elapsed_seconds':time.time()-start}
 for path in sorted(modeldir.glob('*')):
  if path.suffix in ['.safetensors','.bin']:
   h=hashlib.sha256()
   with path.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
   config['weights_sha256'][path.name]=h.hexdigest()
 (args.output/'smbg_180_run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 print(f'Results saved to {args.output}',flush=True)
if __name__=='__main__':main()

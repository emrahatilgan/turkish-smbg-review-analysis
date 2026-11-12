from pathlib import Path
import json, hashlib, shutil, argparse, re
import numpy as np
import pandas as pd
import openpyxl
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,cohen_kappa_score
p=argparse.ArgumentParser();p.add_argument('--input',default=str(Path(__file__).resolve().parent/'study_data'));p.add_argument('--d2',default='d2.xlsx');p.add_argument('--output',default='output/SMBG_Final_Analysis');a=p.parse_args();src=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(src/'smbg_180_predictions.csv');assert len(d)==180 and set(d.review_id)==set(range(1,181))
probs=d[['p_neutral','p_positive','p_negative']].to_numpy();assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all();assert np.allclose(probs.sum(1),1,atol=1e-6);assert np.array_equal(probs.argmax(1),d.predicted_class_id)
labels=['positive','negative','neutral'];mapping={'Pozitif':'positive','Negatif':'negative','Nötr':'neutral','Belirsiz':'uncertain','Var':'present','Yok':'absent'}
for name,file in [('d1','d1.xlsx'),('d2',a.d2)]:
 rows=list(openpyxl.load_workbook(src/file,data_only=True)['Etiketleme'].values)[5:185]
 r=pd.DataFrame([[int(x[0]),str(x[1]),mapping[x[2]],mapping[x[3]]] for x in rows],columns=['review_id',name+'_text',name+'_sentiment',name+'_safety'])
 d=d.merge(r,on='review_id',validate='one_to_one');assert all(re.sub(r'[^\w]', '', x)==re.sub(r'[^\w]', '', y) for x,y in zip(d.text,d[name+'_text']))
res={'n':180,'prediction_counts':d.predicted_sentiment.value_counts().to_dict(),'human_agreement':{},'model_vs_rater':{},'limitations':['No consensus labels supplied; compare separately with each independent rater.','Negative sentiment probability is not calibrated clinical safety probability.','Threshold results are exploratory on these 180 reviews, not independently validated threshold selection.','Binary safety agreement does not establish agreement on individual safety themes.']}
for field in ['sentiment','safety']:
 x=d['d1_'+field];y=d['d2_'+field];res['human_agreement'][field]={'n':180,'agreement_count':int((x==y).sum()),'agreement':float((x==y).mean()),'kappa':cohen_kappa_score(x,y),'uncertain_included':True,'disagreement_ids':d.loc[x!=y,'review_id'].tolist()}
for name in ['d1','d2']:
 sub=d[d[name+'_sentiment']!='uncertain'];y=sub[name+'_sentiment'];pred=sub.predicted_sentiment
 res['model_vs_rater'][name]={'n':len(sub),'excluded_ids':d.loc[d[name+'_sentiment']=='uncertain','review_id'].tolist(),'accuracy':accuracy_score(y,pred),'kappa':cohen_kappa_score(y,pred),'classification_report':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'matrix_order':labels}
sub=d[(d.d1_safety!='uncertain')&(d.d2_safety!='uncertain')].copy();assert (sub.d1_safety==sub.d2_safety).all();gold=sub.d1_safety.eq('present');screen=sub.predicted_sentiment.eq('negative')
def binary(s):
 tp=int((s&gold).sum());fp=int((s&~gold).sum());fn=int((~s&gold).sum());tn=int((~s&~gold).sum());return {'screened':int(s.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'recall':tp/(tp+fn),'precision':tp/(tp+fp) if tp+fp else 0,'specificity':tn/(tn+fp)}
res['negative_argmax_safety_screen']={'n':len(sub),**binary(screen),'safety_ids':sub.loc[gold,'review_id'].tolist(),'missed_safety_ids':sub.loc[gold&~screen,'review_id'].tolist()}
thresholds=[]
for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:thresholds.append({'threshold':t,**binary(sub.p_negative>=t)})
pd.DataFrame(thresholds).to_csv(out/'threshold_sensitivity.csv',index=False)
d.to_csv(out/'joined_review_results.csv',index=False);d[d.d1_safety=='present'].to_csv(out/'human_safety_reviews.csv',index=False)
json.dump(res,open(out/'analysis_results.json','w'),ensure_ascii=False,indent=2)
manifest={f:hashlib.sha256((src/f).read_bytes()).hexdigest() for f in ['smbg_180_predictions.csv','d1.xlsx',a.d2]};json.dump(manifest,open(out/'source_hashes.json','w'),indent=2)
inputs=out/'inputs';inputs.mkdir(exist_ok=True)
for f in manifest:shutil.copy2(src/f,inputs/f)
print(json.dumps(res,ensure_ascii=False,indent=2));print(pd.DataFrame(thresholds).to_string(index=False))

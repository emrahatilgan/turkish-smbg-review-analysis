from pathlib import Path
import csv,hashlib,json
r=Path(__file__).resolve().parent
expected=json.loads((r/'data_hashes.json').read_text())
for name,n in [('train.csv',60976),('test.csv',15000)]:
 p=r/'data'/name
 assert p.is_file(),f'Missing: {p}'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[name],f'Hash mismatch: {name}'
 with p.open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['comment','label'];rows=list(reader)
 assert len(rows)==n,(name,len(rows))
 print(name,len(rows),'OK')
with (r/'results/split.csv').open(newline='') as f:split=list(csv.DictReader(f))
assert len(split)==60976
assert sum(x['partition']=='train' for x in split)==50976
assert sum(x['partition']=='validation' for x in split)==10000
assert len(set(int(x['source_train_row']) for x in split))==60976
s=(r/'run_bert.py').read_text()
assert 'def save_resume' in s and 'SMBG_ALL_MODELS_V2' in s
print('Split and resumable code OK. ALL REQUIRED INPUT FILES PRESENT.')

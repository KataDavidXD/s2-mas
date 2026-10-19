"""Summarize the independent synthetic companion and its financial structure.

Usage: python synthetic_persona_statistics.py --input-dir PATH --output-dir PATH
Reads existing persona shards; writes aggregate tables, never individual rows.
"""
from pathlib import Path
import argparse, csv, gzip, json, time
import numpy as np

FEATURES=['Household size','Monthly income','Essential spending','Adjustable spending',
          'Opening cash','Other assets','Opening debt','Monthly debt payment']

def table(path, fields, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    info=json.loads((args.input_dir/'generation_info.json').read_text(encoding='utf-8'))
    n=info['independent_synthetic_households'];x=np.empty((n,8),dtype=np.float64)
    offset=0;t=time.perf_counter()
    for shard in info['shards']:
        start=offset
        with gzip.open(args.input_dir/shard['file'],'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                assert r['household_id']==f'new-persona-{offset:09d}'
                assert r['record_type']=='independent_synthetic_household'
                d=r['opening_debt']
                x[offset]=[r['household_size'],r['monthly_income_cents']/100,
                   r['essential_spending_cents']/100,r['adjustable_spending_cents']/100,
                   r['opening_cash_cents']/100,r['opening_asset_cents']/100,
                   d['principal_cents']/100,d['monthly_payment_cents']/100]
                offset+=1
        assert offset-start==shard['rows']
        print(json.dumps({'rows':offset,'elapsed_seconds':round(time.perf_counter()-t,2)}),flush=True)
    assert offset==n and np.isfinite(x).all() and (x>=0).all() and (x[:,1]>0).all()
    distributions=[]
    def hist(metric, values, edges, labels):
        counts=np.histogram(values,bins=edges)[0]
        assert int(counts.sum())==n
        distributions.extend({'metric':metric,'category':label,'count':int(c),
           'share_pct':float(c/n*100)} for label,c in zip(labels,counts))
    hist('Household size',x[:,0],[.5,1.5,2.5,3.5,4.5,5.5],['1','2','3','4','5'])
    hist('Monthly income',x[:,1],[0,2500,5000,10000,15000,20000,30000,np.inf],
         ['<2.5k','2.5–5k','5–10k','10–15k','15–20k','20–30k','≥30k'])
    cash_months=np.rint(x[:,4]/x[:,1]);debt_months=np.rint(x[:,6]/x[:,1])
    assert np.max(np.abs(x[:,4]/x[:,1]-cash_months))<.001
    assert np.max(np.abs(x[:,6]/x[:,1]-debt_months))<.001
    hist('Cash buffer',cash_months,[-.5,.5,1.5,2.5,3.5,4.5,5.5,6.5],list(map(str,range(7))))
    hist('Debt exposure',debt_months,[-.5,.5,6.5,12.5,18.5,24.5],['0','1–6','7–12','13–18','19–24'])
    table(args.output_dir/'synthetic_persona_distributions.csv',
          ['metric','category','count','share_pct'],distributions)
    summaries=[]
    for j,name in enumerate(FEATURES):
        q=np.quantile(x[:,j],[.05,.25,.5,.75,.95])
        summaries.append(dict(feature=name,n=n,mean=float(x[:,j].mean()),
            p05=float(q[0]),p25=float(q[1]),median=float(q[2]),p75=float(q[3]),p95=float(q[4])))
    table(args.output_dir/'synthetic_persona_summary.csv',list(summaries[0]),summaries)
    z=x.copy();z[:,1:]=np.log1p(z[:,1:])
    z=(z-z.mean(axis=0))/z.std(axis=0,ddof=1)
    correlation=z.T@z/(n-1)
    eigenvalues,eigenvectors=np.linalg.eigh(correlation)
    order=np.argsort(eigenvalues)[::-1];eigenvalues=eigenvalues[order];eigenvectors=eigenvectors[:,order]
    for j in range(8):
        if eigenvectors[1,j]<0:eigenvectors[:,j]*=-1
    assert np.allclose(correlation,correlation.T) and np.allclose(np.diag(correlation),1)
    pca=[{'component':f'PC{j+1}','eigenvalue':float(eigenvalues[j]),
          'explained_variance_pct':float(eigenvalues[j]/eigenvalues.sum()*100)} for j in range(8)]
    loadings=[{'feature':name,'pc1_weight':float(eigenvectors[j,0]),
          'pc2_weight':float(eigenvectors[j,1])} for j,name in enumerate(FEATURES)]
    table(args.output_dir/'synthetic_persona_pca.csv',list(pca[0]),pca)
    table(args.output_dir/'synthetic_persona_loadings.csv',list(loadings[0]),loadings)
    result={'population':'independent synthetic companion','n':n,'shards_read':len(info['shards']),
      'features':FEATURES,'complete_records':n,'pca_transform':'log1p(CNY) for seven financial fields; household size untransformed; all eight standardized',
      'pca_weight_definition':'unit-length eigenvectors of the correlation matrix; PC1 and PC2 oriented toward positive income',
      'pca_variance_pct':[r['explained_variance_pct'] for r in pca],
      'monthly_spending_exceeds_income_n':int(((x[:,2]+x[:,3])>x[:,1]).sum()),
      'monthly_spending_and_dues_exceeds_income_n':int(((x[:,2]+x[:,3]+x[:,7])>x[:,1]).sum()),
      'elapsed_seconds':time.perf_counter()-t,'statistical_scope':'descriptive complete-population summaries',
      'model_calls':0,'generation_calls':0}
    (args.output_dir/'synthetic_persona_statistics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'complete','n':n,'pca_variance_pct':result['pca_variance_pct'][:2]}),flush=True)
if __name__=='__main__':main()

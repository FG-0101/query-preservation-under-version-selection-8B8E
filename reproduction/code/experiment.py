from pathlib import Path
import os
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
 os.environ[variable]="1"
import hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parent
INPUT=ROOT.parent/"data"
RASTER=INPUT/"raster_panel.npz"
SEED=20261011

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stable_hash(x):return int(hashlib.sha256((bytes.fromhex('4d555349435f5349474d4f445f76317c').decode('ascii')+str(x)).encode()).hexdigest()[:8],16)

def music():
 folder=INPUT/'audio'
 z=np.load(folder/'submission_metadata.npz',allow_pickle=False)
 x=np.asarray(np.load(folder/'submission_features.npy',mmap_mode='r'),np.float64)
 ids,group=np.unique(z['mbid'],return_inverse=True);count=np.bincount(group)
 mean=np.zeros((len(ids),4));np.add.at(mean,group,x);mean/=count[:,None]
 order=np.argsort(group,kind='stable');starts=np.r_[0,np.cumsum(count)]
 candidates=[np.unique(x[order[starts[i]:starts[i+1]]],axis=0) for i in range(len(ids))]
 hold=np.array([stable_hash(i)%5==0 for i in ids])
 return ids,candidates,mean,count,hold,z,x,group

def raster():
 z=np.load(RASTER,allow_pickle=False);a=z['integer_sums'][:,1].astype(np.float64)
 y=np.log1p(a);t=np.arange(16);centered=t-t.mean();post=np.arange(5)-2
 features=np.stack([y.mean(2),np.einsum('eci,i->ec',y,centered)/sum(centered**2),np.einsum('eci,i->ec',y[:,:,11:],post)/sum(post**2),y[:,:,11]-y[:,:,10]],axis=2)
 ids=np.array(['cell_'+str(i) for i in range(a.shape[1])]);candidates=[np.unique(features[:,i,:],axis=0) for i in range(len(ids))]
 mean=features.mean(0);count=np.full(len(ids),2);hold=np.array([stable_hash(i)%5==0 for i in ids])
 return ids,candidates,mean,count,hold,features

def m1(out):
 rows=[];details=[];rng=np.random.default_rng(SEED)
 for domain,data in [('audio',music()),('raster',raster())]:
  ids,candidates,ref,count,hold=data[:5];scale=np.maximum(ref[~hold].std(0),1e-12)
  first=np.array([c[0] for c in candidates]);last=np.array([c[-1] for c in candidates])
  medoid=np.array([c[np.argmin(np.sum(((c-ref[i])/scale)**2,axis=1))] for i,c in enumerate(candidates)])
  equal=np.array([c.mean(0) for c in candidates])
  policies={'first_lexicographic':first,'last_lexicographic':last,'equal_distinct_vector_mean':equal,'mean_nearest_candidate':medoid,'submission_mean':ref}
  minima=np.array([c.min(0) for c in candidates]);maxima=np.array([c.max(0) for c in candidates])
  radius=np.array([np.max(np.linalg.norm((c-ref[i])/scale,axis=1)) for i,c in enumerate(candidates)])
  for split,mask in [('development',~hold),('holdout',hold)]:
   positions=np.flatnonzero(mask);thresholds=np.quantile(ref[~hold],np.linspace(.01,.99,99),axis=0)
   refmean=ref[mask].mean(0);pooled=np.average(ref[mask],axis=0,weights=count[mask])
   covariance=((count[mask,None]-count[mask].mean())*(ref[mask]-refmean)).mean(0)/count[mask].mean()
   residual=float(np.max(abs(pooled-refmean-covariance)));assert residual<1e-9
   multi=positions[np.array([len(candidates[i])>1 for i in positions])]
   anchors=multi[np.argsort([stable_hash(ids[i]) for i in multi])[:500]]
   for name,value in policies.items():
    contrast=(value[mask]-ref[mask]).mean(0)
    changes=[];certified=0;policy_certified=0;certification_failures=0;set_loss=[];top1=[];margin_ratios=[]
    policy_radius=np.linalg.norm((value-ref)/scale,axis=1)
    for threshold in thresholds:
     for feature in range(4):
      target=ref[mask,feature]>=threshold[feature];selected=value[mask,feature]>=threshold[feature]
      uncertain=(minima[mask,feature]<threshold[feature])&(maxima[mask,feature]>=threshold[feature])
      changed=target!=selected;assert not np.any(changed&~uncertain)
      changes.append({'feature':feature,'threshold':float(threshold[feature]),'changed_entities':int(changed.sum()),'count_difference':int(selected.sum()-target.sum()),'candidate_crossings':int(uncertain.sum())})
    for anchor in anchors:
     eligible=positions[positions!=anchor]
     d0=np.linalg.norm((ref[eligible]-ref[anchor])/scale,axis=1);d1=np.linalg.norm((value[eligible]-value[anchor])/scale,axis=1)
     b=np.argsort(d0,kind='stable');c=np.argsort(d1,kind='stable');inside=b[:20];outside=b[20:]
     bound=radius[anchor]+radius[eligible]
     assert np.max(abs(d1-d0)-bound)<1e-8
     certificate=bool(np.max(d0[inside]+bound[inside])<np.min(d0[outside]-bound[outside]))
     policy_bound=policy_radius[anchor]+policy_radius[eligible]
     assert np.max(abs(d1-d0)-policy_bound)<1e-8
     policy_certificate=bool(np.max(d0[inside]+policy_bound[inside])<np.min(d0[outside]-policy_bound[outside]))
     same=set(eligible[inside])==set(eligible[c[:20]])
     certified+=certificate;policy_certified+=policy_certificate;certification_failures+=(certificate or policy_certificate) and not same
     aa=set(eligible[inside]);bb=set(eligible[c[:20]]);set_loss.append(1-len(aa&bb)/len(aa|bb));top1.append(bool(b[0]==c[0]))
    assert certification_failures==0
    result={'domain':domain,'split':split,'policy':name,'entities':int(mask.sum()),'uncertain_entities':len(multi),'queries':len(changes),'anchors':len(anchors),'mean_contrast':contrast.tolist(),'mean_absolute_threshold_membership_change':float(np.mean([c['changed_entities']/mask.sum() for c in changes])),'max_threshold_count_change':max(abs(c['count_difference']) for c in changes),'mean_top20_jaccard':1-float(np.mean(set_loss)) if set_loss else None,'top1_agreement':float(np.mean(top1)) if top1 else None,'candidate_radius_certified_queries':int(certified),'certificate_failures':int(certification_failures),'weight_identity_max_abs_residual':residual}
    result['policy_radius_certified_queries']=int(policy_certified)
    rows.append(result);details.append({'domain':domain,'split':split,'policy':name,'thresholds':changes})
  np.savez_compressed(out/(domain+'_contract_arrays.npz'),ids=ids,reference=ref,holdout=hold,minima=minima,maxima=maxima,radius=radius,scale=scale,**policies)
  print(json.dumps({'stage':'version_selection','domain':domain,'policies':len(policies),'cases':sum(r['queries']+r['anchors'] for r in rows if r['domain']==domain)}),flush=True)
 dump(out/'version_selection_comparisons.json',rows);dump(out/'version_selection_all_threshold_queries.json',details)
 return rows

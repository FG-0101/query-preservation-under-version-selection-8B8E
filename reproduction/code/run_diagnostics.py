#!/usr/bin/env python3
"""Compute the supplemental ranking, build-overlap and coverage diagnostics."""
import argparse, collections, hashlib, itertools, json, os, platform, resource, time
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
from run_queries import FEATURES, grouped, sha, dump
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--primary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);t=time.monotonic()
    protocol={"status":"Exploratory diagnostic selected after the primary analysis",
      "split":"Same fixed recording-ID split as the primary experiment",
      "ranking":"Up to 50 smallest ID-hash anchors within each stratum; compare each policy with allmean using the same policy-covered split-wide neighbor universe and development scale",
      "strata":["all","multiple_submissions","multiple_payloads"],
      "pair_selection":"Maximum development recordings with a common reported MD5 across two builds; then maximum development shared recording-MD5 pairs; lexicographic build pair breaks ties",
      "matched_build_contrast":"Within each recording, average each build within every common reported MD5, contrast those averages, then give each recording equal weight",
      "interpretation":"Reported digests are not independently validated audio; pair selection uses development support only; archive sample is not a known probability sample"}
    dump(a.out/"supplemental_protocol.json",protocol)
    md=np.load(a.data/"submission_metadata.npz",allow_pickle=False)
    primary=np.load(a.primary/"policy_recordings.npz",allow_pickle=False)
    ids=primary["recording"].astype(str);hold=primary["holdout"];R=len(ids)
    rec=md["mbid"].astype(str);pay=md["md5"].astype(str);build=md["build"].astype(str)
    ri=np.searchsorted(ids,rec);assert np.array_equal(ids[ri],rec)
    fi=md["feature"].astype(str);ix=[np.flatnonzero(fi==x).item() for x in FEATURES]
    X=np.load(a.data/"submission_features.npy",mmap_mode="r")
    V=np.asarray(X[:,ix],np.float64)
    ref=primary["recording_allmean"];scale=np.maximum(ref[~hold].std(0),1e-12)
    h=np.array([int(hashlib.sha256((bytes.fromhex('4d555349435f5349474d4f445f76317c').decode('ascii')+r).encode()).hexdigest()[:8],16) for r in ids],np.uint64)
    strata={"all":np.ones(R,bool),"multiple_submissions":primary["n_sub"]>1,"multiple_payloads":primary["n_payload"]>1}
    policies={k:primary[k] for k in primary.files if k.startswith("recording_") and k!="recording"}
    rows=[];details=[]
    for split,sm in {"development":~hold,"holdout":hold}.items():
        for st,stm in strata.items():
            candidates=np.flatnonzero(sm&stm);anchors=candidates[np.argsort(h[candidates],kind="stable")[:50]]
            for policy,A in policies.items():
                support=sm&np.isfinite(A).all(1);univ=np.flatnonzero(support);jc=[];ta=[];changed=0
                for anchor in anchors:
                    if not support[anchor]:continue
                    eligible=univ[univ!=anchor]
                    if len(eligible)<20:continue
                    d0=np.sum(((ref[eligible]-ref[anchor])/scale)**2,axis=1)
                    d1=np.sum(((A[eligible]-A[anchor])/scale)**2,axis=1)
                    n0=eligible[np.argsort(d0,kind="stable")[:20]];n1=eligible[np.argsort(d1,kind="stable")[:20]]
                    j=len(set(n0)&set(n1))/len(set(n0)|set(n1));top=bool(n0[0]==n1[0])
                    jc.append(j);ta.append(top);changed+=int(not np.array_equal(n0,n1))
                    details.append({"split":split,"stratum":st,"policy":policy,"anchor":str(ids[anchor]),
                        "jaccard":j,"same_top1":top,"reference_neighbors":ids[n0].tolist(),"policy_neighbors":ids[n1].tolist()})
                rows.append({"split":split,"stratum":st,"policy":policy,"stratum_recordings":len(candidates),
                    "eligible_neighbor_recordings":len(univ),"anchors_selected":len(anchors),"anchors_evaluated":len(jc),
                    "mean_top20_jaccard":float(np.mean(jc)) if jc else None,"top1_agreement":float(np.mean(ta)) if ta else None,
                    "ordered_top20_changed_anchors":changed})
    dump(a.out/"stratified_ranking.json",rows);dump(a.out/"anchor_query_details.json",details)
    
    builds=sorted(set(build));brec={};bp={}
    for b in builds:
        ii=np.flatnonzero(build==b);brec[b]=set(ri[ii].tolist());bp[b]={}
        for i in ii:
            if pay[i]!="":bp[b].setdefault((int(ri[i]),str(pay[i])),[]).append(int(i))
    graph=[]
    for ba,bb in itertools.combinations(builds,2):
        common=brec[ba]&brec[bb];shared=set(bp[ba])&set(bp[bb]);matched_rec={k[0] for k in shared}
        entry={"build_A":ba,"build_B":bb}
        for split,sm in {"development":~hold,"holdout":hold}.items():
            entry[split]={"common_recordings":sum(bool(sm[r]) for r in common),
                "matched_payload_recordings":sum(bool(sm[r]) for r in matched_rec),
                "matched_recording_payload_pairs":sum(bool(sm[k[0]]) for k in shared)}
        graph.append(entry)
    dump(a.out/"complete_build_overlap.json",{"build_count":len(builds),"pairs":graph})
    ranked=sorted(graph,key=lambda g:(-g["development"]["matched_payload_recordings"],
        -g["development"]["matched_recording_payload_pairs"],g["build_A"],g["build_B"]))
    selected=ranked[0];ba=selected["build_A"];bb=selected["build_B"]
    shared=sorted(set(bp[ba])&set(bp[bb]));byrec=collections.defaultdict(list)
    for key in shared:
        byrec[key[0]].append(V[bp[bb][key]].mean(0)-V[bp[ba][key]].mean(0))
    contrasts=[]
    for split,sm in {"development":~hold,"holdout":hold}.items():
        entries=[(r,np.mean(d,axis=0)) for r,d in sorted(byrec.items()) if sm[r]]
        d=np.array([v for r,v in entries],float).reshape(-1,len(FEATURES))
        for j,f in enumerate(FEATURES):
            contrasts.append({"split":split,"feature":f,"matched_recordings":len(entries),
                "matched_payload_pairs":sum(len(byrec[r]) for r,v in entries),
                "mean_contrast":float(d[:,j].mean()) if len(d) else None,
                "median_absolute_contrast":float(np.median(np.abs(d[:,j]))) if len(d) else None,
                "max_absolute_contrast":float(np.max(np.abs(d[:,j]))) if len(d) else None})
    dump(a.out/"development_selected_matched_build.json",{"selected_pair":selected,
        "development_pair_supported":selected["development"]["matched_payload_recordings"]>0,
        "contrasts":contrasts})
    
    decomposition=[]
    for split,sm in {"development":~hold,"holdout":hold}.items():
        full=ref[sm].mean(0)
        for policy,A in policies.items():
            sp=sm&np.isfinite(A).all(1)
            if not sp.any():continue
            covref=ref[sp].mean(0);ans=A[sp].mean(0)
            within=ans-covref;coverage=covref-full;total=ans-full
            assert np.max(np.abs(total-within-coverage))<1e-9
            for j,f in enumerate(FEATURES):
                decomposition.append({"split":split,"policy":policy,"feature":f,"full_recordings":int(sm.sum()),
                    "covered_recordings":int(sp.sum()),"total_difference":float(total[j]),
                    "within_support_representation_difference":float(within[j]),"coverage_composition_difference":float(coverage[j]),
                    "residual":float(total[j]-within[j]-coverage[j])})
    dump(a.out/"support_decomposition.json",decomposition)
    dump(a.out/"diagnostic_receipt.json",{"code_sha256":sha(__file__),"input_sha256":{
        n:sha(a.data/n) for n in ["submission_features.npy","submission_metadata.npz"]},
        "primary_policy_sha256":sha(a.primary/"policy_recordings.npz"),
        "software":{"python":platform.python_version(),"numpy":np.__version__},
        "elapsed_seconds":time.monotonic()-t,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs_sha256":{x.name:sha(x) for x in a.out.iterdir() if x.is_file()}})
    print(json.dumps({"build_pairs":len(graph),"selected_pair":selected,"stratified_summaries":len(rows),
        "elapsed_seconds":time.monotonic()-t}),flush=True)
if __name__=="__main__":main()

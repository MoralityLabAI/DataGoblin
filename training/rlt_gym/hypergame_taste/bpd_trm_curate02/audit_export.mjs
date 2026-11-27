// Audit pinned BPD train-only export. Usage: node audit_export.mjs <export-directory>
import {readFileSync} from "node:fs";
import {join} from "node:path";
function sha256(s){
  const K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
  const H=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19],w=new Int32Array(64),ro=(x,n)=>(x>>>n)|(x<<(32-n));
  // All pinned source rows contain ASCII only. Fail closed for other encodings.
  if(/[^\x00-\x7f]/.test(s))throw Error("non ASCII");
  const n=s.length,padded=Math.ceil((n+9)/64)*64,bits=n*8;
  for(let off=0;off<padded;off+=64){
    for(let j=0;j<16;j++){let v=0;for(let k=0;k<4;k++){let i=off+4*j+k,b=i<n?s.charCodeAt(i):i===n?128:i>=padded-8?Math.floor(bits/2**(8*(padded-1-i)))&255:0;v=(v<<8)|b;}w[j]=v;}
    for(let j=16;j<64;j++){const a=w[j-15],b=w[j-2],s0=ro(a,7)^ro(a,18)^(a>>>3),s1=ro(b,17)^ro(b,19)^(b>>>10);w[j]=(w[j-16]+s0+w[j-7]+s1)|0;}
    let [a,b,c,d,e,f,g,h]=H;
    for(let j=0;j<64;j++){const s1=ro(e,6)^ro(e,11)^ro(e,25),ch=(e&f)^(~e&g),t1=(h+s1+ch+K[j]+w[j])|0,s0=ro(a,2)^ro(a,13)^ro(a,22),maj=(a&b)^(a&c)^(b&c),t2=(s0+maj)|0;h=g;g=f;f=e;e=(d+t1)|0;d=c;c=b;b=a;a=(t1+t2)|0;}
    [a,b,c,d,e,f,g,h].forEach((v,j)=>H[j]=(H[j]+v)|0);
  }return H.map(x=>(x>>>0).toString(16).padStart(8,"0")).join("");
}
function auditExport(manifest,decRaw,prefRaw){
 const A=(b,m)=>{if(!b)throw Error(m)},mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
 const hashes={"candidate_decisions.jsonl":sha256(decRaw),"curation_preferences.jsonl":sha256(prefRaw)};
 A(Object.entries(hashes).every(([k,v])=>manifest.files_sha256[k]===v),"file sha256");
 A(manifest.train_only===true&&manifest.evaluation_rows_exported===0,"manifest custody");
 const rows=decRaw.trimEnd().split("\n").map(JSON.parse),prefs=prefRaw.trimEnd().split("\n").map(JSON.parse);
 const train=x=>x.reduce((a,b,i)=>a+b*4**i,0)%5>=2,verts=Array.from({length:256},(_,i)=>[i&3,i>>2&3,i>>4&3,i>>6&3]),edgeKey=(r,h,s)=>JSON.stringify([r,h,s]),tuple=a=>"("+a.join(", ")+")";
 const groups=new Map(),fit=new Set(),feedback=new Set(),tune=new Set();let trainEdges=0,evalEdges=0;
 for(const r of verts)for(let s=0;s<4;s++)for(let v=0;v<4;v++)if(v!==r[s]){
  const h=[...r];h[s]=v;if(train(r)!==train(h))continue;if(!train(r)){evalEdges++;continue;}trainEdges++;
  if(r[s]>v)continue;const key=[s,r[s],v].join("/");if(!groups.has(key))groups.set(key,[]);
  const repr=`(20261127, 'split', (${tuple(r)}, ${tuple(h)}, ${s}))`;
  groups.get(key).push({r,h,s,hash:sha256(repr)});
 }
 for(const a of groups.values()){
  a.sort((x,y)=>x.hash.localeCompare(y.hash));const hold=Math.max(1,Math.floor(a.length/6));
  a.forEach((x,i)=>{const g=i<hold?feedback:i<2*hold?tune:fit;g.add(edgeKey(x.r,x.h,x.s));g.add(edgeKey(x.h,x.r,x.s));});
 }
 A(fit.size===624&&feedback.size===144&&tune.size===144&&trainEdges===912&&evalEdges===312,"semantic partition");
 const ids=new Map(),windows=new Map(),roots=new Map(),exposures=new Map(),finiteKeys=["train_loss","predicted_gain","update_norm","predicted_utility"];
 let selected=0,positive=0,selectedPos=0,selectedPosAccepted=0,selectedPosRejected=0,signMatch=0;
 for(const x of rows){
  A(x.schema_version==="datagoblin.trm_curation_decision.v1"&&x.split==="train"&&x.training_eligible===true&&x.evaluation_leakage===false,"row custody");
  const v=x.input,p=x.provenance,f=x.mechanistic_features,o=x.outcome,c=x.resource_cost.charged_cpu_seconds;
  A(fit.has(edgeKey(v.edge_receiver,v.edge_hybrid,v.edit_slot)),"outside fit "+x.record_id);
  A(Number.isInteger(v.delay)&&v.delay>=0&&p.arm==="D_mech_grad"&&p.experiment===manifest.experiment&&manifest.source_sha256["root_"+p.root]===p.source_file_sha256,"provenance");
  A(Number.isInteger(p.window)&&p.window>=0&&p.window<80&&/^[a-f0-9]{64}$/.test(p.checkpoint_sha256)&&p.epoch===`w${p.window}:${p.checkpoint_sha256.slice(0,16)}`,"epoch");
  const recId=sha256(JSON.stringify({checkpoint_sha256:p.checkpoint_sha256,edge:[v.edge_receiver,v.edge_hybrid,v.edit_slot],root:p.root,window:p.window}));
  A(recId===x.record_id&&!ids.has(recId),"ID");ids.set(recId,x);
  A(finiteKeys.every(k=>Number.isFinite(f[k]))&&Number.isFinite(c)&&c>0&&Number.isFinite(o.realized_gain)&&Number.isFinite(o.realized_utility),"finite");
  A(Math.abs(o.realized_utility-o.realized_gain/Math.max(c,1e-9))<1e-8*Math.max(1,Math.abs(o.realized_utility)),"realized utility");
  A(Math.abs(f.predicted_utility-f.predicted_gain/c)<1e-8*Math.max(1,Math.abs(f.predicted_utility)),"predicted utility");
  A(o.sign_match===((f.predicted_gain>0)===(o.realized_gain>0))&&typeof o.selected==="boolean"&&typeof o.window_accepted==="boolean","flags");
  A(Number.isFinite(o.window_cal_before)&&Number.isFinite(o.window_cal_after)&&o.window_accepted===(o.window_cal_after<=o.window_cal_before+1e-6),"window acceptance");
  const key=[p.root,p.arm,p.window].join("/");if(!windows.has(key))windows.set(key,[]);windows.get(key).push(x);
  if(!roots.has(p.root))roots.set(p.root,{rows:0,selected:0,selectedPositive:0,acceptedSelectedPositive:0,positive:0,acceptedWindows:0,pairs:0});
  const r=roots.get(p.root);r.rows++;if(o.realized_gain>0){positive++;r.positive++;}if(o.sign_match)signMatch++;
  if(o.selected){selected++;r.selected++;const pair=[v.edge_receiver,v.edge_hybrid].map(JSON.stringify).sort();const e=[p.root,...pair,v.edit_slot].join("/");exposures.set(e,(exposures.get(e)||0)+1);if(o.realized_gain>0){selectedPos++;r.selectedPositive++;if(o.window_accepted){selectedPosAccepted++;r.acceptedSelectedPositive++;}else selectedPosRejected++;}}
 }
 let acceptedWindows=0,positivePairs=0;
 for(const [k,a] of windows){A(a.length===16&&new Set(a.map(x=>x.provenance.checkpoint_sha256)).size===1&&a.filter(x=>x.outcome.selected).length<=4,"window roster");const o=a[0].outcome;A(a.every(x=>x.outcome.window_accepted===o.window_accepted&&x.outcome.window_cal_before===o.window_cal_before&&x.outcome.window_cal_after===o.window_cal_after),"window shared labels");if(o.window_accepted){acceptedWindows++;roots.get(a[0].provenance.root).acceptedWindows++;}}
 const pairWindows=new Set();
 for(const p of prefs){
  A(p.schema_version==="datagoblin.trm_curation_preference.v1"&&p.split==="train"&&p.training_eligible===true&&p.evaluation_leakage===false,"pair custody");
  const a=ids.get(p.chosen.record_id),b=ids.get(p.rejected.record_id);A(a&&b,"pair references");
  A(a.provenance.root===p.root&&b.provenance.root===p.root&&a.provenance.window===p.window&&b.provenance.window===p.window&&a.provenance.checkpoint_sha256===p.checkpoint_sha256&&b.provenance.checkpoint_sha256===p.checkpoint_sha256,"pair comparability");
  A(a.outcome.selected&&!b.outcome.selected&&a.outcome.realized_gain>0&&b.outcome.realized_gain<=0,"preference outcome");
  for(const [x,y] of [[p.chosen,a],[p.rejected,b]])A(JSON.stringify(x.input)===JSON.stringify(y.input)&&JSON.stringify(x.mechanistic_features)===JSON.stringify(y.mechanistic_features)&&x.realized_gain===y.outcome.realized_gain,"pair copied fields");
  A(Math.abs(p.realized_margin-(a.outcome.realized_gain-b.outcome.realized_gain))<1e-12,"preference margin");
  const k=p.root+"/"+p.window;A(!pairWindows.has(k),"duplicate preference window");pairWindows.add(k);roots.get(p.root).pairs++;positivePairs++;
 }
 A(rows.length===manifest.total_candidate_decisions&&selectedPos===manifest.successful_selected_decisions&&prefs.length===manifest.verified_preference_pairs,"manifest counts");
 const maxExposure=Math.max(...exposures.values());A(maxExposure<=4,"unordered selected exposure cap");
 // Root-held-out rank and sign diagnostics on existing scores, no new model fit.
 const rootStats=[...roots].sort((a,b)=>a[0]-b[0]).map(([root,x])=>({root,...x}));
 return{schema:"BPD-TRM-CURATE-02-datagoblin-ingress-audit-v1",hashes,semantic:{trainEdges,evalEdges,fit:fit.size,feedback:feedback.size,tune:tune.size},rows:rows.length,pairs:prefs.length,windows:windows.size,acceptedWindows,selected,positiveProbeRows:positive,selectedPositiveProbeRows:selectedPos,selectedPositiveInAcceptedWindows:selectedPosAccepted,selectedPositiveInRejectedWindows:selectedPosRejected,signAccuracy:signMatch/rows.length,maxSelectedUnorderedEdgeExposurePerRoot:maxExposure,rootStats,scope:"hash, fit-view, provenance, window, exposure and pair audit; individual probe outcomes are not actual committed update outcomes"};
}
const dir=process.argv[2];if(!dir)throw Error("export directory required");
const read=f=>readFileSync(join(dir,f),"utf8");
console.log(JSON.stringify(auditExport(JSON.parse(read("manifest.json")),read("candidate_decisions.jsonl"),read("curation_preferences.jsonl")),null,2));

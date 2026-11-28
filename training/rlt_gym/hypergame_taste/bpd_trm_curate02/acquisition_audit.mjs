// Replay only. Run: node acquisition_audit.mjs candidate_decisions.jsonl
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
function audit(raw){
 const A=(x,m)=>{if(!x)throw Error(m)},sum=a=>a.reduce((x,y)=>x+y,0);
 const rows=raw.trimEnd().split("\n").map(JSON.parse),groups=new Map(),ids=new Set();
 for(const x of rows){const p=x.provenance;A(x.split==="train"&&x.training_eligible===true&&x.evaluation_leakage===false&&p.arm==="D_mech_grad"&&!ids.has(x.record_id)&&Number.isFinite(x.mechanistic_features.predicted_gain),"invalid row");ids.add(x.record_id);const k=[p.root,p.window,p.checkpoint_sha256].join(":");if(!groups.has(k))groups.set(k,[]);groups.get(k).push(x);}
 A(rows.length===10240&&groups.size===640,"roster");
 const windows=[...groups.values()].map(r=>{const o=r[0].outcome,p=r[0].provenance;A(r.length===16&&r.filter(x=>x.outcome.selected).length===4&&r.every(x=>x.outcome.window_cal_before===o.window_cal_before&&x.outcome.window_cal_after===o.window_cal_after&&x.outcome.window_accepted===o.window_accepted),"window");
 const slots=Array.from({length:4},(_,i)=>r.filter(x=>x.input.edit_slot===i).sort((a,b)=>a.record_id.localeCompare(b.record_id)));A(slots.every(z=>z.length===4),"slot balance");
 const ordered=Array.from({length:4},(_,j)=>slots.map(s=>s[j])).flat();
 const gain=o.window_cal_before-o.window_cal_after;A(Number.isFinite(gain),"gain");
 return{root:p.root,window:p.window,accepted:o.window_accepted,gain,positiveGain:o.window_accepted?Math.max(0,gain):0,max:[4,8,12,16].map(n=>Math.max(...ordered.slice(0,n).map(x=>x.mechanistic_features.predicted_gain)))};
 });
 const roots=[...new Set(windows.map(w=>w.root))].sort((a,b)=>a-b);A(roots.length===8&&roots.every(r=>windows.filter(w=>w.root===r).length===80),"root count");
 // A threshold of null disables early stop. The final full-screen zero gate is mandatory.
 function decision(w,t){for(let j=0;j<3;j++)if(t[j]!==null&&w.max[j]<=t[j])return{acquired:(j+1)*4,attempt:false};return{acquired:16,attempt:w.max[3]>0};}
 function stats(ws,t,always=false){let acq=0,attempts=0,accepted=0,miss=0,gain=0,missGain=0,signedMiss=0;
 for(const w of ws){let d=always?{acquired:16,attempt:true}:decision(w,t);acq+=d.acquired;attempts+=+d.attempt;accepted+=+w.accepted;gain+=w.positiveGain;if(!d.attempt&&w.accepted){miss++;missGain+=w.positiveGain;signedMiss+=w.gain;}}
 return{windows:ws.length,candidatesAcquired:acq,attempts,accepted,acceptedMissed:miss,rejectedAvoided:ws.length-attempts-miss,positiveAcceptedGain:gain,missedPositiveGain:missGain,missedSignedGain:signedMiss,retainedFraction:gain?1-missGain/gain:1};
 }
 const grid=[null,-.1,-.05,-.02,-.01,-.005,0],policies=grid.flatMap(a=>grid.flatMap(b=>grid.map(c=>[a,b,c])));
 const baseline=[null,null,null];
 const folds=[];
 for(const root of roots){const train=windows.filter(w=>w.root!==root),test=windows.filter(w=>w.root===root);
 const trainGroups=roots.filter(r=>r!==root).map(r=>train.filter(w=>w.root===r));
 const feasible=policies.map(t=>({t,s:stats(train,t)})).filter(z=>z.s.retainedFraction>=.995&&trainGroups.every(g=>stats(g,z.t).retainedFraction>=.98));
 feasible.sort((a,b)=>a.s.candidatesAcquired-b.s.candidatesAcquired||a.s.attempts-b.s.attempts||a.s.missedPositiveGain-b.s.missedPositiveGain||JSON.stringify(a.t).localeCompare(JSON.stringify(b.t)));
 A(feasible.length>0,"no feasible policy");const best=feasible[0];folds.push({root,thresholds:best.t,training:best.s,eligiblePolicies:feasible.length,test:stats(test,best.t),fullGate:stats(test,baseline)});
 }
 const keys=["windows","candidatesAcquired","attempts","accepted","acceptedMissed","rejectedAvoided","positiveAcceptedGain","missedPositiveGain","missedSignedGain"];
 const held=Object.fromEntries(keys.map(k=>[k,sum(folds.map(f=>f.test[k]))]));held.retainedFraction=1-held.missedPositiveGain/held.positiveAcceptedGain;
 const controls={always:stats(windows,baseline,true),fullGate:stats(windows,baseline),early4:stats(windows,[0,null,null]),early8:stats(windows,[null,0,null]),early12:stats(windows,[null,null,0])};
 A(controls.fullGate.acceptedMissed===1&&controls.always.accepted===203&&controls.fullGate.attempts===364,"prior replay mismatch");
 return{schema:"bpd-curate03b-paid-acquisition-v1",scope:"spent development trajectories; replay of skip/acquire decisions for original recorded minibatch, never relabel an alternative minibatch; acquisition counts are logical, not measured CPU",ordering:"round-robin edit slot 0..3, record ID ascending within slot; order constructed post-hoc, must be replaced by prospective stable edge key before deployment",training:"leave-one-root-out exhaustive 343 threshold triples; constrain pooled positive recorded accepted gain retention >=99.5%, every training root >=98%; minimize acquired candidates, then attempts, then lost positive gain; fixed thresholds are post-hoc comparators",controls,heldoutAdaptive:held,folds};
}
const raw=readFileSync(process.argv[2],"utf8");
if(createHash("sha256").update(raw).digest("hex")!=="805220a18fd5c8f58e940e7238ff6065e5682b6006fcd01ceb85a4693484a748")throw Error("source hash mismatch");
console.log(JSON.stringify(audit(raw),null,2));

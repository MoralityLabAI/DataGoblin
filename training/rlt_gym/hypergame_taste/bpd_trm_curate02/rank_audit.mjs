// Development-only root-grouped candidate ranking audit.
import {readFileSync} from "node:fs";
function evaluate(raw){
 const rows=raw.trimEnd().split("\n").map(JSON.parse),A=(v,m)=>{if(!v)throw Error(m)},sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length,seq=n=>Array.from({length:n},(_,i)=>i);
 const roots=[...new Set(rows.map(x=>x.provenance.root))].sort((a,b)=>a-b);
 function feats(x){
  const f=x.mechanistic_features,i=x.input,p=f.predicted_gain,l=Math.log1p(Math.max(0,f.train_loss)),u=f.update_norm;
  // All features are available after gradient and analytical displacement, before evaluating post-step calibration.
  return[1,p,p*Math.abs(p),l,u,i.delay/3,p*l,p*u,...seq(4).map(k=>+(i.edit_slot===k)),...i.edge_receiver.map(v=>v/3),...i.edge_hybrid.map(v=>v/3)];
 }
 function fit(a){
  const x=a.map(feats),y=a.map(r=>r.outcome.realized_gain),D=x[0].length,mu=seq(D).map(k=>k===0?0:mean(x.map(v=>v[k]))),scale=seq(D).map(k=>k===0?1:Math.max(1e-6,Math.sqrt(mean(x.map(v=>(v[k]-mu[k])**2)))));
  const norm=f=>f.map((v,k)=>(v-mu[k])/scale[k]),M=seq(D).map(i=>seq(D+1).map(j=>i===j?1:0));
  for(let i=0;i<x.length;i++){const z=norm(x[i]);for(let j=0;j<D;j++){for(let k=0;k<D;k++)M[j][k]+=z[j]*z[k];M[j][D]+=z[j]*y[i];}}
  for(let i=0;i<D;i++){let p=i;for(let j=i+1;j<D;j++)if(Math.abs(M[j][i])>Math.abs(M[p][i]))p=j;[M[i],M[p]]=[M[p],M[i]];const v=M[i][i];A(Math.abs(v)>1e-12,"singular");for(let j=i;j<=D;j++)M[i][j]/=v;for(let k=0;k<D;k++)if(k!==i){const z=M[k][i];for(let j=i;j<=D;j++)M[k][j]-=z*M[i][j];}}
  const w=M.map(a=>a[D]);return{score:x=>sum(norm(feats(x)).map((z,i)=>z*w[i])),w,mu,scale};
 }
 const results=[],arms=["loss","predicted","predicted_per_cpu","ridge","oracle","recorded"];
 for(const root of roots){
  const train=rows.filter(x=>x.provenance.root!==root),test=rows.filter(x=>x.provenance.root===root),m=fit(train),windows=new Map();
  for(const x of test){const k=x.provenance.window;if(!windows.has(k))windows.set(k,[]);windows.get(k).push(x);}
  const metrics=Object.fromEntries(arms.map(k=>[k,{top4:[],positive:[],best:[],pairCorrect:0,pairCount:0}]));
  const score={loss:x=>x.mechanistic_features.train_loss,predicted:x=>x.mechanistic_features.predicted_gain,predicted_per_cpu:x=>x.mechanistic_features.predicted_utility,ridge:m.score,oracle:x=>x.outcome.realized_gain,recorded:x=>+x.outcome.selected};
  for(const a of windows.values()){
   A(a.length===16,"window length");
   for(const arm of arms){const z=[...a].sort((x,y)=>score[arm](y)-score[arm](x)||x.record_id.localeCompare(y.record_id)).slice(0,4),v=z.map(x=>x.outcome.realized_gain);
    metrics[arm].top4.push(mean(v));metrics[arm].positive.push(mean(v.map(x=>+(x>0))));metrics[arm].best.push(Math.max(...v));
    if(arm!=="recorded")for(let i=0;i<a.length;i++)for(let j=i+1;j<a.length;j++){const y=a[i].outcome.realized_gain-a[j].outcome.realized_gain,p=score[arm](a[i])-score[arm](a[j]);if(y!==0){metrics[arm].pairCount++;metrics[arm].pairCorrect+=p===0?.5:+(Math.sign(y)===Math.sign(p));}}
   }
  }
  const metric=Object.fromEntries(arms.map(k=>[k,{meanTop4RealizedGain:mean(metrics[k].top4),top4PositiveFraction:mean(metrics[k].positive),meanBestOf4:mean(metrics[k].best),pairAccuracy:metrics[k].pairCount?metrics[k].pairCorrect/metrics[k].pairCount:null}]));
  results.push({root,trainRows:train.length,testRows:test.length,windows:windows.size,metrics:metric,model:{weights:m.w,mean:m.mu,scale:m.scale},regressionMSE:{ridge:mean(test.map(x=>(m.score(x)-x.outcome.realized_gain)**2)),predicted:mean(test.map(x=>(score.predicted(x)-x.outcome.realized_gain)**2))}});
 }
 const metrics=["meanTop4RealizedGain","top4PositiveFraction","meanBestOf4","pairAccuracy"];
 const summary=Object.fromEntries(arms.map(k=>[k,Object.fromEntries(metrics.map(f=>[f,k==="recorded"&&f==="pairAccuracy"?null:mean(results.map(r=>r.metrics[k][f]))]))]));
 const a=results.map(r=>r.metrics.ridge.meanTop4RealizedGain-r.metrics.predicted.meanTop4RealizedGain),d=mean(a),sd=Math.sqrt(sum(a.map(x=>(x-d)**2))/(a.length-1));
 return{schema:"BPD-TRM-CURATE-02-root-heldout-rank-audit-v1",scope:"post-hoc within-development leave-one-root-out candidate ranking; top-4 independent probe labels, not committed minibatch or held-out task accuracy",features:"20-dimensional: intercept, predicted gain, signed quadratic, log train loss, update norm, delay/3, two interactions, slot onehot, receiver/hybrid coordinates. Root held-out normalization; L2=1. No realized/selection/window-outcome fields in learned inputs.",summary,ridgeVsPredicted:{mean:d,descriptive95:[d-1.96*sd/Math.sqrt(a.length),d+1.96*sd/Math.sqrt(a.length)],positiveRoots:a.filter(x=>x>0).length},meanMSE:{ridge:mean(results.map(r=>r.regressionMSE.ridge)),predicted:mean(results.map(r=>r.regressionMSE.predicted))},results};
}
console.log(JSON.stringify(evaluate(readFileSync(process.argv[2],"utf8")),null,2));

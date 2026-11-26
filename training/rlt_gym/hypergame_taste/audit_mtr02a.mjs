import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
export function auditMtr02a(receipt,weights) {
  const check=(v,m)=>{if(!v)throw Error(m);},sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  const metric=["utility","cost","queries","publication","falsePublication","fullPublication"];
  let checks=0;
  for(const a of receipt.aggregate){
    const roots=receipt.rootResults.filter(r=>r.split===a.split);
    check(roots.length===a.roots,"root count");
    for(const [p,m] of Object.entries(a.mean))for(const k of metric){
      check(Math.abs(mean(roots.map(r=>r.results[p][k]))-m[k])<=0.000000501,"mean mismatch");checks++;
    }
    for(const [p,m] of Object.entries(a.comparisons))for(const k of Object.keys(m)){
      const d=roots.map(r=>r.results.rlt[k]-r.results[p][k]),x=mean(d),sd=Math.sqrt(sum(d.map(v=>(v-x)**2))/(d.length-1));
      const expected=[x,x-1.96*sd/Math.sqrt(d.length),x+1.96*sd/Math.sqrt(d.length)];
      check(expected.every((v,i)=>Math.abs(v-m[k][i])<=0.000000501),"interval mismatch");checks+=3;
    }
  }
  check(new Set(receipt.rootResults.map(r=>r.id)).size===receipt.rootResults.length,"duplicate evaluation ID");
  check(weights.length===9&&weights.every(w=>w.p.length===receipt.parameterCount&&w.p.every(Number.isFinite)),"invalid weights");
  const details=receipt.aggregate.map(a=>{
    const roots=receipt.rootResults.filter(r=>r.split===a.split),seedMeans={};
    for(const p of ["rlt","damping","pooled"])seedMeans[p]=receipt.modelSeeds.map((_,j)=>mean(roots.map(r=>r.results[p].seedUtility[j])));
    return {split:a.split,seedMeans,queryRatioVsDamping:a.mean.rlt.queries/a.mean.damping.queries,queryRatioVsPooled:a.mean.rlt.queries/a.mean.pooled.queries,
      creditRatioVsDamping:a.mean.rlt.cost/a.mean.damping.cost,creditRatioVsPooled:a.mean.rlt.cost/a.mean.pooled.cost};
  });
  return {scope:"receipt arithmetic and finite parameter checks; not independent policy replay",checks,details};
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const read=name=>JSON.parse(readFileSync(new URL(name,import.meta.url),"utf8"));
  console.log(JSON.stringify(auditMtr02a(read("mtr02a_receipt.json"),read("mtr02a_weights.json")),null,2));
}

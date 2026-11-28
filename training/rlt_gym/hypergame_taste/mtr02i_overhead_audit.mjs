// Post-hoc forced-route overhead control. Pinned source transformation.
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
function transform(original){
  let s=original.slice(original.indexOf("export function"),original.indexOf("\nif(process.argv"));
  s=s.replace("export function mtr02i(seed=20261127,scale=1){","function overhead(reference){\n const {seed,scale}=reference;");
  const a="      }return best;\n    }\n    function fingerprint(){";
  if(s.split(a).length!==2)throw Error("unexpected scorer tail");
  s=s.replace(a,"      }return mode===\"charged_route\"?0:best;\n    }\n    function fingerprint(){");
  const i=s.indexOf("  const trainN=32*scale");if(i<0)throw Error("training boundary missing");
  s=s.slice(0,i)+`  const rows=[],models={permuted:Array(70).fill(0)};
  for(const ref of reference.rootResults){
    const i=Number(ref.id.split("/").at(-1)),r=make(ref.split,i);
    A(r.id===ref.id,"root identity");
    for(const b of ref.atBudget){
      const x=run(r,b.budget,"charged_route",models),route=run(r,b.budget,"route");
      for(const k of ["raw","utility","lower","upper","work","actionJobs","boundJobs","optionCount","completedOptions","falsePublication"]){
        A(x[k]===b.arms.learned[k],"charged control mismatch "+k);
        A(route[k]===b.arms.route[k],"route replay mismatch "+k);
      }
      A(JSON.stringify(x.counts)===JSON.stringify(b.arms.learned.counts),"work counters");
      A(JSON.stringify(x.options)===JSON.stringify(b.arms.learned.options),"option trace");
      rows.push({id:r.id,split:r.split,budget:b.budget,routeUtility:route.utility,chargedUtility:x.utility,routeSelectionWork:(route.counts.priority||0)+(route.counts.feature||0),chargedSelectionWork:(x.counts.priority||0)+(x.counts.feature||0),matched:true});
    }
  }
  return {schema:"MTR-02I-posthoc-overhead-control-v1",scope:"frozen public development roots; forced route with zero-weight scoring loop; no retraining",rows,aggregate:["ordinary","ood"].flatMap(split=>reference.budgets.map(budget=>{const rr=rows.filter(r=>r.split===split&&r.budget===budget);return {split,budget,roots:rr.length,matched:rr.filter(x=>x.matched).length,route:mean(rr.map(x=>x.routeUtility)),charged:mean(rr.map(x=>x.chargedUtility)),extraSelectionWork:mean(rr.map(x=>x.chargedSelectionWork-x.routeSelectionWork))};}))};
}`;
  return s;
}
const original=readFileSync(new URL("mtr02i.mjs",import.meta.url),"utf8"),bytes=Buffer.from(original);
const sha=createHash("sha1").update("blob "+bytes.length+"\0").update(bytes).digest("hex");
if(sha!=="2e2e2539607ac9b36e9fb5e368c35f6436fa6a91")throw Error("pinned source changed");
const reference=JSON.parse(readFileSync(new URL("mtr02i_receipt.json",import.meta.url),"utf8"));
const run=new Function(transform(original)+"\nreturn overhead;")();
console.log(JSON.stringify(run(reference),null,2));

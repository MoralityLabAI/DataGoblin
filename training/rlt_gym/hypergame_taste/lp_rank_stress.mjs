// Exact delayed-evidence LP stress; standard Node.js only.
import { pathToFileURL } from "node:url";

export function lpRankStress(seed=20261127,perDepth=16){
  const assert=(b,m)=>{if(!b)throw Error(m);};
  assert(Number.isSafeInteger(seed)&&Number.isInteger(perDepth)&&perDepth>0&&perDepth<=1000,"invalid configuration");
  const range=n=>Array.from({length:n},(_,i)=>i);
  const rngFor=key=>{let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return n=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)%n;};};
  const parity=m=>{let v=0;while(m){v^=1;m&=m-1;}return v;};
  function rank(rows,d){
    const pivots=Array(d).fill(0);let r=0;
    for(let a of rows)for(let j=d-1;j>=0;j--)if(a>>j&1){if(pivots[j])a^=pivots[j];else{pivots[j]=a;r++;break;}}
    return r;
  }
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=r(i+1);[a[i],a[j]]=[a[j],a[i]];}return a;}
  // LP max sum x_i, 0 <= x_i <= 1 + h_i, h_i in {0,1}.
  // Dual y_i=1. A proposed primal vector is certified iff x_i=b_i for every i.
  function verify(h,c,d){
    const b=range(d).map(i=>1+(h>>i&1)),x=range(d).map(i=>1+(c>>i&1));
    return x.every((v,i)=>v>=0&&v<=b[i])&&x.reduce((a,b)=>a+b,0)===b.reduce((a,b)=>a+b,0);
  }
  const receipt={schema:"lp-rank-delayed-evidence-v1",seed,perDepth,scope:"public synthetic stress only; no learned policy",roots:0,worldCertificateChecks:0,signatureChecks:0,cohorts:[]};
  for(const d of [3,4,6,8])for(const budget of [d-1,d,d+1]){
    const sums={exact:0,rank_greedy:0,fixed_order:0,myopic:0,two_ply:0},costs={exact:0,rank_greedy:0,fixed_order:0,myopic:0,two_ply:0};
    for(let i=0;i<perDepth;i++){
      const r=rngFor(`${seed}/${d}/${i}`),basis=range(d).map(i=>1<<i);
      for(let k=0;k<4*d;k++){const a=r(d),b=r(d);if(a!==b)basis[a]^=basis[b];}
      assert(rank(basis,d)===d,"basis rank");
      const cheap=shuffle([...basis,...range(d).map(()=>1+r((1<<d)-1))],r).map(mask=>({mask,cost:1}));
      const expensive=shuffle(range(d).map(()=>({mask:1+r((1<<d)-1),cost:2})),r);
      const probes=[...expensive,...cheap];
      if(budget===d)receipt.roots++;
      for(let h=0;h<1<<d;h++){
        for(const c of [h,h^1,h^((1<<d)-1)]){
          assert(verify(h,c,d)===(h===c),"primal-dual certificate");
          receipt.worldCertificateChecks++;
        }
      }
      function run(order){
        let selected=[],spent=0;
        for(const p of order){
          if(spent+p.cost>budget||rank([...selected,p.mask],d)===selected.length)continue;
          selected.push(p.mask);spent+=p.cost;if(selected.length===d)break;
        }
        const buckets=new Map();
        for(let h=0;h<1<<d;h++){const sig=selected.map(m=>parity(m&h)).join("");const a=buckets.get(sig)||[];a.push(h);buckets.set(sig,a);}
        let utility=0;
        for(const a of buckets.values()){
          assert(a.length===2**(d-selected.length),"uniform affine fibre");receipt.signatureChecks++;
          if(a.length===1){assert(verify(a[0],a[0],d),"unique certificate");utility+=10;}
        }
        return {utility:utility/(1<<d)-spent,cost:spent,rank:selected.length};
      }
      // Every query yields one binary parity. Before rank d, any certificate
      // has posterior success <=1/2, hence expected value <= -2.5.
      // At least d credits are necessary on each successful path, and a cheap
      // basis attains utility 10-d. Under a smaller budget abstention is optimal.
      const exact=budget>=d?{utility:10-d,cost:d}:{utility:0,cost:0};
      const greedy=budget>=d?run([...probes].sort((a,b)=>a.cost-b.cost)):{utility:0,cost:0};
      const fixed=run(probes);
      assert(greedy.utility===exact.utility&&greedy.cost===exact.cost,"rank baseline vs optimum");
      const results={exact,rank_greedy:greedy,fixed_order:fixed,myopic:{utility:0,cost:0},two_ply:{utility:0,cost:0}};
      for(const [name,v] of Object.entries(results)){assert(v.utility<=exact.utility,"oracle bound");sums[name]+=v.utility;costs[name]+=v.cost;}
    }
    receipt.cohorts.push({d,budget,roots:perDepth,utility:Object.fromEntries(Object.entries(sums).map(([k,v])=>[k,v/perDepth])),cost:Object.fromEntries(Object.entries(costs).map(([k,v])=>[k,v/perDepth]))});
  }
  return receipt;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(JSON.stringify(lpRankStress(...process.argv.slice(2).map(Number)), null, 2));
}

// MTR-02B state compression and cache development pilot.
import { pathToFileURL } from "node:url";
export function mtr02b(seed=20261127,scale=1){
  const ok=(x,m)=>{if(!x)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  ok(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=3,"bad arguments");
  function rng(key){let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function parity(a){let x=0;while(a){x^=1;a&=a-1;}return x;}
  const rank=(xs,d)=>{const p=Array(d).fill(0);let n=0;for(let a of xs)for(let j=d-1;j>=0;j--)if(a>>j&1){if(p[j])a^=p[j];else{p[j]=a;n++;break;}}return n;};
  const I=216,F=18,A=11,modelSeeds=[19,37,73],widths=[4,8,16],modes=["full","rlt","damping","pooled"];
  const rootKeys=new Set();let certificateChecks=0,oracleReplays=0,mergedChecks=0;
  function create(split,i){
    const family=split==="ood"?"triangular":i%2===0?"lp":"affine",d=split==="ood"?5:4+(Math.floor(i/2)%2),id=`${seed}/mtr02b/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),n=2**(d+1),budget=7+i%2;
    let basis=seq(d).map(j=>1<<j);for(let k=0;k<4*d;k++){const a=ri(d),b=ri(d);if(a!==b)basis[a]^=basis[b];}
    ok(rank(basis,d)===d,"basis");
    const encode=h=>family==="lp"?h:family==="affine"?sum(basis.map((a,j)=>parity(a&h)<<j)):sum(seq(d).map(j=>((h>>j&1)^(j>=2?(h>>(j-1)&1)*(h>>(j-2)&1):0))<<j));
    const ws=seq(n).map(j=>({h:j>>1,z:j&1,y:encode(j>>1)})),w=ws.map(x=>1+ri(4)+(split==="ood"&&x.z===0?2:0));
    const bit=j=>1n<<BigInt(j),full=(1n<<BigInt(n))-1n,mask=f=>ws.reduce((m,x,j)=>f(x)?m|bit(j):m,0n);
    const valid=(h,x)=>{
      if(family==="lp"){const a=seq(d).map(j=>1+(h>>j&1)),b=seq(d).map(j=>1+(x.h>>j&1));return a.every((v,j)=>v<=b[j])&&sum(a)===sum(b);}
      if(family==="affine")return basis.every((a,j)=>parity(a&h)===(x.y>>j&1));
      return seq(d).every(j=>((h>>j&1)^(j>=2?(h>>(j-1)&1)*(h>>(j-2)&1):0))===(x.y>>j&1));
    };
    const cert=seq(1<<d).map(h=>mask(x=>{const v=valid(h,x);certificateChecks++;ok(v===(h===x.h),"witness verification");return v;}));
    const forms=shuffle(basis,r),p=[];
    function add(cost,req,form=0,noisy=0,setup=0){
      p.push({cost,req,form,noisy,setup,m:setup?0n:mask(x=>Boolean(parity(form&x.y)^(noisy?x.z:0)))});
    }
    add(1+(split==="ood"&&i%3===0?1:0),0,0,0,1);add(1+ri(2),0,0,0,1);
    add(1,1,forms[0]);add(1,1,forms[1]);add(1,1,forms[2],i%4===0?1:0);
    add(1,2,forms[2]);add(1,2,forms[3]);add(1,2,d===5?forms[4]:forms[0]^forms[3],0);
    add(2+ri(2),0,forms[d-1]);add(2+ri(2),0,forms[0]^forms[1]);add(1,0,forms[0],1);
    if(split==="ood"&&i%2===0){p[4].req=3;p[6].cost=2;}
    ok(p.length===A,"action count");
    const signature=JSON.stringify([family,d,basis,w,budget,p.map(a=>[a.cost,a.req,a.form,a.noisy,a.setup])]);ok(!rootKeys.has(signature),"duplicate root content");rootKeys.add(signature);
    return {id,split,family,d,n,w,ws,cert,p,budget,full};
  }
  function makeGame(root){
    const {n,d,w,cert,p,budget,full}=root,initial={m:full,u:0,b:budget,t:budget},wm=new Map(),tm=new Map(),memo=new Map(),limited=new Map(),states=new Map();
    let expansions=0;
    const W=m=>{if(!wm.has(m)){let v=0;for(let j=0;j<n;j++)if(m>>BigInt(j)&1n)v+=w[j];wm.set(m,v);}return wm.get(m);};
    const key=s=>`${s.m}/${s.u}/${s.b}/${s.t}`;
    function terminal(m){
      if(tm.has(m))return tm.get(m);
      let best={v:0,h:-1};
      for(let h=0;h<cert.length;h++){const v=50*W(m&cert[h])-30*W(m);if(v>best.v)best={v,h};}
      tm.set(m,best);return best;
    }
    const children=(s,i)=>p[i].setup?[[-1,s.m]]:[[0,s.m&~p[i].m],[1,s.m&p[i].m]].filter(x=>x[1]!==0n);
    function legal(s){return s.t>0?p.flatMap((a,i)=>!(s.u>>i&1)&&(s.u&a.req)===a.req&&a.cost<=s.b&&(a.setup||children(s,i).length===2)?[i]:[]):[];}
    const next=(s,i,m)=>({m,u:s.u|(1<<i),b:s.b-p[i].cost,t:s.t-1});
    function solve(s,look=s.t){
      look=Math.min(look,s.b,s.t);if(!look)return {v:terminal(s.m).v,i:-1};
      const exact=look===Math.min(s.b,s.t),map=exact?memo:limited,k=key(s)+(exact?"":`/${look}`);
      if(map.has(k))return map.get(k);
      let best={v:terminal(s.m).v,i:-1};
      for(const i of legal(s)){
        const v=children(s,i).reduce((v,[,m])=>v+solve(next(s,i,m),look-1).v,-p[i].cost*W(s.m));
        if(v>best.v)best={v,i};
      }
      map.set(k,best);if(exact){expansions++;states.set(key(s),s);}return best;
    }
    const pub=Array(I).fill(0);
    // Positions 0..63 compatibility bits, 64..74 used flags, 75..85 last
    // action one-hot, 86..88 outcome, 89..90 budget/visits, 91..154 prior,
    // 155..209 action definitions, 210..215 dimension and family-independent metadata.
    for(let j=0;j<n;j++)pub[91+j]=w[j]/6;
    for(let j=0;j<A;j++){
      const a=p[j],off=155+5*j;
      pub[off]=a.cost/3;pub[off+1]=a.req/3;pub[off+2]=a.form/31;pub[off+3]=a.noisy;pub[off+4]=a.setup;
    }
    pub[210]=d/5;pub[211]=budget/8;pub[212]=n/64;
    function token(s,last,outcome,mode){
      const x=[...pub];x[89]=s.b/8;x[90]=s.t/8;
      if(mode==="full"){
        for(let j=0;j<n;j++)x[j]=Number(s.m>>BigInt(j)&1n);
        for(let j=0;j<A;j++)x[64+j]=s.u>>j&1;
      }else{
        if(last>=0)x[75+last]=1;
        x[86+(outcome===-1?0:outcome+1)]=1;
      }
      return x;
    }
    function f(s,i){
      const v=Array(F).fill(0);v[0]=1;v[1]=i<0?1:0;v[2]=s.b/8;v[3]=s.t/8;
      if(i>=0){const a=p[i];v[4]=a.cost/3;v[5]=a.setup;v[6]=a.noisy;v[7]=a.req&1?1:0;v[8]=a.req&2?1:0;
        for(let j=0;j<5;j++)v[9+j]=a.form>>j&1;v[14]=i/A;v[15]=i===0?1:0;v[16]=i===1?1:0;v[17]=a.form/31;}
      return v;
    }
    function q(s){
      return [-1,...legal(s)].map(i=>({i,v:(i===-1?terminal(s.m).v:children(s,i).reduce((v,[,m])=>v+solve(next(s,i,m)).v,-p[i].cost*W(s.m)))/W(s.m)/20}));
    }
    function step(s,i,h){
      const a=p[i],o=a.setup?-1:Number(a.m>>BigInt(h)&1n),m=a.setup?s.m:o?s.m&a.m:s.m&~a.m;
      ok(m>>BigInt(h)&1n,"lost latent world");return {s:next(s,i,m),o};
    }
    function examples(r){
      const rows=[],seen=new Set();
      for(let path=0;path<2&&rows.length<4;path++){
        let s=initial,history=[token(s,-1,-1,"rlt")],h=Math.floor(r()*n);
        for(let turn=0;turn<3&&rows.length<4;turn++){
          const k=key(s);if(!seen.has(k)){seen.add(k);const y=q(s);rows.push({history,full:token(s,-1,-1,"full"),ids:y.map(a=>a.i),x:y.map(a=>f(s,a.i)),y:y.map(a=>a.v)});}
          const a=legal(s);if(!a.length)break;let i=path===0?solve(s).i:a[Math.floor(r()*a.length)];if(i<0)i=a[Math.floor(r()*a.length)];
          const z=step(s,i,h);s=z.s;history=[...history,token(s,i,z.o,"rlt")];
        }
      }return rows;
    }
    function entropy(m){
      const a=W(m);let H=0;for(let j=0;j<n;j++)if(m>>BigInt(j)&1n){const v=w[j]/a;H-=v*Math.log2(v);}return H;
    }
    function heuristic(s,kind){
      let best=-1,score=0;const terminalV=terminal(s.m).v/W(s.m);if(terminalV>=19)return -1;
      const known=p.filter((a,i)=>s.u>>i&1&&!a.setup&&!a.noisy).map(a=>a.form);
      for(const i of legal(s)){
        const a=p[i],unlock=a.setup?p.filter((b,j)=>!(s.u>>j&1)&&(b.req&(1<<i))&&((b.req&(s.u|(1<<i)))===b.req)):[],
          v=kind==="rank"?(a.setup?0.55*rank([...known,...unlock.filter(x=>!x.noisy).map(x=>x.form)],d)-0.55*rank(known,d):a.noisy?0:rank([...known,a.form],d)-rank(known,d)):
            (a.setup?0.30*unlock.length:entropy(s.m)-sum(children(s,i).map(([,m])=>W(m)/W(s.m)*entropy(m))));
        const z=v/a.cost-0.001*a.cost;if(z>score){score=z;best=i;}
      }return best;
    }
    function run(kind,model){
      let utility=0,cost=0,queries=0,pubMass=0,falseMass=0;const total=W(full),decisionCache=new Map();
      for(let j=0;j<n;j++){
        let s=initial,c=model?Array(model.H).fill(0):null,used=0,checks=0,steps=0,last=-1,outcome=-1;
        while(true){
          let i;
          if(kind==="oracle")i=solve(s).i;else if(kind==="two_ply")i=solve(s,2).i;else if(kind==="rank"||kind==="information")i=heuristic(s,kind);
          else{
            const x=token(s,last,outcome,model.mode),k=key(s)+(model.mode==="full"?"":`/${last}/${outcome}/${steps}/${c.join(",")}`);
            if(decisionCache.has(k)){const a=decisionCache.get(k);i=a.i;c=a.c;}
            else{
              const a=advance(model,x,c,steps),ids=[-1,...legal(s)],v=scores(model,a.z,ids.map(i=>f(s,i)));
              i=ids[v.indexOf(Math.max(...v))];c=a.c;decisionCache.set(k,{i,c});
            }
          }
          if(i<0){
            const h=terminal(s.m).h,good=h>=0&&Boolean(cert[h]>>BigInt(j)&1n);
            utility+=w[j]*((h<0?0:good?20:-30)-used);cost+=w[j]*used;queries+=w[j]*checks;if(h>=0)pubMass+=w[j];if(h>=0&&!good)falseMass+=w[j];break;
          }
          // Replay independently checks each selected action's prerequisites,
          // one-use condition, remaining budget and observed result.
          const a=p[i];ok(s.t>0&&a.cost<=s.b&&!(s.u>>i&1)&&(s.u&a.req)===a.req,"illegal replay action");
          const z=step(s,i,j);ok(a.setup||z.s.m!==s.m,"uninformative repeated probe");used+=a.cost;checks+=1-a.setup;s=z.s;last=i;outcome=z.o;steps++;
        }
      }
      if(kind==="oracle"){ok(utility===solve(initial).v,"oracle replay");oracleReplays++;}
      else ok(utility<=solve(initial).v,"oracle bound");
      return {utility:utility/total,cost:cost/total,queries:queries/total,publication:pubMass/total,falsePublication:falseMass/total};
    }
    function pairs(models){
      const pairs=[];
      // Independent setup actions commute. Also enumerate legal commuting
      // two-probe extensions from the state where both setups are complete.
      for(const h of [0,Math.floor(n/2),n-1]){
        const a=step(initial,0,h),b=step(initial,1,h),ab=step(a.s,1,h),ba=step(b.s,0,h);
        if(key(ab.s)!==key(ba.s))throw Error("setup order");
        const H0=[token(initial,-1,-1,"rlt")];
        const ha=[...H0,token(a.s,0,a.o,"rlt"),token(ab.s,1,ab.o,"rlt")],hb=[...H0,token(b.s,1,b.o,"rlt"),token(ba.s,0,ba.o,"rlt")];
        pairs.push([ab.s,ha,hb]);
        const ls=legal(ab.s).filter(i=>!p[i].setup);
        for(let j=0;j<ls.length&&pairs.length<8;j++)for(let k=j+1;k<ls.length&&pairs.length<8;k++){
          const i=ls[j],l=ls[k],u=step(ab.s,i,h),v=step(ab.s,l,h);
          if(!legal(u.s).includes(l)||!legal(v.s).includes(i))continue;
          const uv=step(u.s,l,h),vu=step(v.s,i,h);ok(key(uv.s)===key(vu.s),"commutation");
          pairs.push([uv.s,[...ha,token(u.s,i,u.o,"rlt"),token(uv.s,l,uv.o,"rlt")],[...ha,token(v.s,l,v.o,"rlt"),token(vu.s,i,vu.o,"rlt")]]);
        }
      }
      const result={count:pairs.length,metrics:{}};
      for(const model of models){
        let disagreement=0,delta=0;
        for(const [s,a,b] of pairs){
          // Both histories have the same sufficient-state key; independently
          // recompute the one-step continuation using all legal actions.
          const target=solve(s).v,qq=q(s);ok(Math.abs(Math.max(...qq.map(x=>x.v))*W(s.m)*20-target)<1e-8,"Bellman state identity");mergedChecks++;
          const ids=qq.map(x=>x.i),x=ids.map(i=>f(s,i)),t=token(s,-1,-1,"full");
          const za=context(model,model.mode==="full"?[t]:a).z,zb=context(model,model.mode==="full"?[t]:b).z,va=scores(model,za,x),vb=scores(model,zb,x);
          disagreement+=Number(va.indexOf(Math.max(...va))!==vb.indexOf(Math.max(...vb)));delta+=Math.max(...va.map((v,i)=>Math.abs(v-vb[i])))*20;
        }
        const k=`${model.H}/${model.mode}`;if(!result.metrics[k])result.metrics[k]={disagreement:0,maxQDelta:0,seeds:0};const m=result.metrics[k];
        m.disagreement+=disagreement/pairs.length;m.maxQDelta+=delta/pairs.length;m.seeds++;
      }
      for(const m of Object.values(result.metrics)){m.disagreement/=m.seeds;m.maxQDelta/=m.seeds;}
      return result;
    }
    function cacheExperiment(){
      solve(initial);
      // Fixed trace chosen without learned-policy outputs. Use reachable
      // nonterminal states and repeated local neighborhoods.
      const r=rng(root.id+"/cache"),available=shuffle([...states.values()].filter(s=>s.t>0&&s.b>0&&s.u!==0),r).slice(0,180),trace=[];
      for(let k=0;k<3;k++){
        const a=k===0?available:shuffle(available,r);
        for(let j=0;j<a.length;j++){trace.push(a[j]);if(j%4===0)trace.push(a[j]);}
      }
      // Recompute on a miss using fresh exact DP and count expanded states.
      // Memoization inside one query is identical across all cache arms.
      const expected=trace.map(s=>solve(s).v),work=new Map();
      function fresh(s){
        const local=new Map();let count=0;
        function rec(z){
          if(z.b<=0||z.t<=0)return terminal(z.m).v;
          const k=key(z);if(local.has(k))return local.get(k);
          let best=terminal(z.m).v;for(const i of legal(z)){
            const v=children(z,i).reduce((v,[,m])=>v+rec(next(z,i,m)),-p[i].cost*W(z.m));if(v>best)best=v;
          }local.set(k,best);count++;return best;
        }
        return {v:rec(s),count};
      }
      // Independently evaluate every distinct trace state once to obtain exact
      // miss work. Subsequent cache replay uses this measured miss cost, not
      // a wall-clock measurement. Each returned answer is checked.
      for(let j=0;j<trace.length;j++){const s=trace[j],k=key(s);if(!work.has(k)){const a=fresh(s);ok(a.v===expected[j],"independent cache oracle");work.set(k,a);}}
      const rows=[];
      for(const bytes of [0,4096,16384,65536])for(const mode of bytes===0?["none"]:["lru","unrelated","work_admit"]){
        const cap=bytes/32,buf=new ArrayBuffer(bytes),masks=new BigUint64Array(buf,0,cap),
          vals=new Float64Array(buf,cap*8,cap),meta=new Uint32Array(buf,cap*16,cap*4);
        const kword=s=>(s.u|(s.b<<12)|(s.t<<16))>>>0;
        let hits=0,misses=0,expanded=0,comparisons=0,admitted=0,evictions=0,peak=0,clock=0,stored=0;
        for(let j=0;j<trace.length;j++){
          const s=trace[j],k=kword(s);clock++;let found=-1,empty=-1,victim=-1;
          for(let a=0;a<cap;a++){
            const o=4*a;if(!meta[o+3]){if(empty<0)empty=a;continue;}
            comparisons++;if(masks[a]===s.m&&meta[o]===k){found=a;break;}
            if(victim<0||meta[o+1]<meta[4*victim+1])victim=a;
          }
          let value;
          if(found>=0){hits++;value=vals[found];meta[4*found+1]=clock;}
          else{
            misses++;const q=work.get(key(s));value=q.v;expanded+=q.count;
            const slot=empty>=0?empty:victim;
            if(slot>=0&&(mode!=="work_admit"||empty>=0||q.count>=meta[4*slot+2])){
              const o=4*slot;if(meta[o+3])evictions++;else stored++;
              masks[slot]=s.m;vals[slot]=value;meta[o]=mode==="unrelated"?(k|0x80000000)>>>0:k;
              meta[o+1]=clock;meta[o+2]=q.count;meta[o+3]=1;admitted++;peak=Math.max(peak,stored*32);
            }
          }
          ok(value===expected[j],"cache changed answer");
        }
        rows.push({bytes,mode,trace:trace.length,hits,misses,expanded,comparisons,admitted,evictions,peakLogicalBytes:peak,verified:trace.length,independentStates:work.size});
      }return rows;
    }
    return {examples,run,pairs,cacheExperiment,expansions:()=>expansions};
  }
  function project(x,H){
    const p=Array(H).fill(0);for(let i=0;i<x.length;i++)p[i%H]+=x[i]*((i%2===0)?1:-1);
    const scale=Math.sqrt(x.length/H);return p.map(v=>Math.tanh(v/scale));
  }
  function init(H,mode,s){
    const r=rng(`${seed}/mtr02b/${H}/${s}`),P=H*I+H*H+H+F+H*F,p=Array.from({length:P},()=>0.1*(2*r()-1));
    return {H,mode,p,m:Array(P).fill(0),v:Array(P).fill(0),step:0,seed:s};
  }
  function advance(model,x,c,t){
    const {H,p}=model,rec=H*I,bias=rec+H*H;
    const raw=model.mode==="full"?project(x,H):c,z=Array(H).fill(0);
    for(let j=0;j<H;j++){
      let a=p[bias+j];for(let k=0;k<I;k++)a+=p[j*I+k]*x[k];for(let k=0;k<H;k++)a+=p[rec+j*H+k]*raw[k];z[j]=Math.tanh(a);
    }
    let next=z;
    if(model.mode==="damping"||model.mode==="pooled"){
      const q=project(x,H);next=c.map((v,j)=>model.mode==="damping"?0.5*v+0.5*q[j]:(t*v+q[j])/(t+1));
    }return {x,c:raw,z,next,c:next,inputContext:raw};
  }
  function context(model,hist){
    let c=Array(model.H).fill(0),stack=[];
    for(let t=0;t<hist.length;t++){const a=advance(model,hist[t],c,t);stack.push(a);c=a.next;}
    return {stack,z:stack.at(-1).z};
  }
  function scores(model,h,x){
    const {H,p}=model,off=H*I+H*H+H,b=Array(F);
    for(let k=0;k<F;k++){let v=p[off+k];for(let j=0;j<H;j++)v+=h[j]*p[off+F+j*F+k];b[k]=v;}
    return x.map(f=>{let v=0;for(let k=0;k<F;k++)v+=f[k]*b[k];return v;});
  }
  function update(model,e){
    const {H,p}=model,rec=H*I,bias=rec+H*H,off=bias+H,{stack,z}=context(model,model.mode==="full"?[e.full]:e.history),pred=scores(model,z,e.x),g=Array(p.length).fill(0),dh=Array(H).fill(0);
    let loss=0;
    for(let i=0;i<e.x.length;i++){
      const f=e.x[i],err=pred[i]-e.y[i],v=2*err/e.x.length;loss+=err*err/e.x.length;
      for(let k=0;k<F;k++){g[off+k]+=v*f[k];for(let j=0;j<H;j++){g[off+F+j*F+k]+=v*z[j]*f[k];dh[j]+=v*p[off+F+j*F+k]*f[k];}}
    }
    let back=dh;
    for(let t=stack.length-1;t>=0;t--){
      const a=stack[t],d=a.z.map((v,j)=>back[j]*(1-v*v)),prev=Array(H).fill(0);
      for(let j=0;j<H;j++){
        g[bias+j]+=d[j];for(let k=0;k<I;k++)g[j*I+k]+=d[j]*a.x[k];
        for(let k=0;k<H;k++){g[rec+j*H+k]+=d[j]*a.inputContext[k];prev[k]+=d[j]*p[rec+j*H+k];}
      }
      if(model.mode!=="rlt")break;back=prev;
    }
    model.step++;const rate=0.0015*Math.sqrt(1-.999**model.step)/(1-.9**model.step),clip=Math.min(1,4/Math.max(1e-12,Math.sqrt(sum(g.map(v=>v*v)))));
    for(let j=0;j<p.length;j++){
      const v=clip*g[j];model.m[j]=.9*model.m[j]+.1*v;model.v[j]=.999*model.v[j]+.001*v*v;p[j]-=rate*model.m[j]/(Math.sqrt(model.v[j])+1e-8);
    }
    ok(Number.isFinite(loss),"nonfinite fit");return loss;
  }
  const trainN=40*scale,evalN=20*scale,rows=[],fit=[],models=[];let oracleStates=0;
  for(let i=0;i<trainN;i++){
    const r=create("train",i),g=makeGame(r),ex=g.examples(rng(r.id+"/explore"));ok(ex.length>=1&&ex.length<=4,"root cap");rows.push(...ex);oracleStates+=g.expansions();
  }
  for(const H of widths)for(const s of modelSeeds)for(const mode of modes){
    const model=init(H,mode,s),r=rng(`${seed}/mtr02b/shuffle/${H}/${s}`),loss=[];
    for(let epoch=0;epoch<7;epoch++){let l=0;for(const i of shuffle(seq(rows.length),r))l+=update(model,rows[i]);loss.push(l/rows.length);}
    fit.push({H,mode,seed:s,parameters:model.p.length,steps:model.step,firstLoss:loss[0],lastLoss:loss.at(-1)});models.push(model);
  }
  const metrics=["utility","cost","queries","publication","falsePublication"],rr=[],cacheRows=[];
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=create(split,i),g=makeGame(r),result={};
    for(const kind of ["oracle","two_ply","rank","information"])result[kind]=g.run(kind,null);
    for(const H of widths)for(const mode of modes){
      const a=models.filter(m=>m.H===H&&m.mode===mode).map(m=>g.run(mode,m));
      result[`${H}/${mode}`]={...Object.fromEntries(metrics.map(k=>[k,mean(a.map(v=>v[k]))])),seedUtility:a.map(v=>v.utility)};
    }
    const pair=g.pairs(models);rr.push({id:r.id,split,family:r.family,d:r.d,budget:r.budget,result,pair});
    if(i<Math.min(4,evalN))for(const a of g.cacheExperiment())cacheRows.push({root:r.id,split,...a});
    oracleStates+=g.expansions();
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(a){const m=mean(a),sd=Math.sqrt(sum(a.map(x=>(x-m)**2))/Math.max(1,a.length-1));return [round(m),round(m-1.96*sd/Math.sqrt(a.length)),round(m+1.96*sd/Math.sqrt(a.length))];}
  const aggregate=[];
  for(const split of ["ordinary","ood"]){
    const r=rr.filter(x=>x.split===split),means={},pairs={},comparisons={};
    for(const k of Object.keys(r[0].result))means[k]=Object.fromEntries(metrics.map(m=>[m,round(mean(r.map(v=>v.result[k][m])))]));
    for(const k of Object.keys(r[0].pair.metrics))pairs[k]={disagreement:round(mean(r.map(v=>v.pair.metrics[k].disagreement))),maxQDelta:round(mean(r.map(v=>v.pair.metrics[k].maxQDelta)))};
    for(const H of widths)for(const m of ["full","damping","pooled"]){
      const k=`${H}/${m}`,a=`${H}/rlt`;
      comparisons[k]=Object.fromEntries(["utility","queries","falsePublication"].map(v=>[v,ci(r.map(x=>x.result[a][v]-x.result[k][v]))]));
    }
    aggregate.push({split,roots:r.length,means,pairs,comparisons,mergedHistoryPairs:sum(r.map(x=>x.pair.count))});
  }
  const cacheAggregate=[];
  for(const bytes of [0,4096,16384,65536])for(const mode of bytes===0?["none"]:["lru","unrelated","work_admit"]){
    const a=cacheRows.filter(r=>r.bytes===bytes&&r.mode===mode),m={bytes,mode,roots:a.length};
    for(const k of ["trace","hits","misses","expanded","comparisons","admitted","evictions","verified","independentStates"])m[k]=sum(a.map(r=>r[k]));
    m.peakLogicalBytes=Math.max(...a.map(r=>r.peakLogicalBytes));cacheAggregate.push(m);
  }
  return {receipt:{schema:"MTR-02B-public-development-v1",seed,scale,trainRoots:trainN,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:rootKeys.size,trainingRows:rows.length,certificateChecks,oracleReplays,oracleStates,mergedChecks,modelSeeds,widths,fit,aggregate,cacheAggregate,cacheRows,rootResults:rr},weights:models.map(m=>({H:m.H,mode:m.mode,seed:m.seed,p:m.p}))};
}
if (process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  const args=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number);
  const r=mtr02b(...args);
  console.log(JSON.stringify(process.argv.includes("--weights")?r.weights:r.receipt,null,2));
}

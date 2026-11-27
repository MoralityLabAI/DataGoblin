// Standalone native benchmark of weighted-mass versus full claim aggregation.
// CLI: node mtr02h_mass_native.mjs full|split [seed] [roots]
const mode=process.argv[2], seed=Number(process.argv[3]||20261127), roots=Number(process.argv[4]||24);
if(!['full','split'].includes(mode)||!Number.isSafeInteger(seed)||!Number.isInteger(roots)||roots<1)throw Error('arguments');
const N=128,BITS=Array.from({length:N},(_,i)=>1n<<BigInt(i)),BI=new Map(BITS.map((b,i)=>[b,i]));
let z=seed>>>0||1;function rnd(n){z^=z<<13;z^=z>>>17;z^=z<<5;return(z>>>0)%n;}
let checksum=0,expectedChecksum=0,queries=0,worldOps=0,massOps=0,terminalStates=0,massStates=0,elapsed=0,cpu=0;
for(let root=0;root<roots;root++){
  const w=BITS.map(()=>1+rnd(4)),shift=rnd(128),labels=BITS.map((_,h)=>{const a=h^shift,b=a>>3,p=a&7;return[h,4*b+(p>>1),b,(p>>2)^((p>>1&1)*(p&1))];});
  const masks=[],seen=new Set();while(masks.length<256){let m=0n,n=1+rnd(96);for(let k=0;k<n;k++)m|=BITS[rnd(N)];if(!seen.has(m)){seen.add(m);masks.push(m);}}
  const requests=[];for(let i=0;i<1024;i++){const m=masks[rnd(masks.length)];requests.push([m,rnd(5)===0]);}
  const oracle=new Map();for(const m of masks){let mass=0,bins=[Array(128).fill(0),Array(64).fill(0),Array(16).fill(0),[0,0]];for(let h=0;h<N;h++)if(m&BITS[h]){mass+=w[h];for(let k=0;k<4;k++)bins[k][labels[h][k]]+=w[h];}const max=bins.map(x=>Math.max(...x));oracle.set(m,{mass,value:Math.max(0,90*max[0]-54*mass,60*max[1]-36*mass,38*max[2]-23*mass,20*max[3]-12*mass)});}
  for(const [m,terminal] of requests){const x=oracle.get(m);expectedChecksum+=terminal?x.value:x.mass;}
  const sm=new Map(),mm=new Map();
  function stats(m){if(sm.has(m))return sm.get(m);const bins=[new Map(),new Map(),new Map(),new Map()],max=[0,0,0,0],cached=mode==='split'?mm.get(m):null;let mass=cached?.mass||0,n=cached?.n||0;for(let t=m;t;t&=t-1n){const h=BI.get(t&-t),v=w[h];worldOps+=cached?4:5;if(!cached){mass+=v;n++;}for(let k=0;k<4;k++){const j=labels[h][k],u=(bins[k].get(j)||0)+v;bins[k].set(j,u);max[k]=Math.max(max[k],u);}}const q={mass,n,value:Math.max(0,90*max[0]-54*mass,60*max[1]-36*mass,38*max[2]-23*mass,20*max[3]-12*mass)};sm.set(m,q);if(mode==='split'&&!cached)mm.set(m,{mass,n});return q;}
  function mass(m){if(mode==='full')return stats(m).mass;if(mm.has(m))return mm.get(m).mass;let mass=0,n=0;for(let t=m;t;t&=t-1n){mass+=w[BI.get(t&-t)];n++;massOps++;}mm.set(m,{mass,n});return mass;}
  const t0=process.hrtime.bigint(),c0=process.cpuUsage();for(const [m,terminal] of requests){checksum+=terminal?stats(m).value:mass(m);queries++;}const dt=process.cpuUsage(c0);cpu+=dt.user+dt.system;elapsed+=Number(process.hrtime.bigint()-t0)/1e6;
  for(const [m,x] of sm){const o=oracle.get(m);if(x.mass!==o.mass||x.value!==o.value)throw Error('terminal differential');}for(const [m,x] of mm)if(x.mass!==oracle.get(m).mass)throw Error('mass differential');terminalStates+=sm.size;massStates+=mm.size;
}
if(checksum!==expectedChecksum)throw Error('checksum');console.log(JSON.stringify({mode,seed,roots,queries,checksum,expectedChecksum,worldOps,massOps,terminalStates,massStates,elapsedMs:elapsed,cpuMicros:cpu,maxRSSKiB:process.resourceUsage().maxRSS,node:process.version,platform:process.platform,arch:process.arch}));

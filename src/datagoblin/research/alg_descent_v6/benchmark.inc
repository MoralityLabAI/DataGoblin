// ALG-DESCENT-06: exact root-local canonical IDs, matched allocation and slot controls.
#include <numeric>
#include <cstddef>
struct InternInfo {U32 unique; U64 ns; U64 scratch; U64 node_capacity;};
InternInfo intern(Root& r){
    auto t=Clock::now(); std::vector<U16> order(r.nodes.size()); std::iota(order.begin(),order.end(),U16(0));
    std::sort(order.begin(),order.end(),[&](U16 a,U16 b){const auto& x=r.nodes[a].poly;const auto& y=r.nodes[b].poly;if(x.len!=y.len)return x.len<y.len;if(x.bytes!=y.bytes)return x.bytes<y.bytes;return a<b;});
    U32 unique=0; U16 rep=0; bool first=true;
    for(auto i:order){if(first||!(r.nodes[i].poly==r.nodes[rep].poly)){rep=i;unique++;first=false;}r.nodes[i].rep=rep;}
    U64 ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();
    for(auto& n:r.nodes) require(n.poly==r.nodes[n.rep].poly&&r.nodes[n.rep].rep==n.rep,"representative equality");
    return {unique,ns,order.capacity()*sizeof(U16),r.nodes.capacity()};
}
struct IdSlot{U32 id=UINT32_MAX,stamp=0,saved=0;};
struct IdCache{std::unique_ptr<IdSlot[]> slots;U32 count=0,tick=0,used=0,peak=0;bool utility=false;
    IdCache(U32 cap,bool u):slots(cap/sizeof(IdSlot)?new IdSlot[cap/sizeof(IdSlot)]():nullptr),count(cap/sizeof(IdSlot)),utility(u){}
    bool hit(U16 k){for(U32 i=0;i<count;i++)if(slots[i].id==k){slots[i].stamp=++tick;return true;}return false;}
    template<typename Count>void insert(U16 k,U32 paid,Count pending){if(!count)return;U32 idx=count;for(U32 i=0;i<count;i++){if(slots[i].id==k)return;if(slots[i].id==UINT32_MAX&&idx==count)idx=i;}if(idx<count){used++;peak=std::max(peak,used);}else{auto value=[&](U32 i){return U64(pending(slots[i].id))*slots[i].saved;};idx=0;for(U32 i=1;i<count;i++){if(utility){auto a=value(i),b=value(idx);if(a<b||(a==b&&slots[i].stamp<slots[idx].stamp))idx=i;}else if(slots[i].stamp<slots[idx].stamp)idx=i;}if(utility&&U64(pending(k))*(paid-1)<value(idx))return;}slots[idx]={k,++tick,paid-1};}
};
struct IdRun{const Root& r;IdCache cache;Pending pending;Result out;
    IdRun(const Root& root,U32 cap,bool u):r(root),cache(cap,u){}
    bool visit(U16 id){const auto& node=r.nodes[id];if(node.poly.len==2)return canonical(node.poly,r.p);if(cache.count&&out.work<3*r.n*r.n&&cache.hit(node.rep)){out.work++;out.hits++;return true;}const U32 begin=out.work;for(auto& a:node.actions){auto c=cost(a);if(out.work+c>3*r.n*r.n||out.calls>=8*r.n)return false;auto [ok,paid]=verify(node.poly,a,r.p);require(c==paid,"unexpected verifier meter");out.work+=paid;out.calls++;if(!ok){out.rejects++;continue;}for(auto& child:a.children)require(child.poly==r.nodes[child.id].poly,"valid certificate disagrees with bank child");if(cache.count)for(auto it=a.children.rbegin();it!=a.children.rend();++it)if(it->poly.len>2)pending.push(r.nodes[it->id].rep);for(auto& child:a.children){if(cache.count&&child.poly.len>2)pending.pop(r.nodes[child.id].rep);if(!visit(child.id))return false;}cache.insert(node.rep,out.work-begin,[&](U32 key){U32 n=0;for(U32 j=0;j<pending.len;j++)n+=pending.ids[j]==key;return n;});return true;}return false;}
    Result execute(){out.closed=visit(U16(r.nodes.size()-1));out.peak=cache.peak;out.pending_peak=pending.peak;require(!out.closed||pending.len==0,"unclosed frontier");require(out.work<=3*r.n*r.n&&out.calls<=8*r.n&&out.peak<=cache.count,"budget");return out;}
};
struct Arm6{U32 cap;bool utility;U8 kind;}; // 0 packed, 1 interned equal bytes, 2 interned equal slots.
int main(int argc,char** argv){try{
    selftest(); require(argc>=2,"usage: native BANK [repeats]");
    int repeats=argc>2?std::stoi(argv[2]):61;require(repeats>0&&repeats<=1001,"bad repeats");
    require(sizeof(Node)==sizeof(LegacyNode)&&alignof(Node)==alignof(LegacyNode),"ABI has no room for representative; charge sidecar and redesign");
    require(sizeof(Slot)==36&&sizeof(IdSlot)==12&&sizeof(Cache)==sizeof(IdCache),"unexpected ABI slot layout");
    auto start=Clock::now();auto roots=load(argv[1]);auto loadns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    std::cerr<<"legacy_node="<<sizeof(LegacyNode)<<" node="<<sizeof(Node)<<" node_align="<<alignof(Node)<<" rep_offset="<<offsetof(Node,rep)<<" action_offset="<<offsetof(Node,actions)<<" packed_slot="<<sizeof(Slot)<<" id_slot="<<sizeof(IdSlot)<<" control="<<sizeof(Cache)<<" pending="<<sizeof(Pending)<<" load_ns="<<loadns<<" roots="<<roots.size()<<"\n";
    std::vector<Arm6> arms{{0,false,0}};for(U32 c:{36,72,108,252,504,1008})for(bool u:{false,true})for(U8 k:{0,1,2})arms.push_back({c,u,k});
    std::cout<<"seed,n,cap,policy,kind,slots,allocated,closed,work,calls,rejects,hits,peak,pending_peak,median_ns,first_ns,unique,intern_ns,scratch_bytes,node_capacity\n";
    for(auto& r:roots){auto info=intern(r);std::vector<std::vector<U64>> timings(arms.size());std::vector<Result> results(arms.size());std::vector<int> order(arms.size());std::iota(order.begin(),order.end(),0);std::mt19937 rng(r.seed);
        for(int rep=0;rep<repeats;rep++){std::shuffle(order.begin(),order.end(),rng);for(int i:order){auto a=arms[i];U32 actual=a.kind==2?a.cap/sizeof(Slot)*sizeof(IdSlot):a.cap;auto t0=Clock::now();Result result;if(a.kind){IdRun run(r,actual,a.utility);result=run.execute();}else{Run run(r,actual,a.utility);result=run.execute();}auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count();timings[i].push_back(ns);if(rep==0)results[i]=result;else require(results[i].same(result),"nondeterministic receipt");}}
        for(size_t i=0;i<arms.size();i++){auto& ts=timings[i];auto first=ts[0];std::sort(ts.begin(),ts.end());auto a=arms[i];auto res=results[i];U32 size=a.kind?sizeof(IdSlot):sizeof(Slot);U32 actual=a.kind==2?a.cap/sizeof(Slot)*sizeof(IdSlot):a.cap;auto slots=actual/size;std::cout<<r.seed<<','<<r.n<<','<<a.cap<<','<<(a.utility?"pending":"lru")<<','<<(a.kind==0?"packed":a.kind==1?"interned":"matched")<<','<<slots<<','<<slots*size<<','<<res.closed<<','<<res.work<<','<<res.calls<<','<<res.rejects<<','<<res.hits<<','<<res.peak<<','<<res.pending_peak<<','<<ts[ts.size()/2]<<','<<first<<','<<info.unique<<','<<info.ns<<','<<info.scratch<<','<<info.node_capacity<<'\n';}
    }
    struct rusage ru{};getrusage(RUSAGE_SELF,&ru);std::cerr<<"max_rss_kib="<<ru.ru_maxrss<<" repeats="<<repeats<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

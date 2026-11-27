// ALG-DESCENT-05: native exact public-witness routing, C++17, no dependencies.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
using U8=uint8_t; using U16=uint16_t; using U32=uint32_t; using U64=uint64_t;
using Clock=std::chrono::steady_clock;
void require(bool ok,const char* msg){if(!ok) throw std::runtime_error(msg);}
constexpr int MAXN=48, PACK=(MAXN+2)/2;
struct Key {
    std::array<U8,PACK> bytes{}; U8 len=0;
    U8 at(int i) const {return (bytes[i/2]>>(4*(i%2)))&15;}
    void set(int i,U8 c){bytes[i/2]=(bytes[i/2]&~(15<<(4*(i%2))))|(c<<(4*(i%2)));}
    bool operator==(const Key& o) const{return len==o.len&&bytes==o.bytes;}
};
bool canonical(const Key& k,int p){if(k.len<2||k.len>MAXN+1||k.at(k.len-1)!=1)return false; for(int i=0;i<k.len;i++)if(k.at(i)>=p)return false; return true;}
struct Child{U16 id;Key poly;};
struct Action{std::vector<Child> children;};
struct Node{Key poly;std::vector<Action> actions;};
struct Root{U32 seed,n,p;std::vector<Node> nodes;};
U32 readint(std::istream& in,int size){U32 v=0;for(int i=0;i<size;i++){int c=in.get();require(c!=EOF,"truncated bank");v|=U32(c)<<(8*i);}return v;}
Key readkey(std::istream& in,int p){Key k;k.len=readint(in,1);require(k.len>=2&&k.len<=MAXN+1,"bad key length");for(int i=0;i<k.len;i++){auto c=readint(in,1);require(c<U32(p),"bad coefficient");k.set(i,c);}require(canonical(k,p),"nonmonic key");return k;}
std::vector<Root> load(const char* path){std::ifstream in(path,std::ios::binary);require(bool(in),"bank open failed");require(readint(in,4)==0x42354441,"bad magic");auto nr=readint(in,4);require(nr>0&&nr<=1000,"bad root count");std::vector<Root> roots;roots.reserve(nr);for(U32 j=0;j<nr;j++){Root r;r.seed=readint(in,4);r.n=readint(in,1);r.p=readint(in,1);auto nn=readint(in,2);require(r.n>=2&&r.n<=MAXN&&(r.p==2||r.p==3||r.p==5||r.p==7||r.p==11)&&nn==r.n*(r.n+1)/2,"bad root");r.nodes.reserve(nn);for(U32 i=0;i<nn;i++){Node node;node.poly=readkey(in,r.p);auto na=readint(in,1);require(na<=8&&bool(na)==(node.poly.len>2),"bad action count");for(U32 a=0;a<na;a++){Action act;auto nc=readint(in,1);require(nc>=2&&nc<=8,"bad child count");for(U32 c=0;c<nc;c++){auto id=readint(in,2);require(id<i,"nondecreasing child id");auto k=readkey(in,r.p);require(k.len==r.nodes[id].poly.len&&k.len<node.poly.len,"bad child degree");act.children.push_back({U16(id),k});}node.actions.push_back(std::move(act));}r.nodes.push_back(std::move(node));}require(r.nodes.back().poly.len==r.n+1,"bad root degree");roots.push_back(std::move(r));}require(in.get()==EOF,"trailing bank bytes");return roots;}
U32 cost(const Action& a){U32 c=0,n=1;for(auto& ch:a.children){c+=n*ch.poly.len;n+=ch.poly.len-1;}return c;}
std::pair<bool,U32> verify(const Key& parent,const Action& a,int p){if(!canonical(parent,p)||parent.len<3||a.children.size()<2||a.children.size()>8)return {false,0};int d=0;for(auto& ch:a.children){if(!canonical(ch.poly,p)||ch.poly.len>=parent.len)return {false,0};d+=ch.poly.len-1;}if(d!=parent.len-1)return {false,0};std::array<U8,MAXN+1> acc{},next{};acc[0]=1;int n=1;U32 paid=0;for(auto& ch:a.children){next.fill(0);auto& q=ch.poly;for(int i=0;i<n;i++)for(int j=0;j<q.len;j++)next[i+j]=(next[i+j]+acc[i]*q.at(j))%p;paid+=n*q.len;n+=q.len-1;acc=next;}for(int i=0;i<n;i++)if(acc[i]!=parent.at(i))return {false,paid};return {true,paid};}
struct Slot{Key key;U32 stamp=0,saved=0;};
struct Cache{std::unique_ptr<Slot[]> slots;U32 count=0,tick=0,used=0,peak=0;bool utility=false;
    Cache(U32 cap,bool u):slots(cap/sizeof(Slot)?new Slot[cap/sizeof(Slot)]():nullptr),count(cap/sizeof(Slot)),utility(u){}
    bool hit(const Key& k){for(U32 i=0;i<count;i++)if(slots[i].key==k){slots[i].stamp=++tick;return true;}return false;}
    template<typename Count> void insert(const Key& k,U32 paid,Count pending){if(!count)return;U32 idx=count;for(U32 i=0;i<count;i++){if(slots[i].key==k)return;if(!slots[i].key.len&&idx==count)idx=i;}if(idx<count){used++;peak=std::max(peak,used);}else{auto value=[&](U32 i){return U64(pending(slots[i].key))*slots[i].saved;};idx=0;for(U32 i=1;i<count;i++){if(utility){auto a=value(i),b=value(idx);if(a<b||(a==b&&slots[i].stamp<slots[idx].stamp))idx=i;}else if(slots[i].stamp<slots[idx].stamp)idx=i;}if(utility&&U64(pending(k))*(paid-1)<value(idx))return;}slots[idx]={k,++tick,paid-1};}
};
struct Pending{std::array<U16,MAXN> ids{};U32 len=0,peak=0;void push(U16 id){require(len<ids.size(),"pending overflow");ids[len++]=id;peak=std::max(peak,len);}void pop(U16 id){require(len&&ids[len-1]==id,"pending order");--len;}};
struct Result{bool closed=false;U32 work=0,calls=0,rejects=0,hits=0,peak=0,pending_peak=0;bool same(const Result& r) const{return closed==r.closed&&work==r.work&&calls==r.calls&&rejects==r.rejects&&hits==r.hits&&peak==r.peak&&pending_peak==r.pending_peak;}};
struct Run{const Root& r;Cache cache;Pending pending;Result out;
    Run(const Root& root,U32 cap,bool u):r(root),cache(cap,u){}
    bool visit(U16 id){const auto& node=r.nodes[id];if(node.poly.len==2)return canonical(node.poly,r.p);if(cache.count&&out.work<3*r.n*r.n&&cache.hit(node.poly)){out.work++;out.hits++;return true;}const U32 begin=out.work;for(auto& a:node.actions){auto c=cost(a);if(out.work+c>3*r.n*r.n||out.calls>=8*r.n)return false;auto [ok,paid]=verify(node.poly,a,r.p);require(c==paid,"unexpected verifier meter");out.work+=paid;out.calls++;if(!ok){out.rejects++;continue;}for(auto& child:a.children)require(child.poly==r.nodes[child.id].poly,"valid certificate disagrees with bank child");if(cache.count)for(auto it=a.children.rbegin();it!=a.children.rend();++it)if(it->poly.len>2)pending.push(it->id);for(auto& child:a.children){if(cache.count&&child.poly.len>2)pending.pop(child.id);if(!visit(child.id))return false;}cache.insert(node.poly,out.work-begin,[&](const Key& key){U32 n=0;for(U32 j=0;j<pending.len;j++)n+=r.nodes[pending.ids[j]].poly==key;return n;});return true;}return false;}
    Result execute(){out.closed=visit(U16(r.nodes.size()-1));out.peak=cache.peak;out.pending_peak=pending.peak;require(!out.closed||pending.len==0,"unclosed frontier");require(out.work<=3*r.n*r.n&&out.calls<=8*r.n&&out.peak<=cache.count,"budget");return out;}
};
Key make(std::initializer_list<U8> xs){Key k;k.len=xs.size();int i=0;for(auto c:xs)k.set(i++,c);return k;}
void selftest(){auto a=make({0,1}),b=make({1,1}),c=make({0,1,1});Action good{{{0,a},{1,b}}},bad{{{0,b},{1,b}}},degree{{{0,a},{1,c}}},residue{{{0,make({2,1})},{1,b}}};require(verify(c,good,2)==std::make_pair(true,U32(6)),"positive fixture");require(verify(c,bad,2)==std::make_pair(false,U32(6)),"corrupt fixture");require(verify(c,degree,2)==std::make_pair(false,U32(0)),"degree fixture");require(verify(c,residue,2)==std::make_pair(false,U32(0)),"residue fixture");for(int p:{2,3,5,7,11})for(int n=2;n<=49;n++)for(int j=0;j<n-1;j++){Key k;k.len=n;k.set(n-1,1);auto v=k;v.set(j,1);require(canonical(k,p)&&canonical(v,p)&&!(k==v)&&v.at(j)==1,"key mutation");}Cache no(0,false);no.insert(c,6,[](auto){return 0;});require(!no.hit(c),"zero cap");Cache one(sizeof(Slot),false);one.insert(a,6,[](auto){return 0;});one.insert(b,6,[](auto){return 0;});require(!one.hit(a)&&one.hit(b)&&one.peak==1,"single slot LRU");Cache util(sizeof(Slot),true);util.insert(a,6,[](auto){return 0;});util.insert(b,6,[&](auto k){return k==a?1:0;});require(util.hit(a)&&!util.hit(b),"single slot utility");}
struct Arm{U32 cap;bool utility;};
int main(int argc,char** argv){try{selftest();if(argc<2){std::cerr<<"usage: native BANK [repeats]\n";return 2;}int repeats=argc>2?std::stoi(argv[2]):31;require(repeats>0&&repeats<=1001,"bad repeats");auto start=Clock::now();auto roots=load(argv[1]);auto loadns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();std::vector<Arm> arms{{0,false}};for(U32 c:{64,128,256,512,1024})for(bool u:{false,true})arms.push_back({c,u});std::cerr<<"slot_bytes="<<sizeof(Slot)<<" cache_control_bytes="<<sizeof(Cache)<<" pending_bytes="<<sizeof(Pending)<<" load_ns="<<loadns<<" roots="<<roots.size()<<"\n";std::cout<<"seed,n,cap,policy,slots,allocated,closed,work,calls,rejects,hits,peak,pending_peak,median_ns,first_ns\n";for(auto& r:roots){std::vector<std::vector<U64>> timings(arms.size());std::vector<Result> results(arms.size());std::vector<int> order(arms.size());for(size_t i=0;i<order.size();i++)order[i]=i;std::mt19937 rng(r.seed);for(int rep=0;rep<repeats;rep++){std::shuffle(order.begin(),order.end(),rng);for(int i:order){auto arm=arms[i];auto t0=Clock::now();Result result;{Run run(r,arm.cap,arm.utility);result=run.execute();}auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count();timings[i].push_back(ns);if(rep==0)results[i]=result;else require(results[i].same(result),"nondeterministic receipt");}}for(size_t i=0;i<arms.size();i++){auto& ts=timings[i];auto first=ts[0];std::sort(ts.begin(),ts.end());auto a=arms[i];auto r0=results[i];auto count=a.cap/sizeof(Slot);std::cout<<r.seed<<','<<r.n<<','<<a.cap<<','<<(a.utility?"pending":"lru")<<','<<count<<','<<count*sizeof(Slot)<<','<<r0.closed<<','<<r0.work<<','<<r0.calls<<','<<r0.rejects<<','<<r0.hits<<','<<r0.peak<<','<<r0.pending_peak<<','<<ts[ts.size()/2]<<','<<first<<'\n';}}struct rusage ru{};getrusage(RUSAGE_SELF,&ru);std::cerr<<"max_rss_kib="<<ru.ru_maxrss<<" repeats="<<repeats<<"\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

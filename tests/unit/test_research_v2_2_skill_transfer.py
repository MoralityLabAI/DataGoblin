"""Small exact subset-search and paired-gate regression."""
from datagoblin.research_v2_2.skill_transfer import Problem, fit, paired_answer, search


def test_dependency_closure_and_budget():
    d = dict(mechanisms=['a','b'],error_bits=0,budget=3,experiments=[
        dict(id='c',cost=1,requires=[],predictions=[0,0]),
        dict(id='e',cost=2,requires=['c'],predictions=[0,1])])
    p = Problem('robust_discrimination',d)
    assert p.inspect(2) is None and p.close(2)==3
    for arm in ('analytic','random'):
        answer,score,work,_ = search(Problem(p.kind,d),arm,4)
        assert answer==dict(status='plan',experiments=['c','e'],cost=3)
        assert score==.25 and work<=4
    d['budget']=2
    assert search(Problem(p.kind,d),'analytic',4)[0]==dict(status='infeasible')
    assert search(Problem(p.kind,d),'analytic',1)[0] is None
    w = fit([(p.kind,d)],1)
    assert len(w)==12


def test_portfolio_and_proof_certificates():
    d = dict(cpu_budget=3,ram_cap=2,scenarios=['a','b'],jobs=[
        dict(id='j0',cpu=1,ram=1,requires=[],group=None,permitted=True,gains=[0,0]),
        dict(id='j1',cpu=2,ram=2,requires=['j0'],group=None,permitted=True,gains=[3,4])])
    p = Problem('research_portfolio',d)
    assert search(p,'analytic',4)[0]==dict(jobs=['j0','j1'],worst_gain=3)
    d = dict(budget=2,claims=[dict(id='c0',state='open',requires=[],value=1),
        dict(id='c1',state='checked',requires=['c0'],value=5)],
        jobs=[dict(id='j0',cost=2,resolves=['c0'])])
    p = Problem('proof_obligations',d)
    assert search(p,'analytic',2)[0]==dict(current=[],jobs=['j0'],conditional=['c0','c1'],conditional_value=6)


def test_paired_root_weighting():
    d = dict(min_positive=1,min_mean='1/1',arms=[dict(id='a',roots=[
        dict(active=[5],sham=[1],active_visits=2,sham_visits=2,active_dose=1,sham_dose=1),
        dict(active=[0,0,0],sham=[2],active_visits=2,sham_visits=2,active_dose=1,sham_dose=1)])])
    a,b = paired_answer(d),paired_answer(d,True)
    assert a['eligible']==['a'] and a['arms'][0]['mean_delta']=='1/1'
    assert b['eligible']==[]

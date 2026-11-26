import itertools
import unittest
from dataclasses import replace

from datagoblin.training_gym.resource_plan import (
    Job,
    Storage,
    WorkerWindow,
    paired_outcomes,
    peak_bytes,
    reclaim_candidates,
    schedule,
)

class TestPlan(unittest.TestCase):
    def test_lifetimes(self):
        a=Storage('a','ram',100,0,3)
        self.assertEqual(peak_bytes([a,a,Storage('b','ram',70,1,2),Storage('c','ram',90,3,4)]),{'ram':170})
        with self.assertRaises(ValueError):
            peak_bytes([a,Storage('a','ram',10,0,3)])
    def test_parallel(self):
        jobs=[Job('a',2,2,1,3),Job('b',2,2,1,3),Job('c',1,0,1,2,('a','b'))]
        self.assertEqual(schedule(jobs,(4,4,2)),({'a':(0,3),'b':(0,3),'c':(3,5)},5))
        self.assertEqual(schedule(jobs,(2,2,1))[1],8)
        with self.assertRaises(ValueError):
            schedule(jobs,(1,2,1))
    def test_cycles(self):
        with self.assertRaises(ValueError):
            schedule([Job('a',1,0,1,1,('b',)),Job('b',1,0,1,1,('a',))],(1,0,1))
    def test_exhaustive_pair_identity(self):
        for a in itertools.product((0,1),repeat=4):
            for b in itertools.product((0,1),repeat=4):
                p=paired_outcomes(a,b)
                self.assertAlmostEqual(p['gain'],(sum(b)-sum(a))/4)
                self.assertAlmostEqual(p['gain'],(1-sum(a)/4)*(p['recovery'] or 0)-sum(a)/4*(p['corruption'] or 0))
    def test_exhaustive_reservations(self):
        for capacities in itertools.product(range(1,4),repeat=3):
            jobs=[Job(str(i),1,1,1,i+1,tuple(str(k) for k in range(i) if (i+k)%2==0)) for i in range(4)]
            intervals, end=schedule(jobs,capacities)
            for t in range(end):
                active=[j for j in jobs if intervals[j.key][0]<=t<intervals[j.key][1]]
                self.assertLessEqual(len(active),min(capacities))
            for j in jobs:
                for dep in j.dependencies:
                    self.assertLessEqual(intervals[dep][1],intervals[j.key][0])
    def test_watchdog(self):
        w=WorkerWindow(123,'start:1',True,True,1000,0.01,0,False,True,True)
        workers = [
            w, replace(w, pid=124, gpu_active=True), replace(w, pid=125, gpu_active=None),
            replace(w, pid=126, io_bytes=1), replace(w, pid=127, owned_leaf=False),
            replace(w, pid=128, lease_expired=False),
        ]
        self.assertEqual(reclaim_candidates(workers,pressure=True,min_rss=500,max_cpu_seconds=.1),[(123,'start:1')])
        self.assertEqual(reclaim_candidates(workers,pressure=False,min_rss=500,max_cpu_seconds=.1),[])

if __name__=='__main__':

    unittest.main()

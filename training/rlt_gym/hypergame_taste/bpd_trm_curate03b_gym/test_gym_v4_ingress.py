"""Synthetic stage-ingress canaries; run with pytest or unittest discovery."""
import copy
import hashlib
import unittest

from training.rlt_gym.hypergame_taste.bpd_trm_curate03b_gym.gym_v4_ingress import SCHEMA, audit, canonical


def rows():
    p = {'task': 'E31', 'root': 1, 'arm': 'C_analytic_full16', 'window': 0,
         'parent_model_sha256': 'p', 'parent_custody_sha256': 'c', 'feedback_cohort_id': 'f'}
    result = []
    for k in (0, 4):
        terminal = k == 4
        s = {'stage': k, 'scores': [.1] * k, 'update_norms': [.2] * k,
             'feedback_loss': 1., 'feedback_age': 0, 'charged_cpu_before': 0., 'cpu_to_stage': k / 10}
        if terminal:
            s.update(selected_indices=[0, 1], cpu_to_proposal=.5)
        result.append({'schema': SCHEMA, 'record_id': hashlib.sha256(canonical(['E31', 1, p['arm'], 0, k])).hexdigest(),
                       'split': 'train', 'evaluation_leakage': False, 'evidence_class': 'DEVELOPMENT_ONLY',
                       'provenance': p, 'state': s, 'behavior_action': 'SELECT_BATCH' if terminal else 'ACQUIRE_4',
                       'label_kind': 'verified_terminal_preference' if terminal else 'behavior_only',
                       'training_eligible': terminal,
                       'preference': {'chosen': 'SELECT_BATCH', 'rejected': 'ABSTAIN'} if terminal else None,
                       'outcome': {'status': 'committed', 'actual_gain': .1, 'cal_before': 1., 'cal_after': .9,
                                   'committed': True} if terminal else None})
    return result


class TestGymV4Ingress(unittest.TestCase):
    def test_valid(self):
        p, c = audit(rows(), [1, 2, 3])
        self.assertEqual(p, {1: 'train', 2: 'calibration', 3: 'test'})
        self.assertEqual(c['train:verified_terminal_preference'], 1)

    def test_canaries(self):
        fixtures = []
        z = rows()
        z[0]['state']['scores'] = [999.]
        fixtures.append(z)
        z = rows()
        z[0]['outcome'] = z[1]['outcome']
        fixtures.append(z)
        z = rows()
        z[1]['outcome']['actual_gain'] = None
        fixtures.append(z)
        z = rows()
        z[1]['split'] = 'eval'
        fixtures.append(z)
        z = rows()
        z.append(copy.deepcopy(z[0]))
        fixtures.append(z)
        for z in fixtures:
            with self.assertRaises(ValueError):
                audit(z, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()

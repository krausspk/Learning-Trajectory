import importlib.util
from pathlib import Path

import numpy as np

SPEC=importlib.util.spec_from_file_location('norm_rank_control_under_test',Path(__file__).with_name('norm_rank_control.py'))
control=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def test_reference_ranks_and_average_ties():
    # Query ranks (2.5, 1, 2.5, 4); candidate ranks (3, 1.5, 4, 1.5).
    c=control.norm_rank_costs([3,1,3,8],[5,2,9,2])
    expected=np.abs(np.array([2.5,1,2.5,4])[:,None]-np.array([3,1.5,4,1.5])[None,:])/3
    assert np.array_equal(c,expected)


def test_separate_monotone_scaling_and_candidate_reordering():
    q=np.array([1.,4.,3.,2.]);c=np.array([8.,2.,3.,9.])
    result=control.norm_rank_costs(q,c)
    assert np.array_equal(result,control.norm_rank_costs(100+q**3,.1+np.sqrt(c)))
    order=np.array([2,0,3,1])
    assert np.array_equal(control.norm_rank_costs(q,c[order]),result[:,order])
    assert np.array_equal(control.norm_rank_costs(q[order],c),result[order])


def test_leave_one_recalibrates_instead_of_subsetting():
    q=np.array([1.,4.,2.,3.]);c=np.array([3.,4.,1.,2.])
    full=control.norm_rank_costs(q,c)
    jack=control.calibrated_leave_one_costs(q,c)
    for deleted in range(4):
        keep=np.arange(4)!=deleted
        assert np.array_equal(jack[deleted],control.norm_rank_costs(q[keep],c[keep]))
    # Nonuniform gaps change the candidate ordering/ties after recalibration.
    assert not np.array_equal(jack[1],full[np.ix_([0,2,3],[0,2,3])])


def test_identical_norms_are_fully_tied_without_diagonal_preference():
    costs=control.norm_rank_costs(np.ones(10),np.ones(10)*7)
    assert np.array_equal(costs,np.zeros((10,10)))

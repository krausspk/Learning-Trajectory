import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'code/validate_results.py'
spec=importlib.util.spec_from_file_location('independent_validation',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_paired_bootstrap_constant_shift_and_reproducibility():
    assert m.bootstrap_difference([.03]*40)==(.03,.03,.03)
    a=m.bootstrap_difference(np.linspace(-1,1,40),seed=71)
    assert a==m.bootstrap_difference(np.linspace(-1,1,40),seed=71)
    assert a[1]<0<a[2]

def test_split_overlap_rejected_before_scoring():
    splits={'one_bit':{'splits':{'train':{'seeds':[1],'run_ids':['a','b']},
                               'validation':{'seeds':[1],'run_ids':['a','b']},
                               'test':{'seeds':[],'run_ids':[]}}}}
    records=[{'cohort':'one_bit','seed':1,'run_id':'a','bits':(0,)},
             {'cohort':'one_bit','seed':1,'run_id':'b','bits':(1,)}]
    with pytest.raises(AssertionError,match='Seed leakage'):
        m.audit_splits(records,splits,False)

def test_bit_imbalance_rejected():
    splits={'one_bit':{'splits':{'train':{'seeds':[1],'run_ids':['a','b']},
                               'validation':{'seeds':[],'run_ids':[]},
                               'test':{'seeds':[],'run_ids':[]}}}}
    records=[{'cohort':'one_bit','seed':1,'run_id':'a','bits':(0,)},
             {'cohort':'one_bit','seed':1,'run_id':'b','bits':(0,)}]
    with pytest.raises(AssertionError,match='Unbalanced bit'):
        m.audit_splits(records,splits,False)

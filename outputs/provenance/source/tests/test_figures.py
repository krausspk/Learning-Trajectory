from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from learning_trajectory.figures import (
    BUILDERS, FIGURE_NAMES, TITLES, FigureData, _groups_from_pairs,
    _s1, _save, _student, make_figures,
)


def test_full_figure_inventory_has_all_published_figures():
    assert len(BUILDERS)==16
    assert FIGURE_NAMES == tuple([f'figure_{i:02d}' for i in range(1,7)] + [f'figure_S{i:02d}' for i in range(1,11)])
    assert set(FIGURE_NAMES)==set(TITLES)


def test_missing_inputs_fail_without_creating_misleading_partial_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError,match='all completed analyses'):
        make_figures(tmp_path)
    assert not (tmp_path/'figures').exists()


def test_student_bands_use_independent_runs():
    frame=pd.DataFrame({'checkpoint':[0]*5+[1]*5,'accuracy':[0,1,2,3,4,2,4,6,8,10]})
    result=_student(frame,'accuracy')
    for checkpoint in (0,1):
        values=frame[frame.checkpoint.eq(checkpoint)].accuracy.to_numpy()
        row=result[result.checkpoint.eq(checkpoint)].iloc[0]
        assert row.estimate==pytest.approx(values.mean())
        assert row.ci_high==pytest.approx(values.mean()+t.ppf(.975,4)*values.std(ddof=1)/np.sqrt(5))


def test_pair_group_reconstruction_and_conflict_detection():
    frame=pd.DataFrame({'index_a':[0,0,1],'index_b':[1,2,2],'group_a':['a','a','b'],'group_b':['b','c','c']})
    assert _groups_from_pairs(frame).tolist()==['a','b','c']
    frame.loc[1,'group_a']='bad'
    with pytest.raises(ValueError,match='Conflicting'):
        _groups_from_pairs(frame)


def test_training_diagnostics_render_real_run_columns_to_png_and_svg(tmp_path):
    # Explicit synthetic fixture solely for renderer verification, never production output.
    rows=[]
    for k,dataset in enumerate(('mnist','fashion_mnist','kmnist')):
        for run in range(3):
            for cp in (0,1,20):
                rows.append(dict(cohort='main',dataset=dataset,run=f'{dataset}{run}',checkpoint=cp,
                                 accuracy=.1+.7*(cp>0)+.01*run-.03*k,loss=2.3/(cp+1)+.02*run+.1*k))
    figure=_s1(SimpleNamespace(training=pd.DataFrame(rows)))
    paths=_save(figure,tmp_path,'figure_S01')
    assert len(paths)==2
    assert all(path.stat().st_size>1000 for path in paths)
    assert '<svg' in (tmp_path/'figure_S01.svg').read_text()
    assert not plt.get_fignums()

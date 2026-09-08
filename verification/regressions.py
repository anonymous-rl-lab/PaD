"""Regression checks for the audited nonfinite readout boundary."""
import numpy as np
from readouts import double_features

def check_readouts():
    cases=0
    for slot in range(4):
        for missing in [np.nan,np.inf,-np.inf]:
            values=[np.array([1.]),np.array([2.]),np.array([.3]),np.array([.2])]
            reference=[v.copy() for v in values];reference[slot][:]=np.nan
            values[slot][:]=missing
            with np.errstate(over='raise',invalid='raise'):
                actual=double_features(*values);expected=double_features(*reference)
                swapped=double_features(values[1],values[0],values[2],values[3])
            for a,b in zip(actual,expected):
                assert np.isfinite(a).all()
                np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(actual[0],-swapped[0],atol=1e-15)
            np.testing.assert_allclose(actual[1],-swapped[1],atol=1e-15)
            np.testing.assert_array_equal(actual[2],swapped[2])
            cases+=1
    return dict(nonfinite_readout_cases=cases,finite_outputs=True,exchange_checks=True)

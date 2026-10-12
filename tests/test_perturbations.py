import numpy as np
import pandas as pd
import pytest
from clinical_lab.perturbations import perturb,training_scales


def data():
    return pd.DataFrame(dict(patient_id=["a"]*4+["b"]*4,hospital="A",hour=list(range(4))*2,
        HR=[60.,np.nan,80.,90.,160.,170.,180.,190.],Age=55,Gender=0,
        ICULOS=list(range(1,5))*2,SepsisLabel=[0]*4+[1]*4))


def test_missingness_never_changes_metadata_or_input():
    df=data();old=df.copy(deep=True)
    out=perturb(df,"missingness",1,seed=8)
    assert out.HR.isna().all()
    pd.testing.assert_frame_equal(out.drop(columns="HR"),df.drop(columns="HR"))
    pd.testing.assert_frame_equal(df,old)
    pd.testing.assert_frame_equal(perturb(df,"missingness",0),df)


def test_delay_causal_boundary_and_existing_holes():
    df=data();out=perturb(df,"delay",1)
    np.testing.assert_equal(out.HR.to_numpy(),np.array([np.nan,60,np.nan,80,np.nan,160,170,180]))
    pd.testing.assert_frame_equal(out.drop(columns="HR"),df.drop(columns="HR"))
    # An unobserved future measurement cannot affect an earlier availability hour.
    edited=df.copy();edited.loc[3,"HR"]=900
    pd.testing.assert_series_equal(perturb(edited,"delay",1).HR,out.HR)
    assert perturb(df,"delay",6).HR.isna().all()


def test_delay_works_with_interleaved_patients_and_arbitrary_indices():
    df=data().iloc[[0,4,1,5,2,6,3,7]].copy();df.index=list(range(20,28))
    out=perturb(df,"delay",1)
    assert list(out.index)==list(df.index)
    np.testing.assert_equal(out.HR.to_numpy(),[np.nan,np.nan,60,160,np.nan,170,80,180])


def test_noise_requires_frozen_training_scales():
    df=data();scales=training_scales(df.iloc[:4])
    assert scales["HR"]==pytest.approx(15/1.349)
    with pytest.raises(ValueError,match="training-only"):
        perturb(df,"noise",.2)
    out=perturb(df,"noise",.2,seed=2,scales=scales)
    pd.testing.assert_frame_equal(out,perturb(df,"noise",.2,seed=2,scales=scales))
    assert out.HR.isna().equals(df.HR.isna())
    pd.testing.assert_frame_equal(out.drop(columns="HR"),df.drop(columns="HR"))
    pd.testing.assert_frame_equal(perturb(df,"noise",0,scales=scales),df)


def test_invalid_parameters_are_rejected():
    for kind,severity in [("missingness",1.1),("delay",.5),("noise",-1),("anything",1)]:
        with pytest.raises(ValueError):
            perturb(data(),kind,severity)

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from clinical_lab.evaluation import evaluate, choose_threshold, selective, alert_hours, _utility_triplet, _HourlyMetrics
from clinical_lab.vendor.evaluate_sepsis_score import compute_prediction_utility, evaluate_sepsis_score


def cohort():
    parts=[]
    for pid, first, n in [("event",6,24),("control",None,24),("left",0,24),("late",20,24)]:
        parts.append(pd.DataFrame(dict(patient_id=pid,hospital="A",hour=np.arange(n),
            SepsisLabel=np.zeros(n,dtype=int) if first is None else (np.arange(n)>=first).astype(int),
            Age=50,Gender=1)))
    return pd.concat(parts,ignore_index=True)


def test_official_scorer_unmodified():
    path=Path(__file__).parents[1]/"clinical_lab/vendor/evaluate_sepsis_score.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5"


def test_official_hand_fixture_and_normalization(tmp_path):
    assert compute_prediction_utility([0,0,0,0,1,1],[0,0,1,1,1,1])==pytest.approx(3.388888888888889)
    df=cohort()
    p=np.tile(np.linspace(.01,.99,24),4)
    ydir,pdir=tmp_path/"labels",tmp_path/"predictions"
    ydir.mkdir(); pdir.mkdir()
    for pid,part in df.groupby("patient_id"):
        part[["SepsisLabel"]].to_csv(ydir/f"{pid}.psv",sep="|",index=False)
        pd.DataFrame({"PredictedProbability":p[part.index],"PredictedLabel":(p[part.index]>=.5).astype(int)}).to_csv(pdir/f"{pid}.psv",sep="|",index=False)
    official=evaluate_sepsis_score(str(ydir),str(pdir))[-1]
    result=evaluate(df,p,.5,bootstrap=0)
    assert result["challenge"]["challenge_utility"]==pytest.approx(official)
    best_predictions=np.zeros(len(df))
    for _,part in df.groupby("patient_id"):
        y=part.SepsisLabel.to_numpy()
        if y.any():
            onset=np.argmax(y)+6
            best_predictions[part.index[max(0,onset-12):min(onset+4,len(part))]]=1
    assert evaluate(df,best_predictions,.5,bootstrap=0)["challenge"]["challenge_utility"]==pytest.approx(1)
    assert evaluate(df,np.zeros(len(df)),.5,bootstrap=0)["challenge"]["challenge_utility"]==pytest.approx(0)


def test_operational_window_and_censoring():
    df=cohort(); p=np.zeros(len(df))
    # Event onset=12, warning window [6,12). Alarm at6 succeeds exactly6h early.
    # Control persistently high has alarms0,6,12,18. Septic prewindow alarm0.
    p[[0,6]]=1; p[24:48]=1
    result=evaluate(df,p,.5,bootstrap=10)
    op=result["pre_onset"]
    assert op["eligible_events"]==1
    assert op["detected_events"]==1
    assert op["median_lead_time_hours"]==6
    assert op["excluded_left_censored"]==1
    assert op["excluded_onset_after_followup"]==1
    assert op["false_alert_episodes"]==5
    assert op["sepsis_free_patient_days"]==30/24
    assert op["false_alerts_per_patient_day"]==4
    assert op["nonseptic_false_alerts_per_patient_day"]==4
    # At onset and after it are never early detections.
    p[:24]=0;p[12:24]=1
    assert evaluate(df,p,.5,bootstrap=0)["pre_onset"]["detected_events"]==0


def test_cooldown_can_suppress_early_event_warning():
    assert alert_hours(np.arange(13),np.ones(13)).tolist()==[0,6,12]
    df=cohort().iloc[:24].copy();p=np.zeros(24);p[[5,6]]=1
    # Alarm just before warning window prevents another alarm at6.
    assert evaluate(df,p,.5,bootstrap=0)["pre_onset"]["event_sensitivity"]==0


def test_weighted_cluster_metrics_match_repeated_samples():
    y=np.array([0,1,0,1,1,0]);p=np.array([.2,.2,.7,.8,.8,.9]);groups=np.array([0,0,1,1,2,2])
    counts=np.array([2,0,1]);cache=_HourlyMetrics(y,p,groups)
    result=cache.metrics(counts)
    idx=np.repeat(np.arange(len(y)),counts[groups])
    assert result["auroc"]==pytest.approx(roc_auc_score(y[idx],p[idx]))
    assert result["average_precision"]==pytest.approx(average_precision_score(y[idx],p[idx]))
    assert result["brier"]==pytest.approx(brier_score_loss(y[idx],p[idx]))


def test_threshold_all_negative_and_null_metrics():
    df=cohort().iloc[24:48].reset_index(drop=True)
    threshold=choose_threshold(df,np.ones(len(df)))
    assert threshold>1
    result=evaluate(df,np.ones(len(df)),threshold,bootstrap=3)
    assert result["challenge"]["challenge_utility"] is None
    assert result["hourly_metrics"]["auroc"] is None
    assert result["hourly_metrics"]["average_precision"] is None
    assert result["pre_onset"]["event_sensitivity"] is None
    import json
    json.dumps(result,allow_nan=False)


def test_selective_reports_withheld_events():
    df=cohort();p=np.full(len(df),.51)
    result=selective(df,p,.5,[0,.02])["results"]
    assert result[0]["coverage"]==1
    assert result[1]["coverage"]==0
    assert result[1]["retained_brier"] is None
    assert result[1]["eligible_events_with_all_warning_hours_withheld"]==1
    assert result[1]["event_sensitivity_with_withheld_as_no_alert"]==0
    assert result[1]["withheld_positive_label_hours"]==int(df.SepsisLabel.sum())


def test_validation_prevents_silent_temporal_alignment_errors():
    df=cohort()
    with pytest.raises(ValueError,match="Probabilities"):
        evaluate(df,np.full(len(df),np.nan),.5)
    df.loc[1,"hour"]=5
    with pytest.raises(ValueError,match="contiguous"):
        evaluate(df,np.zeros(len(df)),.5)

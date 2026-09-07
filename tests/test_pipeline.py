import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from src.preprocessing import add_rul, split_engines, load_fd001, load_trajectories
from src.feature_engineering import build_windows, engine_weights, endpoint_mask
from src.evaluation import ensemble_spread, maintenance_status, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_targets_respect_engine_and_test_censoring():
    df = pd.DataFrame({"engine_id":[1,1,1,2,2], "cycle":[1,2,3,1,2]})
    assert add_rul(df, 125).rul_raw.tolist() == [2,1,0,1,0]
    labeled = add_rul(df, 4, pd.Series({1:5,2:3}))
    assert labeled.rul_raw.tolist() == [7,6,5,4,3]
    assert labeled.rul_target.tolist() == [4,4,4,4,3]


def test_causal_windows_no_cross_engine_no_future():
    df = pd.DataFrame({"engine_id":[1,1,1,2,2], "cycle":[1,2,3,1,2], "sensor_1":[1.,2.,3.,100.,101.]})
    scaler = StandardScaler().fit(df[["sensor_1"]])
    windows = build_windows(df, ["sensor_1"], scaler, 3)
    raw = windows.sequence*scaler.scale_[0]+scaler.mean_[0]
    np.testing.assert_allclose(raw[0,:,0], [1,1,1], atol=1e-5)
    np.testing.assert_allclose(raw[3,:,0], [100,100,100], atol=1e-5)
    changed = df.copy(); changed.loc[2,"sensor_1"] = 99999
    again = build_windows(changed, ["sensor_1"], scaler, 3)
    np.testing.assert_array_equal(windows.sequence[:2], again.sequence[:2])
    np.testing.assert_array_equal(windows.tabular[:2], again.tabular[:2])


def test_parser_rejects_duplicate_and_gap(tmp_path):
    rows = np.ones((2,26)); rows[:,0]=1; rows[:,1]=[1,1]
    path = tmp_path/"bad.txt"; np.savetxt(path, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_trajectories(path)
    rows[:,1]=[1,3]; np.savetxt(path,rows)
    with pytest.raises(ValueError, match="contiguous"):
        load_trajectories(path)


def test_engine_split_scaling_and_selection_isolation():
    config = json.loads((ROOT/"config.json").read_text())
    train, test = load_fd001(ROOT/"data"/"raw", config["rul_cap"])
    fit, val, manifest = split_engines(train, config)
    assert not set(fit.engine_id) & set(val.engine_id)
    assert set(fit.engine_id) | set(val.engine_id) == set(train.engine_id)
    assert len(manifest["validation_cutoffs"]) == val.engine_id.nunique()
    assert (train.groupby("engine_id").rul_raw.last() == 0).all()
    assert (test.groupby("engine_id").rul_raw.last() > 0).all()
    w = engine_weights(fit)
    total = pd.Series(w).groupby(fit.engine_id).sum()
    np.testing.assert_allclose(total, total.iloc[0])


def test_interval_bounds_and_status_boundaries():
    mean, lo, hi, spread = ensemble_spread({"a":np.array([0.,125.]), "b":np.array([0.,125.]),
        "c":np.array([10.,125.]), "d":np.array([20.,125.])},125,1.96)
    assert np.all(lo <= mean) and np.all(mean <= hi)
    assert np.all(lo >= 0) and np.all(hi <= 125)
    assert lo[1] == hi[1] == 125  # Agreement does not guarantee correctness.
    thresholds = {"red_below":20,"orange_below":40,"yellow_below":80}
    assert [maintenance_status(x,thresholds) for x in [19.9,20,40,80]] == ["Red","Orange","Yellow","Green"]


@pytest.mark.skipif(not (ROOT/"artifacts"/"pipeline.joblib").exists(), reason="Run pipeline first")
def test_saved_artifacts_and_label_free_prefix_match_recorded_predictions():
    from src.inference import load_artifacts, predict_prefix
    bundle = load_artifacts(ROOT)
    config = bundle["config"]
    train, test = load_fd001(ROOT/"data"/"raw", config["rul_cap"])
    fit, val, _ = split_engines(train,config)
    np.testing.assert_allclose(bundle["scaler"].mean_, fit[bundle["sensors"]].mean())
    recorded = pd.read_csv(ROOT/"reports"/"test_predictions.csv")
    for engine, cycle in [(1,1), (1,int(test[test.engine_id==1].cycle.max())), (2,15)]:
        prefix = test[(test.engine_id==engine)&(test.cycle<=cycle)].drop(columns=["rul_raw","rul_target"])
        actual = predict_prefix(prefix,bundle)
        row = recorded[(recorded.engine_id==engine)&(recorded.cycle==cycle)].iloc[0]
        for name, prediction in actual["predictions"].items():
            assert prediction == pytest.approx(row[name],abs=1e-3)
        assert actual["shap_baseline"]+actual["shap"].sum() == pytest.approx(actual["raw_xgb"],abs=1e-3)
    metric_rows = pd.read_csv(ROOT/"reports"/"metrics.csv")
    ends = recorded[recorded.is_endpoint]
    assert len(ends) == test.engine_id.nunique()
    for name in bundle["models"]:
        metric = metric_rows.query("partition=='test' and scope=='endpoint' and target=='rul_raw'")
        expected = metric[metric.model==name].iloc[0]
        for key,value in metrics(ends.rul_raw,ends[name]).items():
            assert value == pytest.approx(expected[key])

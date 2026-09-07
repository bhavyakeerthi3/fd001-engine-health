from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/"artifacts"/"pipeline.joblib").exists(), reason="Run pipeline first")
def test_dashboard_engine_cycle_and_empty_sensor_selection():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT/"dashboard"/"app.py"), default_timeout=90).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.selectbox[0].set_value(2).run()
    app.slider[0].set_value(1).run()
    assert not app.exception
    assert "1 actual measurements" in " ".join(x.value for x in app.caption)
    app.multiselect[0].set_value([]).run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert any("Evaluation-only original RUL" in x.value for x in app.info)

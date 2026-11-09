from pathlib import Path
import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

def open_app():
    return AppTest.from_file(APP, default_timeout=60).run()

def test_app_opens_without_errors():
    assert not open_app().exception

def test_app_has_the_three_tabs():
    assert [t.label for t in open_app().tabs][:3] == ["Estimate", "Plan", "How it works"]

def test_a_typed_effort_gives_an_estimate_in_the_page():
    text = " ".join(m.value for m in open_app().markdown)
    assert "Best case" in text

def test_changing_the_typed_time_does_not_crash():
    at = open_app()
    at.text_input[0].set_value("1:45:00").run()
    assert not at.exception

def test_a_bad_time_shows_a_message_not_a_crash():
    at = open_app()
    at.text_input[0].set_value("not a time").run()
    assert not at.exception

def test_a_lone_number_is_read_as_minutes():
    at = open_app()
    at.number_input[0].set_value(5.1)
    at.text_input[0].set_value("25").run()
    assert not at.exception
    assert not at.error
    assert "Best case" in " ".join(m.value for m in at.markdown)

def test_an_unrealistic_pace_shows_a_message_not_an_estimate():
    at = open_app()
    at.number_input[0].set_value(6.0)
    at.text_input[0].set_value("1:55:00").run()
    assert not at.exception
    assert len(at.error) == 1
    assert "not realistic" in at.error[0].value

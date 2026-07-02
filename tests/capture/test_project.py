from pathlib import Path
from agent_core.capture.project import resolve_capture_db
from agent_core.config import BaseConfig


def test_config_has_project_marker_default_none():
    assert BaseConfig().project_marker is None


def test_walk_up_finds_marker(tmp_path):
    home = tmp_path / "home"
    proj = home / "work" / "acme"
    (proj / ".pare").mkdir(parents=True)
    sub = proj / "src" / "deep"
    sub.mkdir(parents=True)
    db, is_project = resolve_capture_db(sub, ".pare", home=home,
                                        xdg_state=tmp_path / "state", channel_id="c1")
    assert is_project is True
    assert db == proj / ".pare" / "capture.db"


def test_marker_none_means_no_project(tmp_path):
    db, is_project = resolve_capture_db(tmp_path, None, home=tmp_path,
                                        xdg_state=tmp_path / "state", channel_id="c1")
    assert is_project is False
    assert (tmp_path / "state") in db.parents


def test_home_ceiling_is_not_a_project(tmp_path):
    home = tmp_path / "home"
    (home / ".pare").mkdir(parents=True)
    db, is_project = resolve_capture_db(home, ".pare", home=home,
                                        xdg_state=tmp_path / "state", channel_id="c1")
    assert is_project is False   # a .pare exactly at $HOME is ignored

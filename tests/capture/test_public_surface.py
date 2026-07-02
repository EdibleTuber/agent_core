def test_public_surface_importable():
    from agent_core.capture import (
        CaptureStore, CaptureRecord, CaptureLayer,
        SearchCapture, ReadCapture, resolve_capture_db,
    )
    assert all([CaptureStore, CaptureRecord, CaptureLayer, SearchCapture, ReadCapture, resolve_capture_db])

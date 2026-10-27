import numpy as np

from reid.geometry import iou_matrix
from reid.tracker import Tracker, TrackerConfig


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_track_keeps_id_while_moving():
    tracker = Tracker(TrackerConfig(min_hits=1))
    ids = []
    for step in range(10):
        x = step * 2
        tracks = tracker.update(np.array([[x, 0, x + 20, 50, 0.9]]))
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_two_people_get_distinct_ids():
    tracker = Tracker(TrackerConfig(min_hits=1))
    tracks = tracker.update(np.array([[0, 0, 20, 50, 0.9], [100, 0, 120, 50, 0.9]]))
    assert len({t.track_id for t in tracks}) == 2


def test_track_survives_short_occlusion_and_expires_after_max_misses():
    tracker = Tracker(TrackerConfig(min_hits=1, max_misses=3))
    first = tracker.update(np.array([[0, 0, 20, 50, 0.9]]))[0].track_id
    for _ in range(2):
        tracker.update(np.zeros((0, 5)))
    assert tracker.update(np.array([[1, 0, 21, 50, 0.9]]))[0].track_id == first
    for _ in range(5):
        tracker.update(np.zeros((0, 5)))
    assert tracker.update(np.array([[1, 0, 21, 50, 0.9]]))[0].track_id != first


def test_appearance_resolves_crossing_people():
    cfg = TrackerConfig(min_hits=1, iou_weight=0.2, min_iou=0.0, max_cost=0.9)
    tracker = Tracker(cfg)
    red, blue = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    tracks = tracker.update(np.array([[0, 0, 20, 50, 0.9], [30, 0, 50, 50, 0.9]]), np.stack([red, blue]))
    id_by_color = {"red": tracks[0].track_id, "blue": tracks[1].track_id}
    # people swap sides; appearance should keep the identities attached to the right person
    tracks = tracker.update(np.array([[28, 0, 48, 50, 0.9], [2, 0, 22, 50, 0.9]]), np.stack([red, blue]))
    by_box = {int(t.box[0]): t.track_id for t in tracks}
    assert by_box[28] == id_by_color["red"]
    assert by_box[2] == id_by_color["blue"]


def test_tentative_tracks_hidden_until_min_hits():
    tracker = Tracker(TrackerConfig(min_hits=3))
    det = np.array([[0, 0, 20, 50, 0.9]])
    assert tracker.update(det) == []
    assert tracker.update(det) == []
    assert len(tracker.update(det)) == 1

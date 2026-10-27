import numpy as np

from reid.gallery import GlobalIdentityGallery


def emb(*values):
    return np.array(values, dtype=float)


def test_same_person_on_second_camera_gets_same_global_id():
    g = GlobalIdentityGallery(threshold=0.8)
    a = g.assign("cam1", [(1, emb(1, 0, 0))])[1]
    b = g.assign("cam2", [(7, emb(0.95, 0.05, 0))])[7]
    assert a == b
    assert g.cross_camera_identities() == {a: ["cam1", "cam2"]}


def test_dissimilar_person_gets_new_id():
    g = GlobalIdentityGallery(threshold=0.8)
    a = g.assign("cam1", [(1, emb(1, 0, 0))])[1]
    b = g.assign("cam2", [(1, emb(0, 1, 0))])[1]
    assert a != b


def test_link_is_stable_for_track_lifetime():
    g = GlobalIdentityGallery(threshold=0.8)
    first = g.assign("cam1", [(3, emb(1, 0, 0))])[3]
    for _ in range(5):
        assert g.assign("cam1", [(3, emb(0.9, 0.1, 0))])[3] == first


def test_identity_not_duplicated_within_one_camera():
    g = GlobalIdentityGallery(threshold=0.5)
    g.assign("cam1", [(1, emb(1, 0, 0))])
    # two simultaneous tracks in cam1 look similar, but only one can be identity 1
    ids = g.assign("cam1", [(1, emb(1, 0, 0)), (2, emb(0.99, 0.01, 0))])
    assert ids[1] != ids[2]


def test_hungarian_picks_best_overall_matching():
    g = GlobalIdentityGallery(threshold=0.5)
    g.assign("cam1", [(1, emb(1, 0, 0)), (2, emb(0, 1, 0))])
    ids = g.assign("cam2", [(10, emb(0.1, 0.9, 0)), (11, emb(0.9, 0.1, 0))])
    assert ids[10] == 2 and ids[11] == 1

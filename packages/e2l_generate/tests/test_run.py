import json

from e2l_common.config import GenerateConfig, SimConfig
from e2l_generate.run import AttemptResult, attempt_seed, episode_stem, summarise


def result(demo, k, status, saved=False, size=None):
    return AttemptResult(
        stem=episode_stem(demo, k),
        demo_id=demo,
        attempt=k,
        episode_index=k % 50,
        status=status,
        steps=10,
        saved=saved,
        size_mb=size,
    )


def test_episode_stem():
    assert episode_stem("20261003_scripted_000", 7) == "20261003_scripted_000_0007"


def test_attempt_seed_is_per_attempt():
    def draw(*a):
        return attempt_seed(*a).random(4).tolist()

    assert draw(0, "d1", 3) == draw(0, "d1", 3)
    assert draw(0, "d1", 3) != draw(0, "d1", 4)
    assert draw(0, "d1", 3) != draw(0, "d2", 3)
    assert draw(0, "d1", 3) != draw(1, "d1", 3)


def test_summarise_counts_yield_and_sizes():
    results = [
        result("d1", 1, "failure"),
        result("d1", 0, "success", saved=True, size=20.0),
        result("d1", 2, "error"),
        result("d1", 3, "placement_failed"),
        result("d2", 0, "success", saved=True, size=30.0),
        result("d2", 1, "failure", saved=True, size=10.0),
    ]
    m = summarise(results, "run0", GenerateConfig(), SimConfig())
    assert m["run_id"] == "run0"
    assert m["demos"]["d1"] == {
        "attempts": 4,
        "successes": 1,
        "failures": 1,
        "placement_failed": 1,
        "errors": 1,
        "yield": 0.25,
    }
    assert m["demos"]["d2"]["yield"] == 0.5
    assert m["mean_episode_mb"] == 20.0  # mean over saved episodes, failures included
    assert [e["stem"] for e in m["episodes"][:2]] == ["d1_0000", "d1_0001"]  # sorted
    assert "size_mb" not in m["episodes"][0]
    assert m["generate_config"]["episodes_per_demo"] == 50
    assert m["sim_config"]["control_freq"] == 20
    json.dumps(m)  # serialisable as-is


def test_summarise_empty():
    m = summarise([], "run0", GenerateConfig(), SimConfig())
    assert m["demos"] == {} and m["episodes"] == [] and m["mean_episode_mb"] is None

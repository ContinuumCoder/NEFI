from nefi.bench import run_benchmark
from nefi.bench.protocol import scene_columns
from nefi.instances.thermal_tomography import ThermalTomography
from nefi.measurement import Measurement


def test_scene_columns_flattens_scalars():
    import torch

    m = Measurement(
        torch.zeros(2, 2),
        meta={
            "fidelity": "x",
            "supersample": 2,
            "big": torch.ones(3),
            "scene": {"n_defects": 3, "shapes": ["box", "sphere"], "depth": 0.4},
        },
    )
    cols = scene_columns(m)
    assert cols == {
        "meta/fidelity": "x",
        "meta/supersample": 2,
        "scene/n_defects": 3,
        "scene/n_shapes": 2,
        "scene/depth": 0.4,
    }


def test_benchmark_rows_carry_scene_strata():
    inst = ThermalTomography(preset="smoke")
    res = run_benchmark(
        inst,
        methods=["neural"],
        n_samples=2,
        seeds=(0,),
        budget_scale=0.05,
        max_total_steps=6,
        progress=False,
        device="cpu",
        warmup=False,
    )
    assert all("scene/n_defects" in r for r in res.rows), res.rows[0].keys()
    strata = res.strata()
    assert "scene/n_defects" in strata
    tab = res.table(by="scene/n_defects")
    assert sum(r["n"] for r in tab) == len(res.rows)
    assert all("scene/n_defects" in r for r in tab)
    md = res.to_markdown(by="scene/n_defects", header=False)
    assert "N Defects" in md
    assert res.summary_csv(by="scene/n_defects").count("\n") >= 1
    # by_class still works and equals by="class"
    assert str(res.table(by_class=True)) == str(res.table(by="class"))  # NaN-safe compare

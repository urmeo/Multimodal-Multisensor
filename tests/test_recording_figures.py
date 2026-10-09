import numpy as np
import pandas as pd
import pytest

from scripts.build_recording_figures import (
    digest,
    main,
    metadata_text,
    payload,
    provenance,
    samples,
    verify,
)


def test_sample_selection_keeps_channels_and_repeated_intervals():
    frame = pd.DataFrame(
        {"reltime": [0, 1, 2, 3], "iSensor": [3, 5, 3, 5], "ibi": [800, 900, 800, 0]}
    )
    assert samples(frame, "ibi") == {
        "3": {"time_s": [0.0, 2.0], "values": [800, 800]},
        "5": {"time_s": [1.0], "values": [900]},
    }


def test_quality_and_nonfinite_readings_are_not_plotted():
    frame = pd.DataFrame(
        {
            "reltime": range(5),
            "iSensor": [0] * 5,
            "pupil": [3, 4, 5, np.inf, 0],
            "pupilQ": [0.5, 0.8, 1.1, 1, 1],
        }
    )
    assert samples(frame, "pupil", quality="pupilQ") == {
        "0": {"time_s": [1.0], "values": [4]}
    }


def test_fractional_channels_and_reversed_clocks_fail():
    with pytest.raises(ValueError, match="integer channel"):
        samples(
            pd.DataFrame({"reltime": [0, 1], "iSensor": [1.1, 1.9], "ibi": [800, 900]}),
            "ibi",
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        samples(
            pd.DataFrame({"reltime": [1, 0], "iSensor": [3, 3], "ibi": [800, 900]}),
            "ibi",
        )


def test_published_panels_are_bound_to_source_selections_and_bytes(tmp_path):
    for name in ("heart", "interval", "pupil"):
        (tmp_path / f"recorded-{name}.png").write_bytes(name.encode())
    info = {"selection_sha256": "source-selection"}
    metadata = {
        **info,
        "figures": {
            f"recorded-{name}.png": digest(tmp_path / f"recorded-{name}.png")
            for name in ("heart", "interval", "pupil")
        },
    }
    target = tmp_path / "recording-figures.toml"
    target.write_text(metadata_text(metadata))
    verify(tmp_path, info)
    with pytest.raises(ValueError, match="selections"):
        verify(tmp_path, {"selection_sha256": "different-selection"})
    (tmp_path / "recorded-heart.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image bytes"):
        verify(tmp_path, info)


def test_actual_recording_selection_inventory():
    data = payload()
    from mms.paths import resolve_data_root

    info = provenance(data, resolve_data_root())
    assert info["counts"] == {
        "heart": {"3": 1432, "5": 1411},
        "interval": {"3": 602, "5": 1299},
        "pupil": {"0": 23091},
    }


def test_invalid_recording_does_not_create_outputs(tmp_path):
    root = tmp_path / "data" / "case-study" / "raw"
    root.mkdir(parents=True)
    for name, values in {
        "hr": {"heart_rate": [80, 90], "confidence": [1, 1]},
        "ibi": {"ibi": [800, 900]},
        "sed": {"pupil": [3, 4], "pupilQ": [1, 1]},
    }.items():
        pd.DataFrame({"reltime": [0, 1], "iSensor": [1.1, 1.9], **values}).to_csv(
            root / f"{name}_01.txt", sep=";", index=False
        )
    destination = tmp_path / "plots"
    with pytest.raises(ValueError, match="integer channel"):
        main(["--data-root", str(tmp_path / "data"), "--output-root", str(destination)])
    assert not destination.exists()

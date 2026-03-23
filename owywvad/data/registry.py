from __future__ import annotations

from owywvad.data.manifests import DatasetSpec


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "ubnormal": DatasetSpec(
        name="ubnormal",
        description="Open-set supervised anomaly benchmark.",
        official_homepage="https://www.crcv.ucf.edu/UBnormal/",
        manual_download=True,
        source_env="OWYWVAD_UBNORMAL_SOURCE",
        expected_splits=["train", "val", "test"],
    ),
    "shanghaitech": DatasetSpec(
        name="shanghaitech",
        description="Frame-level and pixel-level anomaly benchmark.",
        official_homepage="https://svip-lab.github.io/dataset/campus_dataset.html",
        manual_download=True,
        source_env="OWYWVAD_SHANGHAITECH_SOURCE",
        expected_splits=["train", "test"],
    ),
    "ucf_crime": DatasetSpec(
        name="ucf_crime",
        description="Long-video weakly supervised anomaly benchmark.",
        official_homepage="https://www.crcv.ucf.edu/projects/real-world/",
        manual_download=True,
        source_env="OWYWVAD_UCF_CRIME_SOURCE",
        expected_splits=["train", "test"],
    ),
}


def dataset_names() -> list[str]:
    return list(DATASET_REGISTRY)


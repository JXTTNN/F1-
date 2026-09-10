"""领域模型层（纯数据，无 IO）。

导出赛道 / 弯道模型、调教参数全集与症状枚举，供引擎、反馈、报告等模块复用。
"""

from __future__ import annotations

from .track import (
    ALL_TRACKS,
    Corner,
    CornerAnchor,
    Track,
    get_all_tracks,
    get_track_by_id,
    get_track_by_udp_id,
)
from .setup import (
    ALL_GROUPS,
    ALL_SETUP_FIELDS,
    SRC_OFFICIAL_GUIDE,
    SRC_UDP_PACKET5,
    CarSetup,
    SetupField,
    get_field,
    get_fields_by_group,
    validate_value,
)
from .symptoms import (
    DEFAULT_INTENSITY,
    INTENSITY_MAX,
    INTENSITY_MIN,
    SYMPTOM_INFO,
    Symptom,
    SymptomCategory,
    get_symptom_category,
    get_symptom_label,
    get_symptoms_by_category,
    validate_intensity,
)

__all__ = [
    # track.py
    "Track",
    "Corner",
    "CornerAnchor",
    "ALL_TRACKS",
    "get_all_tracks",
    "get_track_by_id",
    "get_track_by_udp_id",
    # setup.py
    "SetupField",
    "ALL_SETUP_FIELDS",
    "ALL_GROUPS",
    "CarSetup",
    "get_field",
    "get_fields_by_group",
    "validate_value",
    "SRC_UDP_PACKET5",
    "SRC_OFFICIAL_GUIDE",
    # symptoms.py
    "SymptomCategory",
    "Symptom",
    "SYMPTOM_INFO",
    "DEFAULT_INTENSITY",
    "INTENSITY_MIN",
    "INTENSITY_MAX",
    "get_symptoms_by_category",
    "get_symptom_label",
    "get_symptom_category",
    "validate_intensity",
]

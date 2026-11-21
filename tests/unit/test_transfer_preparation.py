"""Cross-platform metadata regression; no model fitting."""

import json

import numpy as np

from scripts.prepare_transfer_data import entity_identity


def test_numpy_group_key_is_json_native():
    entity = entity_identity((np.str_("Site 107"), np.int64(1)))
    assert type(entity[0]) is str and type(entity[1]) is int
    assert json.loads(json.dumps({"entity": entity})) == {"entity": ["Site 107", 1]}

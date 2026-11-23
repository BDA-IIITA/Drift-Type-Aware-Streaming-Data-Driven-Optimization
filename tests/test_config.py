from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from griddrift.config import load_project_config, parse_project_config


def test_base_configuration_is_valid():
    config = load_project_config(Path("configs/base.yaml"))
    assert config.data.bidding_zone == "DE-LU"
    assert config.assets.hydro.initial_energy_budget_mwh == 336.0
    assert config.assets.value_of_lost_load_eur_mwh > config.assets.emergency_grid.marginal_cost_eur_mwh


def test_voll_must_exceed_emergency_cost():
    raw = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["assets"]["value_of_lost_load_eur_mwh"] = raw["assets"]["emergency_grid"][
        "marginal_cost_eur_mwh"
    ]
    with pytest.raises(ValueError, match="VOLL"):
        parse_project_config(raw)

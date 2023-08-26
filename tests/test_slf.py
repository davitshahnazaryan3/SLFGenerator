import pytest
from pathlib import Path
import pandas as pd

from slf import SLF


path = Path(__file__).resolve().parent


@pytest.mark.slf
class SLFTest:
    main_path = path.parents[0] / "sample"
    export_path = main_path / "slf.json"

    @pytest.mark.parametrize(
        "component_data, correlation, edp, group", [
            # ("inventory.csv", None, "pfa", False),
            ("inventory.csv", "correlations.csv", "psd", True),
        ]
    )
    def test_backend(self, component_data, correlation, edp, group):
        if component_data is not None:
            component_data = pd.read_csv(self.main_path / component_data)
        if correlation is not None:
            correlation = pd.read_csv(self.main_path / correlation)

        model = SLF(
            component_data,
            edp,
            component=None,
            edp_range=None,
            edp_bin=None,
            correlation_tree=correlation,
            do_grouping=group,
        )

        out = model.generate_slfs()

        model.export_to_json(out, self.export_path)

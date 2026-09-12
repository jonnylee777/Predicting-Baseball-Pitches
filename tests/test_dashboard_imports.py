"""The dashboards must be importable the way ``streamlit run`` loads them.

``streamlit run dashboard/replay.py`` puts the script's own directory on
``sys.path`` and does not add the repository root, so a project import such as
``from dashboard.components import ...`` fails unless the script bootstraps the
root itself. Running the app from the repository root hides this, because the
working directory is then already importable -- which is exactly how it shipped
broken once.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "dashboard"


def run_as_streamlit_would(script: Path) -> subprocess.CompletedProcess:
    """Execute ``script`` with Streamlit's sys.path and an unrelated cwd."""

    probe = textwrap.dedent(
        f"""
        import sys, runpy
        root = {str(PROJECT_ROOT)!r}
        sys.path = [p for p in sys.path if p not in ("", root)]
        sys.path.insert(0, {str(DASHBOARD)!r})
        try:
            runpy.run_path({str(script)!r}, run_name="__main__")
        except ModuleNotFoundError as error:
            print(f"IMPORT_FAILURE {{error}}")
            raise SystemExit(3)
        except Exception:
            # Any other failure means the imports resolved; Streamlit APIs
            # cannot fully run outside a Streamlit runtime.
            raise SystemExit(0)
        raise SystemExit(0)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT.parent,
        timeout=300,
    )


class DashboardImportTests(unittest.TestCase):
    def test_dashboard_imports_without_the_repository_root_on_the_path(self) -> None:
        result = run_as_streamlit_would(DASHBOARD / "replay.py")
        self.assertNotIn("IMPORT_FAILURE", result.stdout, result.stdout)
        self.assertNotEqual(result.returncode, 3, result.stdout)

    def test_dashboard_bootstraps_the_root_before_project_imports(self) -> None:
        """The bootstrap has to precede the first project import to work."""

        source = (DASHBOARD / "replay.py").read_text(encoding="utf-8")
        bootstrap = source.index("sys.path.insert(0, str(PROJECT_ROOT))")
        first_project_import = min(
            source.index("from dashboard."),
            source.index("from pitch_prediction."),
        )
        self.assertLess(bootstrap, first_project_import)


if __name__ == "__main__":
    unittest.main()

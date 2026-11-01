import json
import os
import subprocess
import sys
from pathlib import Path


def test_generate_plan_validate_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONIOENCODING="utf-8")

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "seabed_planner", *map(str, args)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    mission, plan = tmp_path / "mission.json", tmp_path / "plan.json"
    run("generate", "--targets", 8, "--seed", 42, "--output", mission)
    run(
        "plan",
        mission,
        "--iterations",
        3,
        "--ants",
        4,
        "--backend",
        "cpu",
        "--serial",
        "--output",
        plan,
    )
    assert "Valid:" in run("validate", plan)
    assert len(json.loads(plan.read_text(encoding="utf-8"))["mission"]["targets"]) == 8

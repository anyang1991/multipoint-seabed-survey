# Support and contributions

Use [GitHub Issues](https://github.com/anyang1991/multipoint-seabed-survey/issues) for
questions, reproducible problems and feature proposals. For a problem report, include the
software version, command, input or minimal example, random seed, expected result and
observed result. Include `python -m seabed_planner hardware` output for execution issues.

To contribute:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m ruff check src tests experiments benchmarks
```

Submit a focused pull request with a description of the change and its validation.
Changes to numerical methods should document the model, parameters and reproducible
comparison. Preserve original files under `legacy/matlab/` and their provenance hashes.

# Man usuk — execution guide

See [README.md](README.md) for setup, inputs and the default v2 method.

Generate the current submission from this project directory:

```powershell
python -m solution.predict --team "Man usuk" --method v2
python evaluation/validate_submission.py submission/Man_usuk.csv
```

The local environment used for development has the versions in `requirements-local.txt`.
For an isolated environment use `.venv\Scripts\python.exe` after installing that file.
The generation command accepts `--data-dir` and `--output-dir`.

To reproduce the older v1 method without replacing the current file:

```powershell
python -m solution.predict --team "Man usuk" --method v1 --output-dir local-logs/v1
```

To test and package public source (no restricted assets):

```powershell
python -m unittest discover -s tests -v
python package_source.py
```

Do not select methods or tune parameters using the public five-minute ground truth.
V2 is checked with synthetic scenarios, format validation and repeat-run hashes;
no v2 public score has been used to claim or select an improvement.
The submission form is a separate manual step. Nothing in these commands sends it.

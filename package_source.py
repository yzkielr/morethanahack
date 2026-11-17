"""Create a public-source ZIP using an explicit allowlist, never the data directory."""
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    fixed = ['games.py', 'LICENSE', '.gitignore', '.gitattributes', 'requirements-local.txt',
             'README.md', 'UPSTREAM_README.md', 'RUN_PROJECT.md', 'Man_usuk.md', 'Man_usuk.pdf', 'package_source.py',
             'docs/technical-v1.md', '.github/workflows/tests.yml',
             'CHALLENGE.md', 'DATA.md', 'EVALUATION.md', 'RULES.md', 'SUBMISSION.md',
             'starter/baseline.py', 'starter/load_smart_tagging.py',
             'evaluation/possession_metrics.py', 'evaluation/run_evaluation.py',
             'evaluation/validate_submission.py',
             'evaluation/gt/BAR-ATH_possession_gt_public.csv']
    source = fixed + [str(p.relative_to(root)).replace('\\', '/')
                      for folder in ('solution', 'tests')
                      for p in sorted((root/folder).glob('*.py'))]
    release = root/'release'
    release.mkdir(exist_ok=True)
    target = release/'Man_usuk_source.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(root/name, name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(name.startswith(('data/', 'submission/', 'local-logs/', '.venv/'))
                       for name in archive.namelist())
    print('Public source archive created and checked; no licensed match assets included.')


if __name__ == '__main__':
    main()

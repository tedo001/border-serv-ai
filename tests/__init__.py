"""Test package marker.

Not decoration: without it ``tests`` is a namespace package here, and a
regular package of the same name anywhere on ``sys.path`` wins the import.
``ultralytics`` - an optional runtime dependency of this project - pulls in a
distribution that installs a top-level ``tests`` package into site-packages,
which silently shadowed this suite's ``tests.conftest`` the moment the 'torch'
extra was installed. A marker file makes this directory a regular package
again, and ``pythonpath = ["."]`` puts it first.
"""

# Deployment comparisons

These scripts compare the library against the two deployments the paper reports on. They import the deployments' own code (`src.core.standing...`) and run inside the in-home pipeline's container, so they do not run from this repository alone. They are kept here as the record of how the library was checked against a system that had its own standing implementation; the self-contained test suite is in `tests/`.

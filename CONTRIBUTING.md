# Contributing to Bionium-X

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:
- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Developing

To set up a development environment:

1. Fork the repo and create your branch from `main`.
2. Install the library with development dependencies:
   ```bash
   pip install -e .[dev,science,ml]
   ```
3. Install the pre-commit hooks to ensure formatting checks pass automatically:
   ```bash
   pre-commit install
   ```

## Testing

Before submitting a pull request, run the test suite to ensure everything is functioning correctly:
```bash
pytest bioniumx/tests/
```

## Pull Requests

1. Run the test suite and ensure it passes.
2. Update the `README.md` and Sphinx documentation with details of changes to the interface.
3. Open a pull request. Your PR will trigger our automated CI suite which checks for formatting, syntax errors, and runs all tests.

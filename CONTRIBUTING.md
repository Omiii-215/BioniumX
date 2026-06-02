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

## Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for our commit messages. This leads to more readable messages that are easy to follow when looking through the project history.

A typical commit message should look like this:
```
<type>(<scope>): <subject>
```
Examples of `<type>` include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

## Review Process and Timeline

As participants in the **GirlScript Summer of Code (GSSoC)**, we are committed to providing a fast and transparent review process.

- **Issue Assignment**: We aim to respond to issue assignment requests within **24 hours**. If a request is left unanswered for 48+ hours, it may be auto-assigned by the program platform.
- **Pull Request Review**: All pull requests will be thoroughly reviewed before merging. We ask that contributors are able to walk through and explain their implementation. Expect feedback on your PR within **24-48 hours**.
- **Merging**: CI/CD checks must pass, and your PR must receive approval from at least one core maintainer before it can be merged into the `main` branch.

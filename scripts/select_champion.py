"""Prepare a validation candidate under the Week 5 rule; never freeze or read holdout."""

from scripts.compare_validation import compare


if __name__ == "__main__":
    compare(diagnostics=False)

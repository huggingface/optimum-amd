import pytest


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    # from pdb import set_trace; set_trace()
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)

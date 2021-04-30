"""Returns tuple of semantic version"""

import pkg_resources


def _get_version_from_pyproject():
    import toml

    with open("pyproject.toml") as f:
        file_contents = f.read()

    return toml.loads(file_contents)["tool"]["poetry"]["version"]


def _version():
    try:
        ver = pkg_resources.get_distribution("covid19uk").version
    except pkg_resources.DistributionNotFound:
        ver = _get_version_from_pyproject()
    return ver.split(".")


MAJOR, MINOR, PATCH = _version()
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"

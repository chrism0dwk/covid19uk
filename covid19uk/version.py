"""Returns tuple of semantic version"""

import pkg_resources


def version():
    ver = pkg_resources.get_distribution("covid19uk").version
    return ver.split(".")

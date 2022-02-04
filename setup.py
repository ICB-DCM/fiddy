from setuptools import setup, find_packages
import sys
import os
import re


org = "ICB-DCM"
repo = "fiddy"


def read(fname):
    """Read a file."""
    return open(fname).read()


def absolute_links(txt):
    """Replace relative links by absolute links, on GitHub."""

    raw_base = f"(https://raw.githubusercontent.com/{org}/{repo}/main/"
    embedded_base = f"(https://github.com/{org}/{repo}/tree/main/"
    # iterate over links
    for var in re.findall(r"\[.*?\]\((?!http).*?\)", txt):
        if re.match(r".*?.(png|svg)\)", var):
            # link to raw file
            rep = var.replace("(", raw_base)
        else:
            # link to github embedded file
            rep = var.replace("(", embedded_base)
        txt = txt.replace(var, rep)
    return txt


minimum_python_version = "3.8.0"  # for NumPy
if sys.version_info < tuple(map(int, minimum_python_version.split("."))):
    sys.exit(f"{repo} requires Python >= {minimum_python_version}")

# read version from file
__version__ = ""
version_file = os.path.join(repo, "version.py")
# sets __version__
exec(read(version_file))  # pylint: disable=W0122 # nosec

# project metadata
# noinspection PyUnresolvedReferences
setup(
    name=repo,
    version=__version__,
    description="Finite difference methods.",
    long_description=absolute_links(read("README.md")),
    long_description_content_type="text/markdown",
    url=f"https://github.com/{org}/{repo}",
    packages=find_packages(exclude=["doc*", "test*"]),
    install_requires=[
        "joblib",
        "numpy",
        "pandas",
    ],
    include_package_data=True,
    python_requires=f">={minimum_python_version}",
    extras_require={
        "tests": [
            "pytest",
            "sympy",
            "more-itertools",
        ],
        "amici": [
            "amici",
            "petab",
        ],
        "pypesto": [
            # TODO setup.cfg with dependency on amici, petab
            "pypesto",
        ],
        "doc": [
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
            "mock",
        ],
    },
)

"""The manifest must match what the code actually imports.

This exists because it did not. `tests/test_properties.py` imports hypothesis and
the dev extra declared only pytest, so following the README's own install command
produced an environment where pytest aborted collection: not 154 failures, which
someone would notice, but 154 tests silently never running, including the property
search that AGENTS.md names as the instrument that finds what hand-probing misses.

A packaging bug of that shape is invisible to every other test in this repo,
because every other test runs in an environment where the missing dependency
happens to be installed. So the check has to be about the manifest rather than
about behaviour.

Two properties are asserted:

    every third-party module imported anywhere is declared as a dependency;
    the RUNTIME package imports nothing beyond numpy and the standard library,
    which is the scope boundary in AGENTS.md: this has to run on an edge node.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from importlib.metadata import packages_distributions

try:  # tomllib is stdlib only from 3.11, and this project declares >=3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "standing"

# Modules that are part of this repository rather than a dependency.
LOCAL = {"standing"}

# Modules provided by the SYSTEM UNDER TEST, not by this project. The `*_cml.py`
# scripts run inside the deployed CML container and import that pipeline's own
# code in order to compare it with ours. They are integration scripts, never
# collected by pytest (their names do not start with test_), and declaring these
# as dependencies would be wrong: they are not installable, they are the thing
# being differentially tested.
HOST_PROVIDED = {"src", "experiments", "standing_properties"}

# Modules that are stdlib on THIS interpreter but were added later than some
# supported versions. `sys.stdlib_module_names` reflects the running Python, so a
# check that consults it cannot see a problem that only appears at the declared
# floor. This repository shipped exactly that: test_packaging.py imported tomllib
# unconditionally while the manifest declared >=3.10, and on 3.10 the file errors
# at collection and takes the whole suite with it, which is the same failure shape
# as the undeclared hypothesis dependency it was written to prevent.
STDLIB_SINCE = {
    "tomllib": (3, 11),
    "graphlib": (3, 9),
    "zoneinfo": (3, 9),
    "importlib.resources": (3, 9),
}

# What to depend on instead, below the version where the module became stdlib.
BACKPORTS = {"tomllib": "tomli"}


def _declared_distributions() -> set[str]:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = manifest["project"]
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    names = set()
    for requirement in declared:
        # "numpy>=1.24" -> "numpy"; good enough, there are no markers or extras here
        name = requirement.split(";")[0]
        for separator in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(separator)[0]
        names.add(_normalise(name))
    return names


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _third_party(modules: set[str]) -> set[str]:
    """Imports that must appear in the manifest.

    STDLIB_SINCE is consulted here and not only in the requires-python test,
    because `sys.stdlib_module_names` describes the RUNNING interpreter. On 3.12
    tomllib is in it; on 3.10, the declared floor, it is not, so this classified a
    stdlib module as an undeclared third-party import and the suite went red at
    exactly the version the package promises to support. Worse, the fix it
    suggests is to add tomllib to pyproject.toml, which is not installable.
    Whether a backport is declared for the floor is the other test's job.
    """
    return {
        module
        for module in modules
        if module not in sys.stdlib_module_names
        and module not in STDLIB_SINCE
        and module not in LOCAL
        and module not in HOST_PROVIDED
        and not (ROOT / "tests" / f"{module}.py").exists()
    }


def _distribution_for(module: str) -> str:
    """Which installed distribution provides this import name."""
    mapping = packages_distributions()
    provided = mapping.get(module)
    return _normalise(provided[0]) if provided else _normalise(module)


def _python_files() -> list[pathlib.Path]:
    return sorted(
        [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py"), *(ROOT / "examples").rglob("*.py")]
    )


def test_the_integration_scripts_are_never_collected_as_tests():
    """They import the deployment's modules, so importing them here would error.

    Their filenames are what keeps them out of collection, which is fragile
    enough to be worth asserting.
    """
    for path in (ROOT / "tests").glob("*_cml.py"):
        assert not path.name.startswith("test_"), (
            f"{path.name} would be collected by pytest, and it imports modules "
            f"that only exist inside the deployment's container"
        )


def test_every_third_party_import_is_declared():
    declared = _declared_distributions()
    undeclared: dict[str, list[str]] = {}
    for path in _python_files():
        for module in _third_party(_top_level_imports(path)):
            if _distribution_for(module) not in declared:
                undeclared.setdefault(module, []).append(str(path.relative_to(ROOT)))
    assert not undeclared, (
        f"imported but not declared in pyproject.toml: {undeclared}. Following the "
        f"README's install command would produce an environment where these files "
        f"cannot even be collected."
    )


def test_the_runtime_package_depends_on_numpy_and_nothing_else():
    """The scope boundary from AGENTS.md, asserted rather than asked for.

    A new dependency in src/ is not a small convenience: it is a wheel someone has
    to build on whatever node this runs on.
    """
    imported: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        imported |= _third_party(_top_level_imports(path))
    assert imported <= {"numpy"}, (
        f"the runtime package imports {sorted(imported - {'numpy'})}. numpy and the "
        f"standard library are the runtime; pytest and hypothesis are dev only."
    )


def test_the_console_script_points_at_something_real():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for name, target in manifest["project"].get("scripts", {}).items():
        module_path, _, attribute = target.partition(":")
        module_file = ROOT / "src" / pathlib.Path(*module_path.split(".")).with_suffix(".py")
        assert module_file.exists(), f"script {name!r} points at missing {module_path}"
        tree = ast.parse(module_file.read_text(), filename=str(module_file))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert attribute in defined, f"script {name!r} names {attribute!r}, which is not defined"


def test_the_readme_install_command_matches_the_extra_it_promises():
    """The README names what `[dev]` provides; the manifest must actually provide it."""
    readme = (ROOT / "README.md").read_text().lower()
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = {_normalise(r.split(">")[0].split("=")[0]) for r in
           manifest["project"]["optional-dependencies"].get("dev", [])}
    for promised in ("pytest", "hypothesis"):
        if promised in readme:
            assert promised in dev, (
                f"the README mentions {promised} but the dev extra does not declare it"
            )


def _declared_floor() -> tuple[int, int]:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirement = manifest["project"]["requires-python"].strip()
    digits = requirement.lstrip(">=~^ ")
    major, _, minor = digits.partition(".")
    return int(major), int(minor.split(".")[0] or 0)


def test_no_module_is_imported_below_the_python_version_that_provides_it():
    """A version-gated stdlib import needs a declared backport, or a higher floor.

    Everything else in this file consults `sys.stdlib_module_names`, which
    describes the interpreter running the tests rather than the oldest one the
    manifest promises to support. That blind spot is not hypothetical: it shipped.
    """
    floor = _declared_floor()
    declared = _declared_distributions()
    offenders: dict[str, str] = {}
    for path in _python_files():
        for module in _top_level_imports(path):
            since = STDLIB_SINCE.get(module)
            if since is None or since <= floor:
                continue
            backport = BACKPORTS.get(module)
            if backport is None or _normalise(backport) not in declared:
                offenders[module] = (
                    f"{path.relative_to(ROOT)} imports {module}, stdlib only from "
                    f"{since[0]}.{since[1]}, but requires-python is "
                    f"{floor[0]}.{floor[1]}"
                    + (f"; declare {backport} for older versions" if backport else "")
                )
    assert not offenders, offenders


def test_a_module_that_is_stdlib_only_on_newer_pythons_is_not_called_third_party(
    monkeypatch,
):
    """The bug this file exists to prevent, on its fourth appearance.

    `sys.stdlib_module_names` describes the interpreter running the tests, so a
    module that is stdlib on 3.12 and a backport on 3.10 was classified as an
    undeclared dependency at the declared floor. Simulate the older interpreter
    rather than requiring one to be installed.
    """
    older = frozenset(n for n in sys.stdlib_module_names if n != "tomllib")
    monkeypatch.setattr(sys, "stdlib_module_names", older)
    assert "tomllib" not in _third_party({"tomllib"})
    assert "numpy" in _third_party({"numpy"})

"""Project-stack detection for the bare `tj onboard` outro (#85)."""
from __future__ import annotations

from tokenjam.cli.onboard_detect import (
    declared_package_names,
    detect_stack,
    install_hint,
)


def _write(tmp_path, name, content):
    (tmp_path / name).write_text(content)


# --- declared_package_names: pyproject.toml ---------------------------------


def test_pep621_dependencies_are_detected(tmp_path):
    _write(tmp_path, "pyproject.toml", """
[project]
dependencies = ["langchain>=0.2", "requests==2.31.0"]
""")
    names = declared_package_names(tmp_path)
    assert "langchain" in names
    assert "requests" in names


def test_pep621_optional_dependencies_are_detected(tmp_path):
    _write(tmp_path, "pyproject.toml", """
[project]
dependencies = []
[project.optional-dependencies]
agents = ["crewai>=0.28"]
""")
    assert "crewai" in declared_package_names(tmp_path)


def test_poetry_dependencies_are_detected_and_python_excluded(tmp_path):
    _write(tmp_path, "pyproject.toml", """
[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.0"
""")
    names = declared_package_names(tmp_path)
    assert "openai" in names
    assert "python" not in names


def test_poetry_group_dependencies_are_detected(tmp_path):
    _write(tmp_path, "pyproject.toml", """
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
""")
    assert "pytest" in declared_package_names(tmp_path)


def test_malformed_pyproject_does_not_raise(tmp_path):
    _write(tmp_path, "pyproject.toml", "not [ valid toml")
    assert declared_package_names(tmp_path) == set()


# --- declared_package_names: requirements*.txt ------------------------------


def test_requirements_txt_names_are_extracted_with_version_specifiers(tmp_path):
    _write(tmp_path, "requirements.txt", """
# a comment
boto3>=1.34
litellm==1.40.0
-e ./local-pkg
git+https://example.com/foo.git
https://example.com/bar.whl
""")
    names = declared_package_names(tmp_path)
    assert "boto3" in names
    assert "litellm" in names
    assert not any("local-pkg" in n or "foo" in n or "bar" in n for n in names)


def test_multiple_requirements_files_are_all_scanned(tmp_path):
    _write(tmp_path, "requirements.txt", "openai\n")
    _write(tmp_path, "requirements-dev.txt", "pyautogen\n")
    names = declared_package_names(tmp_path)
    assert "openai" in names
    assert "pyautogen" in names


def test_extras_bracket_is_stripped_from_requirement_name(tmp_path):
    _write(tmp_path, "requirements.txt", "llama-index[vector-stores]>=0.10\n")
    assert "llama-index" in declared_package_names(tmp_path)


# --- declared_package_names: setup.cfg --------------------------------------


def test_setup_cfg_install_requires_is_detected(tmp_path):
    _write(tmp_path, "setup.cfg", """
[options]
install_requires =
    anthropic>=0.25
    ag2
""")
    names = declared_package_names(tmp_path)
    assert "anthropic" in names
    assert "ag2" in names


def test_setup_cfg_extras_require_is_detected(tmp_path):
    _write(tmp_path, "setup.cfg", """
[options.extras_require]
langgraph =
    langgraph>=0.1
""")
    assert "langgraph" in declared_package_names(tmp_path)


# --- normalization -----------------------------------------------------------


def test_names_are_pep503_normalized(tmp_path):
    _write(tmp_path, "requirements.txt", "Google_Generativeai==0.5\n")
    assert "google-generativeai" in declared_package_names(tmp_path)


def test_empty_project_dir_yields_no_names(tmp_path):
    assert declared_package_names(tmp_path) == set()


# --- detect_stack: end-to-end matching --------------------------------------


def test_no_manifests_yields_no_matches(tmp_path):
    assert detect_stack(tmp_path) == []


def test_langchain_dependency_yields_langchain_match(tmp_path):
    _write(tmp_path, "requirements.txt", "langchain>=0.2\n")
    matches = detect_stack(tmp_path)
    keys = [m.key for m in matches]
    assert keys == ["langchain"]
    m = matches[0]
    assert m.patch_call == "patch_langchain()"
    assert "patch_langchain" in m.import_line
    assert install_hint(m) == "pip install 'tokenjam[langchain]'"


def test_provider_without_extra_has_no_install_hint(tmp_path):
    _write(tmp_path, "requirements.txt", "anthropic>=0.25\n")
    matches = detect_stack(tmp_path)
    assert matches[0].key == "anthropic"
    assert install_hint(matches[0]) is None


def test_multiple_matches_are_ordered_providers_then_frameworks_alphabetically(tmp_path):
    _write(tmp_path, "requirements.txt", "openai\nlanggraph\nanthropic\ncrewai\n")
    matches = detect_stack(tmp_path)
    keys = [m.key for m in matches]
    assert keys == ["anthropic", "openai", "crewai", "langgraph"]


def test_google_genai_and_generativeai_both_map_to_gemini_without_duplicate(tmp_path):
    _write(tmp_path, "requirements.txt", "google-genai\ngoogle-generativeai\n")
    matches = detect_stack(tmp_path)
    assert [m.key for m in matches] == ["gemini"]


def test_llama_index_variants_both_map_to_llamaindex_without_duplicate(tmp_path):
    _write(tmp_path, "requirements.txt", "llama-index\nllama-index-core\n")
    matches = detect_stack(tmp_path)
    assert [m.key for m in matches] == ["llamaindex"]


def test_autogen_fork_ag2_is_detected(tmp_path):
    _write(tmp_path, "requirements.txt", "ag2>=0.3\n")
    matches = detect_stack(tmp_path)
    assert [m.key for m in matches] == ["autogen"]


def test_litellm_detected_alongside_providers(tmp_path):
    _write(tmp_path, "requirements.txt", "litellm\nopenai\nanthropic\n")
    matches = detect_stack(tmp_path)
    keys = {m.key for m in matches}
    assert {"litellm", "openai", "anthropic"} <= keys


def test_unrelated_dependencies_yield_no_matches(tmp_path):
    _write(tmp_path, "requirements.txt", "flask\nrequests\nnumpy\n")
    assert detect_stack(tmp_path) == []

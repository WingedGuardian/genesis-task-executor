"""Tests for genesis_task_executor.prompts module."""

from __future__ import annotations

import pytest

from genesis_task_executor.prompts import (
    ADVERSARIAL_SYSTEM,
    DUE_DILIGENCE_TRIAGE,
    EXIT_GATE_TEMPLATE,
    FRESH_EYES_SYSTEM,
    PLAN_REVIEW_SYSTEM,
    PLAN_SYSTEM,
    RESEARCH_TEMPLATE,
    RETROSPECTIVE_SYSTEM,
    STEP_EXECUTION_SYSTEM,
    render_template,
)


class TestRenderTemplate:

    def test_single_replacement(self):
        template = "Hello {{name}}!"
        result = render_template(template, name="World")
        assert result == "Hello World!"

    def test_multiple_replacements(self):
        template = "{{a}} and {{b}}"
        result = render_template(template, a="X", b="Y")
        assert result == "X and Y"

    def test_missing_key_uses_none(self):
        """Keys present in template but not in kwargs remain as-is;
        keys present in kwargs but None get replaced with '(none)'."""
        template = "Value: {{key}}"
        result = render_template(template, key=None)
        assert result == "Value: (none)"

    def test_unreplaced_placeholder_stays(self):
        template = "{{present}} {{absent}}"
        result = render_template(template, present="here")
        assert result == "here {{absent}}"

    def test_empty_string_replacement(self):
        """Empty string is treated as falsy by `or` in render_template,
        so it gets replaced with '(none)'."""
        template = "Before{{x}}After"
        result = render_template(template, x="")
        assert result == "Before(none)After"


class TestPromptConstants:

    ALL_PROMPTS = [
        PLAN_SYSTEM,
        PLAN_REVIEW_SYSTEM,
        STEP_EXECUTION_SYSTEM,
        FRESH_EYES_SYSTEM,
        ADVERSARIAL_SYSTEM,
        EXIT_GATE_TEMPLATE,
        RESEARCH_TEMPLATE,
        DUE_DILIGENCE_TRIAGE,
        RETROSPECTIVE_SYSTEM,
    ]

    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_prompt_is_nonempty_string(self, prompt):
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 20  # Meaningful content

    def test_exit_gate_has_placeholders(self):
        assert "{{step_description}}" in EXIT_GATE_TEMPLATE
        assert "{{error_text}}" in EXIT_GATE_TEMPLATE
        assert "{{research_conclusion}}" in EXIT_GATE_TEMPLATE
        assert "{{concrete_blockers}}" in EXIT_GATE_TEMPLATE
        assert "{{prior_rejections}}" in EXIT_GATE_TEMPLATE

    def test_research_template_has_placeholders(self):
        assert "{{step_description}}" in RESEARCH_TEMPLATE
        assert "{{error_text}}" in RESEARCH_TEMPLATE
        assert "{{prior_attempts}}" in RESEARCH_TEMPLATE
        assert "{{due_diligence_results}}" in RESEARCH_TEMPLATE

    def test_due_diligence_has_placeholders(self):
        assert "{{step_description}}" in DUE_DILIGENCE_TRIAGE
        assert "{{error_text}}" in DUE_DILIGENCE_TRIAGE

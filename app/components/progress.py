"""
progress.py
Componente Streamlit para exibição do progresso do pipeline.

Exibe o status de cada etapa do processamento com indicador
visual de modo (local/cloud) e mensagens de progresso.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st

logger = logging.getLogger(__name__)


# ── Enums e Dataclasses ───────────────────────────────────────────────────────


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class StepMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    DEGRADED = "degradado"


@dataclass
class PipelineStep:
    """
    Representa uma etapa do pipeline com seu status atual.

    Atributos:
        name:     Nome da etapa.
        label:    Rótulo exibido na interface.
        status:   Status atual da etapa.
        mode:     Modo de execução (local ou cloud).
        message:  Mensagem de status atual.
        messages: Histórico de mensagens da etapa.
    """

    name: str
    label: str
    status: StepStatus = StepStatus.PENDING
    mode: StepMode | None = None
    message: str = ""
    messages: list[str] = field(default_factory=list)


# ── Etapas do Pipeline ────────────────────────────────────────────────────────


def get_pipeline_steps() -> list[PipelineStep]:
    """Retorna a lista de etapas do pipeline na ordem de execução."""
    return [
        PipelineStep("separator", "🎛️ Separação de Instrumentos"),
        PipelineStep("chords", "🎸 Detecção de Acordes"),
        PipelineStep("melody", "🎵 Detecção de Melodia"),
        PipelineStep("transcriber", "🎤 Transcrição da Letra"),
        PipelineStep("aligner", "🔗 Alinhamento"),
        PipelineStep("formatter", "📝 Formatação da Cifra"),
    ]


# ── Ícones de Status ──────────────────────────────────────────────────────────

STATUS_ICONS = {
    StepStatus.PENDING: "⏳",
    StepStatus.RUNNING: "🔄",
    StepStatus.SUCCESS: "✅",
    StepStatus.WARNING: "⚠️",
    StepStatus.ERROR: "❌",
}

MODE_BADGES = {
    StepMode.LOCAL: "🖥️ local",
    StepMode.CLOUD: "☁️ cloud",
    StepMode.DEGRADED: "🔴 degradado",
}


# ── Renderização ──────────────────────────────────────────────────────────────


def render_pipeline_progress(steps: list[PipelineStep]) -> None:
    """
    Renderiza o painel de progresso do pipeline no Streamlit.

    Exibe cada etapa com seu status, modo de execução e
    última mensagem de progresso.

    Args:
        steps: Lista de etapas do pipeline com seus status atuais.
    """
    st.markdown("### ⚙️ Progresso do Processamento")

    concluidas = sum(1 for s in steps if s.status == StepStatus.SUCCESS)
    total = len(steps)
    progresso = concluidas / total if total > 0 else 0.0

    st.progress(progresso, text=f"{concluidas}/{total} etapas concluídas")
    st.divider()

    for step in steps:
        _render_step(step)


def _render_step(step: PipelineStep) -> None:
    """
    Renderiza uma etapa individual do pipeline.

    Args:
        step: Etapa do pipeline a ser renderizada.
    """
    icone = STATUS_ICONS[step.status]
    badge = MODE_BADGES.get(step.mode, "") if step.mode else ""

    col1, col2 = st.columns([4, 1])

    with col1:
        if step.status == StepStatus.RUNNING:
            st.markdown(f"**{icone} {step.label}**")
            if step.message:
                st.caption(f"↳ {step.message}")
        elif step.status == StepStatus.SUCCESS:
            st.markdown(f"{icone} {step.label}")
            if step.message:
                st.caption(f"↳ {step.message}")
        elif step.status == StepStatus.ERROR:
            st.markdown(f"{icone} ~~{step.label}~~")
            if step.message:
                st.caption(f"↳ {step.message}")
        elif step.status == StepStatus.WARNING:
            st.markdown(f"{icone} {step.label}")
            if step.message:
                st.caption(f"↳ {step.message}")
        else:
            st.markdown(f"{icone} {step.label}")

    with col2:
        if badge:
            st.caption(badge)


def render_quality_indicator(steps: list[PipelineStep]) -> None:
    """
    Renderiza o indicador geral de qualidade do processamento.

    Analisa os modos de execução das etapas e exibe um
    indicador visual de qualidade geral.

    Args:
        steps: Lista de etapas concluídas do pipeline.
    """
    modos = [s.mode for s in steps if s.mode is not None]

    if not modos:
        return

    if all(m == StepMode.LOCAL for m in modos):
        st.success("🟢 **Alta qualidade** — todas as etapas rodaram localmente")
    elif StepMode.DEGRADED in modos:
        st.error("🔴 **Modo básico** — algumas etapas em modo degradado")
    elif StepMode.CLOUD in modos:
        st.warning("🟡 **Qualidade média** — algumas etapas usaram fallback cloud")
    else:
        st.info("🔵 Processamento concluído")


def update_step(
    steps: list[PipelineStep],
    name: str,
    status: StepStatus,
    message: str = "",
    mode: StepMode | None = None,
) -> list[PipelineStep]:
    """
    Atualiza o status de uma etapa do pipeline.

    Args:
        steps:   Lista de etapas do pipeline.
        name:    Nome da etapa a atualizar.
        status:  Novo status da etapa.
        message: Mensagem de status.
        mode:    Modo de execução (local ou cloud).

    Returns:
        Lista de etapas atualizada.
    """
    for step in steps:
        if step.name == name:
            step.status = status
            step.message = message
            if mode:
                step.mode = mode
            if message:
                step.messages.append(message)
            break

    return steps

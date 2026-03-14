"""
main.py
Ponto de entrada principal do MelodyTab AI.

Integra todos os componentes da interface e módulos do pipeline
em uma aplicação Streamlit funcional.
"""

import logging
from pathlib import Path

import streamlit as st

from app.components.chord_display import render_chord_display, render_chord_stats
from app.components.progress import (
    StepMode,
    StepStatus,
    get_pipeline_steps,
    render_pipeline_progress,
    render_quality_indicator,
    update_step,
)
from app.components.uploader import render_uploader
from app.pipeline.aligner import align
from app.pipeline.chord_detector import detect_chords
from app.pipeline.formatter import format_chord
from app.pipeline.melody_detector import detect_melody
from app.pipeline.separator import SeparationMode, separate
from app.pipeline.transcriber import transcribe
from app.utils.logger import setup_logging

# Configura o logging na inicialização
setup_logging()
logger = logging.getLogger(__name__)


# ── Configuração da Página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="MelodyTab AI",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Interface Principal ───────────────────────────────────────────────────────


def main() -> None:
    """Função principal da aplicação Streamlit."""

    _render_header()

    # Upload do arquivo de áudio
    audio_path = render_uploader()

    if audio_path is None:
        return

    # Botão para iniciar o processamento
    st.markdown("---")
    if not st.button("🎵 Gerar Cifra", type="primary", use_container_width=True):
        return

    # Inicia o pipeline
    _run_pipeline(audio_path)


def _render_header() -> None:
    """Renderiza o cabeçalho da aplicação."""
    st.title("🎸 MelodyTab AI")
    st.markdown(
        "Gere cifras melódicas automaticamente a partir de arquivos de áudio "
        "usando Inteligência Artificial."
    )
    st.divider()


# ── Pipeline de Processamento ─────────────────────────────────────────────────


def _run_pipeline(audio_path: Path) -> None:
    """
    Executa o pipeline completo de geração de cifra.

    Coordena todas as etapas do processamento, atualizando
    o progresso na interface em tempo real.

    Args:
        audio_path: Caminho para o arquivo de áudio enviado.
    """
    steps = get_pipeline_steps()
    progress_area = st.empty()

    # Renderiza o progresso inicial
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Etapa 1: Separação de Instrumentos ───────────────────────────────────
    steps = update_step(
        steps, "separator", StepStatus.RUNNING, "Separando instrumentos com Demucs..."
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    separation = separate(audio_path)

    if not separation.success:
        steps = update_step(
            steps, "separator", StepStatus.ERROR, f"Falha: {separation.error}"
        )
        with progress_area.container():
            render_pipeline_progress(steps)
        st.error("❌ Falha na separação de instrumentos. Verifique o arquivo de áudio.")
        return

    modo_sep = (
        StepMode.LOCAL if separation.mode == SeparationMode.LOCAL else StepMode.CLOUD
    )
    steps = update_step(
        steps,
        "separator",
        StepStatus.SUCCESS,
        f"Faixa harmônica: '{separation.harmonic_source}'",
        mode=modo_sep,
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Etapa 2: Detecção de Acordes ──────────────────────────────────────────
    steps = update_step(
        steps, "chords", StepStatus.RUNNING, "Analisando acordes com librosa..."
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    chord_result = detect_chords(separation.harmonic)

    if not chord_result.success:
        steps = update_step(
            steps,
            "chords",
            StepStatus.WARNING,
            "Detecção de acordes falhou — continuando sem acordes",
        )
    else:
        steps = update_step(
            steps,
            "chords",
            StepStatus.SUCCESS,
            f"{len(chord_result.chords)} acordes detectados",
            mode=StepMode.LOCAL,
        )
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Etapa 3: Detecção de Melodia ──────────────────────────────────────────
    steps = update_step(
        steps, "melody", StepStatus.RUNNING, "Detectando melodia com pyin..."
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    melody_result = detect_melody(separation.vocals)

    if not melody_result.success:
        steps = update_step(
            steps,
            "melody",
            StepStatus.WARNING,
            "Detecção de melodia falhou — continuando sem melodia",
        )
    else:
        steps = update_step(
            steps,
            "melody",
            StepStatus.SUCCESS,
            f"{len(melody_result.notes)} notas detectadas",
            mode=StepMode.LOCAL,
        )
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Etapa 4: Transcrição da Letra ─────────────────────────────────────────
    steps = update_step(
        steps,
        "transcriber",
        StepStatus.RUNNING,
        "Transcrevendo letra com Groq Whisper...",
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    transcription = transcribe(separation.vocals)

    if not transcription.success:
        steps = update_step(
            steps, "transcriber", StepStatus.ERROR, f"Falha: {transcription.error}"
        )
        with progress_area.container():
            render_pipeline_progress(steps)
        st.error("❌ Falha na transcrição. Verifique sua chave da API Groq.")
        return

    steps = update_step(
        steps,
        "transcriber",
        StepStatus.SUCCESS,
        f"{len(transcription.words)} palavras transcritas",
        mode=StepMode.CLOUD,
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Etapa 5: Alinhamento ──────────────────────────────────────────────────
    steps = update_step(
        steps,
        "aligner",
        StepStatus.RUNNING,
        "Sincronizando letra, acordes e melodia...",
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    alignment = align(transcription, chord_result, melody_result)

    if not alignment.success:
        steps = update_step(
            steps, "aligner", StepStatus.ERROR, f"Falha: {alignment.error}"
        )
        with progress_area.container():
            render_pipeline_progress(steps)
        st.error("❌ Falha no alinhamento.")
        return

    steps = update_step(
        steps,
        "aligner",
        StepStatus.SUCCESS,
        f"{len(alignment.lines)} linhas geradas",
        mode=StepMode.LOCAL,
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Etapa 6: Formatação da Cifra ──────────────────────────────────────────
    steps = update_step(
        steps, "formatter", StepStatus.RUNNING, "Formatando cifra com Groq LLaMA..."
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    formatted = format_chord(alignment)

    if not formatted.success:
        steps = update_step(
            steps, "formatter", StepStatus.ERROR, f"Falha: {formatted.error}"
        )
        with progress_area.container():
            render_pipeline_progress(steps)
        st.error("❌ Falha na formatação da cifra.")
        return

    steps = update_step(
        steps,
        "formatter",
        StepStatus.SUCCESS,
        "Cifra gerada com sucesso!",
        mode=StepMode.CLOUD,
    )
    with progress_area.container():
        render_pipeline_progress(steps)

    # ── Resultado Final ───────────────────────────────────────────────────────
    render_quality_indicator(steps)

    song_name = audio_path.stem
    render_chord_display(formatted, song_name)
    render_chord_stats(formatted)

    logger.info("Pipeline concluído com sucesso para: %s", audio_path.name)


# ── Entrada ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

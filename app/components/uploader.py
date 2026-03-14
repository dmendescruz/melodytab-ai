"""
uploader.py
Componente Streamlit para upload de arquivos de áudio.

Exibe a área de upload, valida o arquivo enviado e retorna
o caminho local para processamento pelo pipeline.
"""

import logging
from pathlib import Path

import streamlit as st

from app.utils.audio import (
    SUPPORTED_FORMATS,
    format_duration,
    get_audio_info,
    save_uploaded_file,
    validate_audio_file,
)

logger = logging.getLogger(__name__)


def render_uploader() -> Path | None:
    """
    Renderiza o componente de upload de áudio no Streamlit.

    Exibe a área de upload, valida o arquivo enviado,
    mostra informações do áudio e retorna o caminho local.

    Returns:
        Caminho para o arquivo salvo localmente, ou None se
        nenhum arquivo foi enviado ou o arquivo é inválido.
    """
    st.markdown("### 🎵 Envie sua música")

    uploaded = st.file_uploader(
        label="Arraste ou selecione um arquivo de áudio",
        type=[fmt.lstrip(".") for fmt in SUPPORTED_FORMATS],
        help=(
            f"Formatos aceitos: {', '.join(sorted(SUPPORTED_FORMATS)).upper()}. "
            f"Tamanho máximo: 50 MB."
        ),
    )

    if uploaded is None:
        _render_upload_hint()
        return None

    # Salva o arquivo temporariamente
    with st.spinner("Carregando arquivo..."):
        audio_path = save_uploaded_file(uploaded)

    # Valida o arquivo
    valido, mensagem = validate_audio_file(audio_path)
    if not valido:
        st.error(f"❌ {mensagem}")
        logger.warning("Arquivo inválido: %s", mensagem)
        return None

    # Exibe informações do áudio
    _render_audio_info(audio_path, uploaded.name)

    return audio_path


def _render_audio_info(path: Path, nome_original: str) -> None:
    """
    Exibe informações técnicas do arquivo de áudio carregado.

    Args:
        path:           Caminho para o arquivo de áudio.
        nome_original:  Nome original do arquivo enviado.
    """
    info = get_audio_info(path)

    st.success(f"✅ **{nome_original}** carregado com sucesso!")
    st.audio(str(path))

    col1, col2, col3 = st.columns(3)

    with col1:
        duracao = format_duration(info["duration"])
        st.metric("⏱️ Duração", duracao)

    with col2:
        st.metric("🎚️ Sample Rate", f"{info['sample_rate']:,} Hz")

    with col3:
        st.metric("💾 Tamanho", f"{info['size_mb']} MB")


def _render_upload_hint() -> None:
    """Exibe dica visual quando nenhum arquivo foi enviado."""
    st.info(
        "👆 Envie um arquivo de áudio para começar. "
        "O processamento pode levar alguns minutos dependendo "
        "da duração da música.",
        icon="ℹ️",
    )

"""
chord_display.py
Componente Streamlit para exibição da cifra gerada.

Exibe a cifra melódica formatada com acordes, melodia e letra
alinhados, além de opções de download.
"""

import logging

import streamlit as st

from app.pipeline.formatter import FormattedChord

logger = logging.getLogger(__name__)


def render_chord_display(result: FormattedChord, song_name: str = "cifra") -> None:
    """
    Renderiza a cifra gerada na interface do Streamlit.

    Exibe a cifra polida pelo LLaMA com opção de alternar
    para a versão bruta, além de botão de download.

    Args:
        result:    Resultado da formatação com cifra bruta e polida.
        song_name: Nome da música para o arquivo de download.
    """
    if not result.success:
        st.error(f"❌ Erro na geração da cifra: {result.error}")
        return

    st.markdown("---")
    st.markdown("## 🎸 Cifra Gerada")

    # Abas para alternar entre versão polida e bruta
    tab_polida, tab_bruta = st.tabs(["✨ Versão Final", "🔧 Versão Bruta"])

    with tab_polida:
        _render_chord_text(result.polished, "polida")

    with tab_bruta:
        st.caption("Versão bruta gerada localmente antes do polimento pelo LLaMA.")
        _render_chord_text(result.raw, "bruta")

    # Botões de download
    st.markdown("### 💾 Download")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Baixar Cifra (.txt)",
            data=result.polished,
            file_name=f"{_sanitize_filename(song_name)}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="⬇️ Baixar Versão Bruta (.txt)",
            data=result.raw,
            file_name=f"{_sanitize_filename(song_name)}_bruta.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_chord_text(cifra: str, key_suffix: str) -> None:
    """
    Renderiza o texto da cifra com formatação monospace.

    Usa fonte monospace para garantir o alinhamento correto
    entre acordes, melodia e letra.

    Args:
        cifra:      Texto da cifra a ser exibido.
        key_suffix: Sufixo para chave única do componente.
    """
    if not cifra.strip():
        st.warning("Cifra vazia.")
        return

    # Exibe em fonte monospace para preservar alinhamento
    st.code(cifra, language=None)

    # Estatísticas da cifra
    linhas = [linha for linha in cifra.split("\n") if linha.strip()]
    acordes = _count_chords(cifra)

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"📄 {len(linhas)} linhas")
    with col2:
        st.caption(f"🎸 {acordes} acordes detectados")


def _count_chords(cifra: str) -> int:
    """
    Conta o número de acordes únicos na cifra.

    Args:
        cifra: Texto da cifra.

    Returns:
        Número de acordes únicos encontrados.
    """
    import re

    acordes = re.findall(r"\[([A-G][#b]?m?7?)\]", cifra)
    return len(set(acordes))


def _sanitize_filename(name: str) -> str:
    """
    Sanitiza um nome de arquivo removendo caracteres inválidos.

    Args:
        name: Nome original do arquivo.

    Returns:
        Nome sanitizado seguro para uso como nome de arquivo.
    """
    import re

    # Remove caracteres inválidos para nomes de arquivo
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    sanitized = sanitized.strip().replace(" ", "_")
    return sanitized or "cifra"


def render_chord_stats(result: FormattedChord) -> None:
    """
    Renderiza um resumo estatístico da cifra gerada.

    Args:
        result: Resultado da formatação da cifra.
    """
    if not result.success:
        return

    import re

    acordes_lista = re.findall(r"\[([A-G][#b]?m?7?)\]", result.polished)
    acordes_unicos = sorted(set(acordes_lista))

    if acordes_unicos:
        st.markdown("### 📊 Acordes Utilizados")
        st.markdown(" • ".join(f"`{a}`" for a in acordes_unicos))

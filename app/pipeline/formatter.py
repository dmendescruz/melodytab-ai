"""
formatter.py
Formatação do resultado do alinhamento no padrão de cifra brasileiro.

Recebe o AlignmentResult do aligner.py e gera a cifra final em dois passos:
1. Monta a cifra bruta localmente a partir das estruturas de dados
2. Usa o Groq LLaMA para polir e padronizar a formatação final
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from app.pipeline.aligner import AlignedLine, AlignedWord, AlignmentResult

load_dotenv()

logger = logging.getLogger(__name__)

# Modelo LLaMA usado para polimento da cifra
LLAMA_MODEL = "llama-3.3-70b-versatile"


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class FormattedChord:
    """
    Resultado da formatação da cifra.

    Atributos:
        raw:      Cifra bruta gerada localmente.
        polished: Cifra polida pelo LLaMA.
        success:  Indica se a formatação foi bem-sucedida.
        error:    Mensagem de erro, se houver.
    """

    raw: str
    polished: str
    success: bool
    error: str | None = None


# ── Função Principal ──────────────────────────────────────────────────────────


def format_chord(alignment: AlignmentResult) -> FormattedChord:
    """
    Formata o resultado do alinhamento em uma cifra melódica.

    Args:
        alignment: Resultado do alinhamento de letra, acordes e melodia.

    Returns:
        FormattedChord com a cifra bruta e a cifra polida pelo LLaMA.
    """
    logger.info("Iniciando formatação da cifra...")

    try:
        # Passo 1: gera a cifra bruta localmente
        logger.info("Gerando cifra bruta...")
        raw = _build_raw_chord(alignment.lines)
        logger.info("Cifra bruta gerada:\n%s", raw)

        # Passo 2: polish com LLaMA
        logger.info("Enviando para Groq LLaMA para polimento...")
        polished = _polish_with_llama(raw)
        logger.info("✅ Cifra polida com sucesso.")

        return FormattedChord(raw=raw, polished=polished, success=True)

    except Exception as e:
        logger.error("❌ Formatação falhou: %s", str(e))
        return FormattedChord(raw="", polished="", success=False, error=str(e))


# ── Geração da Cifra Bruta ────────────────────────────────────────────────────


def _build_raw_chord(lines: list[AlignedLine]) -> str:
    """
    Gera a cifra bruta a partir das linhas alinhadas.

    Formato de saída para cada linha:
        Linha de acordes: [C]        [Am]
        Linha de melodia: mi sol lá   mi ré dó
        Linha de letra:   quando o sol se pôr

    Args:
        lines: Lista de linhas alinhadas do AlignmentResult.

    Returns:
        String com a cifra bruta formatada.
    """
    resultado = []

    for linha in lines:
        chord_line = _build_chord_line(linha.words)
        melody_line = _build_melody_line(linha.words)
        lyric_line = _build_lyric_line(linha.words)

        # Adiciona as 3 linhas se tiverem conteúdo
        if chord_line.strip():
            resultado.append(chord_line)
        if melody_line.strip():
            resultado.append(melody_line)
        resultado.append(lyric_line)
        resultado.append("")  # linha em branco entre estrofes

    return "\n".join(resultado).strip()


def _build_chord_line(words: list[AlignedWord]) -> str:
    """
    Monta a linha de acordes alinhada com as palavras.

    Acordes são inseridos entre colchetes na posição da palavra
    onde ocorre a mudança de acorde.

    Args:
        words: Lista de palavras de uma linha.

    Returns:
        String com os acordes posicionados sobre as palavras.
    """
    chord_line = ""
    lyric_line = ""

    for word in words:
        palavra = word.word + " "

        if word.chord_change and word.chord:
            acorde = f"[{word.chord}]"

            # Alinha o acorde com a posição da palavra na letra
            pos_acorde = len(lyric_line)
            pos_atual = len(chord_line)

            if pos_atual < pos_acorde:
                chord_line += " " * (pos_acorde - pos_atual)

            chord_line += acorde

        lyric_line += palavra

    return chord_line


def _build_melody_line(words: list[AlignedWord]) -> str:
    """
    Monta a linha de melodia com as notas posicionadas sob as palavras.

    Args:
        words: Lista de palavras de uma linha.

    Returns:
        String com as notas posicionadas sob as palavras.
    """
    melody_line = ""
    lyric_line = ""

    for word in words:
        palavra = word.word + " "

        if word.note:
            nota = word.note + " "

            # Alinha a nota com a posição da palavra
            pos_palavra = len(lyric_line)
            pos_atual = len(melody_line)

            if pos_atual < pos_palavra:
                melody_line += " " * (pos_palavra - pos_atual)

            melody_line += nota

        lyric_line += palavra

    return melody_line


def _build_lyric_line(words: list[AlignedWord]) -> str:
    """
    Monta a linha de letra juntando as palavras com espaços.

    Args:
        words: Lista de palavras de uma linha.

    Returns:
        String com a letra da linha.
    """
    return " ".join(w.word for w in words)


# ── Polimento com LLaMA ───────────────────────────────────────────────────────


def _polish_with_llama(raw_chord: str) -> str:
    """
    Usa o Groq LLaMA para polir e padronizar a cifra bruta.

    O LLaMA corrige espaçamentos, padroniza notação de acordes,
    ajusta quebras de linha e garante que a cifra siga o padrão
    brasileiro de forma legível.

    Args:
        raw_chord: Cifra bruta gerada localmente.

    Returns:
        Cifra polida e padronizada pelo LLaMA.

    Raises:
        RuntimeError: Se a chave da API não estiver configurada.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY não configurada. " "Adicione sua chave ao arquivo .env."
        )

    client = Groq(api_key=api_key)

    prompt = f"""Você é um especialista em cifras musicais brasileiras.
Receba a cifra bruta abaixo e faça apenas os seguintes ajustes:

1. Corrija espaçamentos entre acordes e palavras
2. Padronize a notação dos acordes (ex: Bbm em vez de A#m)
3. Ajuste quebras de linha para melhor legibilidade
4. Mantenha as notas da melodia alinhadas com as palavras
5. NÃO altere os acordes detectados, apenas a formatação
6. NÃO adicione acordes que não estejam na cifra bruta
7. Retorne APENAS a cifra formatada, sem explicações

CIFRA BRUTA:
{raw_chord}
"""

    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # baixa temperatura para saída consistente
        max_tokens=2048,
    )

    polished = response.choices[0].message.content.strip()
    logger.info("LLaMA retornou %d caracteres.", len(polished))

    return polished

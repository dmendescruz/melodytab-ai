"""
aligner.py
Sincronização de letra, acordes e melodia pelos timestamps.

Recebe os resultados dos módulos anteriores e alinha tudo em uma
estrutura unificada de sílabas — cada sílaba contém a palavra,
o acorde vigente naquele momento e a nota da melodia correspondente.
"""

import logging
from dataclasses import dataclass, field

from app.pipeline.chord_detector import ChordDetectionResult, ChordEvent
from app.pipeline.melody_detector import MelodyDetectionResult, NoteEvent
from app.pipeline.transcriber import TranscriptionResult, WordEvent

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class AlignedWord:
    """
    Representa uma palavra alinhada com seu acorde e nota melódica.

    Atributos:
        word:        Texto da palavra.
        start_time:  Tempo de início em segundos.
        end_time:    Tempo de fim em segundos.
        chord:       Acorde vigente no momento da palavra (ex: 'Am').
        note:        Nota melódica no momento da palavra (ex: 'E').
        octave:      Oitava da nota melódica.
        chord_change: True se o acorde muda nesta palavra.
    """

    word: str
    start_time: float
    end_time: float
    chord: str | None = None
    note: str | None = None
    octave: int | None = None
    chord_change: bool = False


@dataclass
class AlignedLine:
    """
    Representa uma linha da cifra com suas palavras alinhadas.

    Atributos:
        words:      Lista de palavras alinhadas da linha.
        start_time: Tempo de início da linha em segundos.
        end_time:   Tempo de fim da linha em segundos.
    """

    words: list[AlignedWord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class AlignmentResult:
    """
    Resultado completo do alinhamento.

    Atributos:
        lines:   Lista de linhas alinhadas prontas para formatação.
        success: Indica se o alinhamento foi bem-sucedido.
        error:   Mensagem de erro, se houver.
    """

    lines: list[AlignedLine]
    success: bool
    error: str | None = None


# ── Função Principal ──────────────────────────────────────────────────────────


def align(
    transcription: TranscriptionResult,
    chords: ChordDetectionResult,
    melody: MelodyDetectionResult,
    words_per_line: int = 6,
) -> AlignmentResult:
    """
    Alinha letra, acordes e melodia em uma estrutura unificada.

    Args:
        transcription:  Resultado da transcrição com palavras e timestamps.
        chords:         Resultado da detecção de acordes com timestamps.
        melody:         Resultado da detecção de melodia com timestamps.
        words_per_line: Número aproximado de palavras por linha da cifra.

    Returns:
        AlignmentResult com as linhas da cifra alinhadas.
    """
    logger.info("Iniciando alinhamento de letra, acordes e melodia...")

    try:
        if not transcription.words:
            raise ValueError("Transcrição não contém palavras.")

        # Alinha cada palavra com seu acorde e nota correspondentes
        logger.info("Alinhando %d palavras...", len(transcription.words))
        aligned_words = _align_words(
            transcription.words,
            chords.chords if chords.success else [],
            melody.notes if melody.success else [],
        )

        # Marca as palavras onde o acorde muda
        aligned_words = _mark_chord_changes(aligned_words)

        # Agrupa as palavras em linhas
        logger.info("Agrupando palavras em linhas...")
        lines = _group_into_lines(aligned_words, words_per_line)

        logger.info("✅ Alinhamento concluído: %d linhas geradas.", len(lines))
        return AlignmentResult(lines=lines, success=True)

    except Exception as e:
        logger.error("❌ Alinhamento falhou: %s", str(e))
        return AlignmentResult(lines=[], success=False, error=str(e))


# ── Alinhamento de Palavras ───────────────────────────────────────────────────


def _align_words(
    words: list[WordEvent],
    chords: list[ChordEvent],
    notes: list[NoteEvent],
) -> list[AlignedWord]:
    """
    Para cada palavra, encontra o acorde e a nota vigentes no mesmo timestamp.

    A estratégia é simples: para cada palavra, procuramos o acorde e a nota
    cujo intervalo de tempo contém o início da palavra.

    Args:
        words:  Lista de palavras com timestamps.
        chords: Lista de acordes com timestamps.
        notes:  Lista de notas com timestamps.

    Returns:
        Lista de AlignedWord com acorde e nota preenchidos.
    """
    aligned = []

    for word_event in words:
        t = word_event.start_time

        chord = _find_at_time(chords, t)
        note = _find_at_time(notes, t)

        aligned.append(
            AlignedWord(
                word=word_event.word,
                start_time=word_event.start_time,
                end_time=word_event.end_time,
                chord=chord.chord if chord else None,
                note=note.note if note else None,
                octave=note.octave if note else None,
            )
        )

    logger.info("%d palavras alinhadas com acordes e notas.", len(aligned))
    return aligned


def _find_at_time(events: list, time: float):
    """
    Encontra o evento (acorde ou nota) vigente em um determinado tempo.

    Retorna o primeiro evento cujo intervalo [start_time, end_time]
    contém o tempo buscado. Se nenhum for encontrado, retorna o
    evento mais próximo antes do tempo.

    Args:
        events: Lista de ChordEvent ou NoteEvent.
        time:   Tempo em segundos a buscar.

    Returns:
        O evento vigente ou None se a lista estiver vazia.
    """
    if not events:
        return None

    ultimo_valido = None

    for event in events:
        if event.start_time <= time <= event.end_time:
            return event
        if event.start_time <= time:
            ultimo_valido = event

    return ultimo_valido


# ── Marcação de Mudanças de Acorde ────────────────────────────────────────────


def _mark_chord_changes(words: list[AlignedWord]) -> list[AlignedWord]:
    """
    Marca as palavras onde ocorre uma mudança de acorde.

    A primeira palavra com um acorde é sempre marcada como mudança.
    As demais são marcadas quando o acorde difere do anterior.

    Args:
        words: Lista de palavras alinhadas.

    Returns:
        A mesma lista com o campo chord_change preenchido.
    """
    acorde_anterior = None

    for word in words:
        if word.chord and word.chord != acorde_anterior:
            word.chord_change = True
            acorde_anterior = word.chord

    mudancas = sum(1 for w in words if w.chord_change)
    logger.info("%d mudanças de acorde marcadas.", mudancas)
    return words


# ── Agrupamento em Linhas ─────────────────────────────────────────────────────


def _group_into_lines(
    words: list[AlignedWord],
    words_per_line: int,
) -> list[AlignedLine]:
    """
    Agrupa as palavras em linhas para exibição na cifra.

    Quebra a linha preferencialmente na mudança de acorde mais próxima
    do limite de words_per_line, para que cada linha comece com um acorde.

    Args:
        words:          Lista de palavras alinhadas.
        words_per_line: Número alvo de palavras por linha.

    Returns:
        Lista de AlignedLine com as palavras agrupadas.
    """
    if not words:
        return []

    lines = []
    i = 0
    n = len(words)

    while i < n:
        limite = min(i + words_per_line, n)

        # Procura uma mudança de acorde próxima ao limite para quebrar a linha
        quebra = limite
        for j in range(limite, min(limite + 3, n)):
            if words[j].chord_change:
                quebra = j
                break

        grupo = words[i:quebra]
        if grupo:
            lines.append(
                AlignedLine(
                    words=grupo,
                    start_time=grupo[0].start_time,
                    end_time=grupo[-1].end_time,
                )
            )
        i = quebra

    logger.info("%d linhas geradas com ~%d palavras cada.", len(lines), words_per_line)
    return lines

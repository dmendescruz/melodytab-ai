"""
melody_detector.py
Detecção de melodia usando librosa pyin.

O algoritmo pyin (Probabilistic YIN) analisa a faixa vocal frame a frame
e estima a frequência fundamental (f0) — ou seja, a altura de cada nota
cantada. As frequências são então convertidas para nomes de notas musicais
(C, D, E, F, G, A, B) para compor a linha melódica da cifra.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# ── Constantes ────────────────────────────────────────────────────────────────

# Frequências mínima e máxima para detecção de voz humana
# C2 (65 Hz) a C6 (1047 Hz) cobre sopranos, tenores, contraltos e baixos
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C6")

# Tamanho do frame — controla a granularidade da detecção
# 512 samples a 22050 Hz ≈ 23ms por frame (granularidade fina para melodia)
HOP_LENGTH = 512

# Confiança mínima para considerar uma nota como válida
# Frames abaixo desse limiar são tratados como silêncio
MIN_CONFIDENCE = 0.5

# Duração mínima de uma nota em segundos
# Notas mais curtas que isso são consideradas ruído
MIN_NOTE_DURATION = 0.08


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class NoteEvent:
    """
    Representa uma nota musical detectada na melodia.

    Atributos:
        note:       Nome da nota (ex: 'C', 'D#', 'G').
        octave:     Oitava da nota (ex: 4 para Dó central).
        frequency:  Frequência em Hz.
        start_time: Tempo de início em segundos.
        end_time:   Tempo de fim em segundos.
        confidence: Confiança média da detecção entre 0.0 e 1.0.
    """

    note: str
    octave: int
    frequency: float
    start_time: float
    end_time: float
    confidence: float


@dataclass
class MelodyDetectionResult:
    """
    Resultado completo da detecção de melodia.

    Atributos:
        notes:   Lista de notas detectadas com seus timestamps.
        success: Indica se a detecção foi bem-sucedida.
        error:   Mensagem de erro, se houver.
    """

    notes: list[NoteEvent]
    success: bool
    error: str | None = None


# ── Função Principal ──────────────────────────────────────────────────────────


def detect_melody(vocals_path: Path) -> MelodyDetectionResult:
    """
    Detecta a linha melódica em uma faixa vocal isolada.

    Args:
        vocals_path: Caminho para a faixa vocal isolada pelo Demucs.

    Returns:
        MelodyDetectionResult com a lista de notas e seus timestamps.
    """
    logger.info("Iniciando detecção de melodia: %s", vocals_path)

    try:
        # Carrega o áudio
        logger.info("Carregando áudio vocal...")
        audio, sr = librosa.load(str(vocals_path), sr=None, mono=True)
        logger.info("Áudio carregado: %.1f segundos | %d Hz", len(audio) / sr, sr)

        # Detecta frequência fundamental com pyin
        logger.info("Executando algoritmo pyin...")
        f0, voiced_flag, voiced_probs = _run_pyin(audio, sr)

        # Converte frequências para notas musicais
        logger.info("Convertendo frequências para notas...")
        raw_notes = _frequencies_to_notes(f0, voiced_flag, voiced_probs, sr)

        # Agrupa notas consecutivas iguais
        logger.info("Agrupando notas consecutivas...")
        grouped = _group_consecutive_notes(raw_notes)

        logger.info("✅ %d notas detectadas na melodia.", len(grouped))
        return MelodyDetectionResult(notes=grouped, success=True)

    except Exception as e:
        logger.error("❌ Detecção de melodia falhou: %s", str(e))
        return MelodyDetectionResult(notes=[], success=False, error=str(e))


# ── Detecção de Pitch (pyin) ──────────────────────────────────────────────────


def _run_pyin(
    audio: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa o algoritmo pyin para detecção de pitch.

    pyin é uma versão probabilística do algoritmo YIN que estima a
    frequência fundamental (f0) e a probabilidade de cada frame
    ser voiced (cantado) ou unvoiced (silêncio/ruído).

    Args:
        audio: Array numpy com o sinal de áudio.
        sr:    Taxa de amostragem do áudio.

    Returns:
        Tupla (f0, voiced_flag, voiced_probs):
            f0:          Frequência fundamental por frame (Hz), NaN se unvoiced.
            voiced_flag: Booleano por frame indicando se há voz.
            voiced_probs: Probabilidade de voz por frame (0.0 a 1.0).
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    n_voiced = int(np.sum(voiced_flag))
    logger.info(
        "pyin concluído: %d frames totais | %d voiced (%.1f%%)",
        len(f0),
        n_voiced,
        100 * n_voiced / max(len(f0), 1),
    )

    return f0, voiced_flag, voiced_probs


# ── Conversão de Frequências para Notas ──────────────────────────────────────


def _frequencies_to_notes(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    sr: int,
) -> list[NoteEvent]:
    """
    Converte as frequências detectadas pelo pyin em eventos de nota musical.

    Apenas frames voiced com confiança acima de MIN_CONFIDENCE são
    convertidos. Frames unvoiced ou com baixa confiança são ignorados.

    Args:
        f0:           Frequências fundamentais por frame.
        voiced_flag:  Flags de voiced/unvoiced por frame.
        voiced_probs: Probabilidades de voz por frame.
        sr:           Taxa de amostragem do áudio.

    Returns:
        Lista de NoteEvent, um por frame voiced válido.
    """
    frame_dur = HOP_LENGTH / sr
    notes = []

    for i, (freq, voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_probs)):
        # Ignora frames sem voz ou com baixa confiança
        if not voiced or prob < MIN_CONFIDENCE or np.isnan(freq):
            continue

        # Converte frequência para nome de nota e oitava
        note_name, octave = _hz_to_note(freq)

        notes.append(
            NoteEvent(
                note=note_name,
                octave=octave,
                frequency=round(float(freq), 2),
                start_time=round(i * frame_dur, 3),
                end_time=round((i + 1) * frame_dur, 3),
                confidence=round(float(prob), 4),
            )
        )

    logger.info("%d frames convertidos para notas.", len(notes))
    return notes


def _hz_to_note(frequency: float) -> tuple[str, int]:
    """
    Converte uma frequência em Hz para nome de nota e oitava.

    Usa a escala temperada de 12 semitons com A4 = 440 Hz como referência.

    Args:
        frequency: Frequência em Hz.

    Returns:
        Tupla (nome_da_nota, oitava). Ex: ('A', 4), ('C#', 5).
    """
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Número de semitons em relação ao A4 (440 Hz)
    semitones_from_a4 = 12 * np.log2(frequency / 440.0)

    # Número MIDI (A4 = 69)
    midi = int(round(69 + semitones_from_a4))

    note_name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1

    return note_name, octave


# ── Agrupamento de Notas ──────────────────────────────────────────────────────


def _group_consecutive_notes(
    notes: list[NoteEvent],
    min_duration: float = MIN_NOTE_DURATION,
) -> list[NoteEvent]:
    """
    Agrupa notas consecutivas iguais em um único evento.
    Remove notas com duração menor que min_duration.

    Args:
        notes:        Lista de notas brutas, uma por frame voiced.
        min_duration: Duração mínima em segundos para manter uma nota.

    Returns:
        Lista de notas agrupadas e filtradas.
    """
    if not notes:
        return []

    grouped = []
    atual = notes[0]
    confiancas = [atual.confidence]

    for proxima in notes[1:]:
        if proxima.note == atual.note and proxima.octave == atual.octave:
            # Mesma nota — estende o evento atual
            atual = NoteEvent(
                note=atual.note,
                octave=atual.octave,
                frequency=atual.frequency,
                start_time=atual.start_time,
                end_time=proxima.end_time,
                confidence=atual.confidence,
            )
            confiancas.append(proxima.confidence)
        else:
            # Nota diferente — finaliza a atual e começa nova
            duracao = atual.end_time - atual.start_time
            if duracao >= min_duration:
                grouped.append(
                    NoteEvent(
                        note=atual.note,
                        octave=atual.octave,
                        frequency=atual.frequency,
                        start_time=atual.start_time,
                        end_time=atual.end_time,
                        confidence=round(float(np.mean(confiancas)), 4),
                    )
                )
            atual = proxima
            confiancas = [proxima.confidence]

    # Adiciona a última nota
    duracao = atual.end_time - atual.start_time
    if duracao >= min_duration:
        grouped.append(
            NoteEvent(
                note=atual.note,
                octave=atual.octave,
                frequency=atual.frequency,
                start_time=atual.start_time,
                end_time=atual.end_time,
                confidence=round(float(np.mean(confiancas)), 4),
            )
        )

    logger.info(
        "Notas após agrupamento: %d (removidas por duração curta: %d)",
        len(grouped),
        len(notes) - len(grouped),
    )

    return grouped

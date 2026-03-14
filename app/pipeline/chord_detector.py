"""
chord_detector.py
Detecção de acordes usando librosa chroma/cqt.

O áudio da faixa harmônica é analisado frame a frame. Para cada frame,
extraímos o vetor de chroma (energia distribuída pelas 12 notas musicais)
e comparamos com templates de acordes conhecidos para identificar o
acorde mais provável em cada momento.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# ── Constantes ────────────────────────────────────────────────────────────────

# Tamanho do frame em samples — controla a granularidade da detecção
# 4096 samples a 22050 Hz ≈ 185ms por frame
HOP_LENGTH = 4096

# Frequências de referência para análise CQT
FMIN = librosa.note_to_hz("C2")  # nota mais grave considerada
N_BINS = 84  # 7 oitavas de C2 a B8

# Templates de acordes — vetor binário com as notas de cada acorde
# Índices: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
CHORD_TEMPLATES = {
    # Acordes maiores
    "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "C#": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "D": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "D#": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "E": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    "F": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "F#": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    "G": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "G#": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "A": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "A#": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "B": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    # Acordes menores
    "Cm": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "C#m": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "Dm": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "D#m": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "Em": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "Fm": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "F#m": [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "Gm": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    "G#m": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    "Am": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "A#m": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "Bm": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    # Acordes com sétima
    "G7": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    "C7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "D7": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    "E7": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    "A7": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    "B7": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ChordEvent:
    """
    Representa um acorde detectado em um momento específico do áudio.

    Atributos:
        chord:      Nome do acorde (ex: 'Am', 'G7').
        start_time: Tempo de início em segundos.
        end_time:   Tempo de fim em segundos.
        confidence: Confiança da detecção entre 0.0 e 1.0.
    """

    chord: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class ChordDetectionResult:
    """
    Resultado completo da detecção de acordes.

    Atributos:
        chords:  Lista de acordes detectados com seus timestamps.
        success: Indica se a detecção foi bem-sucedida.
        error:   Mensagem de erro, se houver.
    """

    chords: list[ChordEvent]
    success: bool
    error: str | None = None


# ── Função Principal ──────────────────────────────────────────────────────────


def detect_chords(harmonic_path: Path) -> ChordDetectionResult:
    """
    Detecta acordes em uma faixa de áudio harmônica.

    Args:
        harmonic_path: Caminho para a faixa harmônica isolada pelo Demucs.

    Returns:
        ChordDetectionResult com a lista de acordes e seus timestamps.
    """
    logger.info("Iniciando detecção de acordes: %s", harmonic_path)

    try:
        # Carrega o áudio
        logger.info("Carregando áudio...")
        audio, sr = librosa.load(str(harmonic_path), sr=None, mono=True)
        logger.info("Áudio carregado: %.1f segundos | %d Hz", len(audio) / sr, sr)

        # Extrai features de chroma
        logger.info("Extraindo features de chroma...")
        chroma = _extract_chroma(audio, sr)

        # Detecta acordes frame a frame
        logger.info("Detectando acordes frame a frame...")
        raw_chords = _detect_chords_from_chroma(chroma, sr)

        # Agrupa acordes consecutivos iguais
        logger.info("Agrupando acordes consecutivos...")
        grouped = _group_consecutive_chords(raw_chords)

        logger.info("✅ %d acordes detectados.", len(grouped))
        return ChordDetectionResult(chords=grouped, success=True)

    except Exception as e:
        logger.error("❌ Detecção de acordes falhou: %s", str(e))
        return ChordDetectionResult(chords=[], success=False, error=str(e))


# ── Extração de Chroma ────────────────────────────────────────────────────────


def _extract_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extrai o vetor de chroma do áudio usando CQT.

    O chroma representa a energia distribuída pelas 12 notas musicais
    (C, C#, D, ..., B) ao longo do tempo, independente da oitava.

    Args:
        audio: Array numpy com o sinal de áudio.
        sr:    Taxa de amostragem do áudio.

    Returns:
        Array de shape (12, n_frames) com os valores de chroma.
    """
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_chroma=12,
    )

    # Normaliza cada frame para que a soma seja 1
    chroma_norm = librosa.util.normalize(chroma, axis=0)
    logger.info("Chroma extraído: %d frames", chroma_norm.shape[1])

    return chroma_norm


# ── Detecção de Acordes ───────────────────────────────────────────────────────


def _detect_chords_from_chroma(
    chroma: np.ndarray,
    sr: int,
) -> list[ChordEvent]:
    """
    Compara cada frame de chroma com os templates de acordes conhecidos
    e identifica o acorde mais provável em cada momento.

    Args:
        chroma: Array de chroma normalizado (12, n_frames).
        sr:     Taxa de amostragem do áudio.

    Returns:
        Lista de ChordEvent, um por frame.
    """
    templates = np.array(list(CHORD_TEMPLATES.values()), dtype=float)
    nomes = list(CHORD_TEMPLATES.keys())

    # Normaliza os templates
    normas = np.linalg.norm(templates, axis=1, keepdims=True)
    templates_norm = templates / normas

    n_frames = chroma.shape[1]
    frame_dur = HOP_LENGTH / sr
    chord_list = []

    for i in range(n_frames):
        frame = chroma[:, i]

        # Normaliza o frame antes de calcular similaridade de cosseno
        norma_frame = np.linalg.norm(frame)
        if norma_frame > 0:
            frame = frame / norma_frame

        # Similaridade de cosseno entre o frame e cada template
        similaridades = templates_norm @ frame

        melhor_idx = int(np.argmax(similaridades))
        melhor_acorde = nomes[melhor_idx]
        confianca = float(similaridades[melhor_idx])

        chord_list.append(
            ChordEvent(
                chord=melhor_acorde,
                start_time=round(i * frame_dur, 3),
                end_time=round((i + 1) * frame_dur, 3),
                confidence=round(confianca, 4),
            )
        )

    return chord_list


# ── Agrupamento de Acordes ────────────────────────────────────────────────────


def _group_consecutive_chords(
    chords: list[ChordEvent],
    min_duration: float = 0.5,
) -> list[ChordEvent]:
    """
    Agrupa acordes consecutivos iguais em um único evento.
    Remove acordes com duração menor que min_duration (ruído de detecção).

    Args:
        chords:       Lista de acordes brutos, um por frame.
        min_duration: Duração mínima em segundos para manter um acorde.

    Returns:
        Lista de acordes agrupados e filtrados.
    """
    if not chords:
        return []

    grouped = []
    atual = chords[0]
    confiancas = [atual.confidence]

    for proximo in chords[1:]:
        if proximo.chord == atual.chord:
            # Mesmo acorde — estende o evento atual
            atual = ChordEvent(
                chord=atual.chord,
                start_time=atual.start_time,
                end_time=proximo.end_time,
                confidence=atual.confidence,
            )
            confiancas.append(proximo.confidence)
        else:
            # Acorde diferente — finaliza o atual e começa novo
            duracao = atual.end_time - atual.start_time
            if duracao >= min_duration:
                grouped.append(
                    ChordEvent(
                        chord=atual.chord,
                        start_time=atual.start_time,
                        end_time=atual.end_time,
                        confidence=round(float(np.mean(confiancas)), 4),
                    )
                )
            atual = proximo
            confiancas = [proximo.confidence]

    # Adiciona o último acorde
    duracao = atual.end_time - atual.start_time
    if duracao >= min_duration:
        grouped.append(
            ChordEvent(
                chord=atual.chord,
                start_time=atual.start_time,
                end_time=atual.end_time,
                confidence=round(float(np.mean(confiancas)), 4),
            )
        )

    logger.info(
        "Acordes após agrupamento: %d (removidos por duração curta: %d)",
        len(grouped),
        len(chords) - len(grouped),
    )

    return grouped

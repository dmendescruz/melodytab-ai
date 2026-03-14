"""
test_aligner.py
Testes unitários para o módulo aligner.py
"""

import pytest

from app.pipeline.aligner import (
    AlignedLine,
    AlignedWord,
    _align_words,
    _find_at_time,
    _group_into_lines,
    _mark_chord_changes,
    align,
)
from app.pipeline.chord_detector import ChordDetectionResult, ChordEvent
from app.pipeline.melody_detector import MelodyDetectionResult, NoteEvent
from app.pipeline.transcriber import TranscriptionResult, WordEvent


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def words():
    """Lista de palavras com timestamps."""
    return [
        WordEvent(word="quando", start_time=0.0, end_time=0.5, confidence=1.0),
        WordEvent(word="o", start_time=0.5, end_time=0.7, confidence=1.0),
        WordEvent(word="sol", start_time=0.7, end_time=1.0, confidence=1.0),
        WordEvent(word="se", start_time=1.0, end_time=1.2, confidence=1.0),
        WordEvent(word="pôr", start_time=1.2, end_time=1.8, confidence=1.0),
        WordEvent(word="estarei", start_time=2.0, end_time=2.6, confidence=1.0),
        WordEvent(word="aqui", start_time=2.6, end_time=3.0, confidence=1.0),
    ]


@pytest.fixture
def chords():
    """Lista de acordes com timestamps."""
    return [
        ChordEvent(chord="C", start_time=0.0, end_time=1.5, confidence=0.9),
        ChordEvent(chord="Am", start_time=1.5, end_time=3.0, confidence=0.85),
    ]


@pytest.fixture
def notes():
    """Lista de notas com timestamps."""
    return [
        NoteEvent(
            note="E",
            octave=4,
            frequency=329.63,
            start_time=0.0,
            end_time=0.5,
            confidence=0.9,
        ),
        NoteEvent(
            note="G",
            octave=4,
            frequency=392.00,
            start_time=0.5,
            end_time=1.0,
            confidence=0.88,
        ),
        NoteEvent(
            note="A",
            octave=4,
            frequency=440.00,
            start_time=1.0,
            end_time=1.8,
            confidence=0.92,
        ),
        NoteEvent(
            note="C",
            octave=5,
            frequency=523.25,
            start_time=2.0,
            end_time=3.0,
            confidence=0.87,
        ),
    ]


@pytest.fixture
def transcription(words):
    """TranscriptionResult com palavras válidas."""
    return TranscriptionResult(
        words=words,
        full_text="quando o sol se pôr estarei aqui",
        language="pt",
        success=True,
    )


@pytest.fixture
def chord_result(chords):
    """ChordDetectionResult com acordes válidos."""
    return ChordDetectionResult(chords=chords, success=True)


@pytest.fixture
def melody_result(notes):
    """MelodyDetectionResult com notas válidas."""
    return MelodyDetectionResult(notes=notes, success=True)


# ── Testes: _find_at_time ─────────────────────────────────────────────────────


class TestFindAtTime:

    def test_encontra_evento_no_intervalo(self, chords):
        """Deve retornar o acorde cujo intervalo contém o tempo."""
        resultado = _find_at_time(chords, 0.5)
        assert resultado.chord == "C"

    def test_encontra_segundo_acorde(self, chords):
        """Deve retornar Am para tempo dentro do seu intervalo."""
        resultado = _find_at_time(chords, 2.0)
        assert resultado.chord == "Am"

    def test_retorna_ultimo_valido_antes_do_tempo(self, chords):
        """Deve retornar o último acorde antes do tempo quando não há match exato."""
        resultado = _find_at_time(chords, 3.5)
        assert resultado.chord == "Am"

    def test_retorna_none_para_lista_vazia(self):
        """Deve retornar None para lista vazia."""
        assert _find_at_time([], 1.0) is None

    def test_encontra_evento_no_inicio_exato(self, chords):
        """Deve retornar o acorde quando o tempo é exatamente o start_time."""
        resultado = _find_at_time(chords, 0.0)
        assert resultado.chord == "C"


# ── Testes: _align_words ──────────────────────────────────────────────────────


class TestAlignWords:

    def test_alinha_palavras_com_acordes(self, words, chords, notes):
        """Palavras antes de 1.5s devem receber acorde C."""
        resultado = _align_words(words, chords, notes)

        assert resultado[0].chord == "C"  # "quando" em 0.0s
        assert resultado[0].word == "quando"

    def test_alinha_palavras_com_novo_acorde(self, words, chords, notes):
        """Palavras após 1.5s devem receber acorde Am."""
        resultado = _align_words(words, chords, notes)

        estarei = next(w for w in resultado if w.word == "estarei")
        assert estarei.chord == "Am"

    def test_alinha_palavras_com_notas(self, words, chords, notes):
        """Primeira palavra deve receber nota E4."""
        resultado = _align_words(words, chords, notes)

        assert resultado[0].note == "E"
        assert resultado[0].octave == 4

    def test_palavra_sem_nota_recebe_none(self, words, chords):
        """Palavra sem nota correspondente deve ter note=None."""
        resultado = _align_words(words, chords, [])

        assert all(w.note is None for w in resultado)

    def test_palavra_sem_acorde_recebe_none(self, words, notes):
        """Palavra sem acorde correspondente deve ter chord=None."""
        resultado = _align_words(words, [], notes)

        assert all(w.chord is None for w in resultado)


# ── Testes: _mark_chord_changes ───────────────────────────────────────────────


class TestMarkChordChanges:

    def test_primeira_palavra_com_acorde_e_marcada(self):
        """Primeira palavra com acorde deve ter chord_change=True."""
        words = [
            AlignedWord("quando", 0.0, 0.5, chord="C"),
            AlignedWord("o", 0.5, 0.7, chord="C"),
        ]
        resultado = _mark_chord_changes(words)

        assert resultado[0].chord_change is True
        assert resultado[1].chord_change is False

    def test_marca_mudanca_de_acorde(self):
        """Palavra onde acorde muda deve ter chord_change=True."""
        words = [
            AlignedWord("quando", 0.0, 0.5, chord="C"),
            AlignedWord("o", 0.5, 0.7, chord="C"),
            AlignedWord("sol", 0.7, 1.0, chord="Am"),
        ]
        resultado = _mark_chord_changes(words)

        assert resultado[2].chord_change is True

    def test_palavra_sem_acorde_nao_marcada(self):
        """Palavra sem acorde não deve ter chord_change=True."""
        words = [
            AlignedWord("quando", 0.0, 0.5, chord=None),
        ]
        resultado = _mark_chord_changes(words)

        assert resultado[0].chord_change is False


# ── Testes: _group_into_lines ─────────────────────────────────────────────────


class TestGroupIntoLines:

    def test_agrupa_palavras_em_linhas(self):
        """Deve agrupar palavras em linhas com o número alvo."""
        words = [AlignedWord(f"palavra{i}", float(i), float(i + 1)) for i in range(12)]
        linhas = _group_into_lines(words, words_per_line=4)

        assert len(linhas) >= 2
        assert all(isinstance(linha, AlignedLine) for linha in linhas)

    def test_lista_vazia_retorna_lista_vazia(self):
        """Lista vazia deve retornar lista vazia."""
        assert _group_into_lines([], words_per_line=6) == []

    def test_timestamps_da_linha_corretos(self):
        """Timestamps da linha devem refletir primeira e última palavra."""
        words = [
            AlignedWord("a", 0.0, 0.5),
            AlignedWord("b", 0.5, 1.0),
            AlignedWord("c", 1.0, 1.5),
        ]
        linhas = _group_into_lines(words, words_per_line=3)

        assert linhas[0].start_time == pytest.approx(0.0)
        assert linhas[0].end_time == pytest.approx(1.5)


# ── Testes: align (fluxo principal) ──────────────────────────────────────────


class TestAlign:

    def test_fluxo_completo_com_sucesso(
        self, transcription, chord_result, melody_result
    ):
        """Deve retornar AlignmentResult com sucesso e linhas preenchidas."""
        resultado = align(transcription, chord_result, melody_result)

        assert resultado.success is True
        assert len(resultado.lines) > 0
        assert all(isinstance(linha, AlignedLine) for linha in resultado.lines)

    def test_palavras_tem_acordes_e_notas(
        self, transcription, chord_result, melody_result
    ):
        """Palavras alinhadas devem ter acordes e notas preenchidos."""
        resultado = align(transcription, chord_result, melody_result)

        todas_palavras = [w for linha in resultado.lines for w in linha.words]
        com_acorde = [w for w in todas_palavras if w.chord]
        assert len(com_acorde) > 0

    def test_falha_com_transcricao_vazia(self, chord_result, melody_result):
        """Deve retornar AlignmentResult com success=False sem palavras."""
        transcricao_vazia = TranscriptionResult(
            words=[],
            full_text="",
            language="pt",
            success=True,
        )
        resultado = align(transcricao_vazia, chord_result, melody_result)

        assert resultado.success is False
        assert resultado.error is not None

    def test_funciona_sem_acordes(self, transcription, melody_result):
        """Deve funcionar mesmo quando detecção de acordes falhou."""
        chords_falhou = ChordDetectionResult(chords=[], success=False)
        resultado = align(transcription, chords_falhou, melody_result)

        assert resultado.success is True

    def test_funciona_sem_melodia(self, transcription, chord_result):
        """Deve funcionar mesmo quando detecção de melodia falhou."""
        melody_falhou = MelodyDetectionResult(notes=[], success=False)
        resultado = align(transcription, chord_result, melody_falhou)

        assert resultado.success is True

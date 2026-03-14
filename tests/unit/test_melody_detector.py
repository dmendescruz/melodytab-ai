"""
test_melody_detector.py
Testes unitários para o módulo melody_detector.py
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.pipeline.melody_detector import (
    MIN_CONFIDENCE,
    MIN_NOTE_DURATION,
    NoteEvent,
    _frequencies_to_notes,
    _group_consecutive_notes,
    _hz_to_note,
    detect_melody,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vocals_path(tmp_path):
    """Cria um arquivo de áudio vocal falso para os testes."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake audio content")
    return audio


@pytest.fixture
def note_sequence():
    """
    Retorna uma sequência de NoteEvents para testes de agrupamento.
    3 frames de C4, 2 frames de E4, 1 frame de G4 (muito curto).
    """
    frame_dur = 0.1
    return [
        NoteEvent("C", 4, 261.63, 0.0, frame_dur, 0.9),
        NoteEvent("C", 4, 261.63, frame_dur, frame_dur * 2, 0.85),
        NoteEvent("C", 4, 261.63, frame_dur * 2, frame_dur * 3, 0.88),
        NoteEvent("E", 4, 329.63, frame_dur * 3, frame_dur * 4, 0.92),
        NoteEvent("E", 4, 329.63, frame_dur * 4, frame_dur * 5, 0.87),
        NoteEvent("G", 4, 392.00, frame_dur * 5, frame_dur * 5 + 0.01, 0.75),
    ]


# ── Testes: _hz_to_note ───────────────────────────────────────────────────────


class TestHzToNote:

    def test_a4_440hz(self):
        """A4 (440 Hz) deve retornar ('A', 4)."""
        note, octave = _hz_to_note(440.0)
        assert note == "A"
        assert octave == 4

    def test_c4_do_central(self):
        """C4 (261.63 Hz) deve retornar ('C', 4)."""
        note, octave = _hz_to_note(261.63)
        assert note == "C"
        assert octave == 4

    def test_c5_oitava_acima(self):
        """C5 (523.25 Hz) deve retornar ('C', 5)."""
        note, octave = _hz_to_note(523.25)
        assert note == "C"
        assert octave == 5

    def test_g4(self):
        """G4 (392.0 Hz) deve retornar ('G', 4)."""
        note, octave = _hz_to_note(392.0)
        assert note == "G"
        assert octave == 4

    def test_nota_sustenido(self):
        """F#4 (369.99 Hz) deve retornar ('F#', 4)."""
        note, octave = _hz_to_note(369.99)
        assert note == "F#"
        assert octave == 4

    def test_nota_baixa_c2(self):
        """C2 (65.41 Hz) deve retornar ('C', 2)."""
        note, octave = _hz_to_note(65.41)
        assert note == "C"
        assert octave == 2


# ── Testes: _frequencies_to_notes ────────────────────────────────────────────


class TestFrequenciesToNotes:

    def test_converte_frame_voiced_valido(self):
        """Frame voiced com confiança alta deve gerar um NoteEvent."""
        f0 = np.array([440.0])
        voiced_flag = np.array([True])
        voiced_probs = np.array([0.9])

        resultado = _frequencies_to_notes(f0, voiced_flag, voiced_probs, sr=22050)

        assert len(resultado) == 1
        assert resultado[0].note == "A"
        assert resultado[0].octave == 4

    def test_ignora_frame_unvoiced(self):
        """Frame unvoiced deve ser ignorado."""
        f0 = np.array([440.0])
        voiced_flag = np.array([False])
        voiced_probs = np.array([0.9])

        resultado = _frequencies_to_notes(f0, voiced_flag, voiced_probs, sr=22050)

        assert len(resultado) == 0

    def test_ignora_frame_com_baixa_confianca(self):
        """Frame com confiança abaixo do mínimo deve ser ignorado."""
        f0 = np.array([440.0])
        voiced_flag = np.array([True])
        voiced_probs = np.array([MIN_CONFIDENCE - 0.1])

        resultado = _frequencies_to_notes(f0, voiced_flag, voiced_probs, sr=22050)

        assert len(resultado) == 0

    def test_ignora_frame_com_nan(self):
        """Frame com frequência NaN deve ser ignorado."""
        f0 = np.array([np.nan])
        voiced_flag = np.array([True])
        voiced_probs = np.array([0.9])

        resultado = _frequencies_to_notes(f0, voiced_flag, voiced_probs, sr=22050)

        assert len(resultado) == 0

    def test_timestamps_corretos(self):
        """Timestamps devem refletir o índice do frame e hop_length."""
        f0 = np.array([440.0, 440.0])
        voiced_flag = np.array([True, True])
        voiced_probs = np.array([0.9, 0.9])

        resultado = _frequencies_to_notes(f0, voiced_flag, voiced_probs, sr=22050)

        assert len(resultado) == 2
        assert resultado[0].start_time == pytest.approx(0.0, abs=0.001)
        assert resultado[1].start_time == pytest.approx(512 / 22050, abs=0.001)


# ── Testes: _group_consecutive_notes ─────────────────────────────────────────


class TestGroupConsecutiveNotes:

    def test_agrupa_notas_consecutivas_iguais(self, note_sequence):
        """Frames consecutivos da mesma nota devem virar um único NoteEvent."""
        resultado = _group_consecutive_notes(note_sequence, min_duration=0.0)

        notas = [n.note for n in resultado]
        assert "C" in notas
        assert "E" in notas

        c = next(n for n in resultado if n.note == "C")
        assert c.start_time == pytest.approx(0.0, abs=0.001)
        assert c.end_time == pytest.approx(0.3, abs=0.001)

    def test_remove_notas_muito_curtas(self, note_sequence):
        """Nota G com duração 0.01s deve ser removida com min_duration=0.08."""
        resultado = _group_consecutive_notes(
            note_sequence, min_duration=MIN_NOTE_DURATION
        )

        notas = [n.note for n in resultado]
        assert "G" not in notas

    def test_lista_vazia_retorna_lista_vazia(self):
        """Lista vazia deve retornar lista vazia."""
        assert _group_consecutive_notes([]) == []

    def test_confianca_e_media_dos_frames(self):
        """Confiança da nota agrupada deve ser a média dos frames."""
        notes = [
            NoteEvent("C", 4, 261.63, 0.0, 0.1, 0.8),
            NoteEvent("C", 4, 261.63, 0.1, 0.2, 0.6),
        ]
        resultado = _group_consecutive_notes(notes, min_duration=0.0)

        assert len(resultado) == 1
        assert resultado[0].confidence == pytest.approx(0.7, abs=0.01)

    def test_notas_mesma_letra_oitavas_diferentes_nao_agrupam(self):
        """C4 e C5 são notas diferentes e não devem ser agrupadas."""
        notes = [
            NoteEvent("C", 4, 261.63, 0.0, 0.1, 0.9),
            NoteEvent("C", 5, 523.25, 0.1, 0.2, 0.9),
        ]
        resultado = _group_consecutive_notes(notes, min_duration=0.0)

        assert len(resultado) == 2


# ── Testes: detect_melody (fluxo principal) ───────────────────────────────────


class TestDetectMelody:

    @patch("app.pipeline.melody_detector.librosa.load")
    @patch("app.pipeline.melody_detector._run_pyin")
    @patch("app.pipeline.melody_detector._frequencies_to_notes")
    @patch("app.pipeline.melody_detector._group_consecutive_notes")
    def test_fluxo_completo_com_sucesso(
        self,
        mock_group,
        mock_freq,
        mock_pyin,
        mock_load,
        vocals_path,
    ):
        """Deve chamar todas as etapas em ordem e retornar sucesso."""
        mock_load.return_value = (np.zeros(22050), 22050)
        mock_pyin.return_value = (
            np.array([440.0]),
            np.array([True]),
            np.array([0.9]),
        )
        mock_freq.return_value = []
        mock_group.return_value = [NoteEvent("A", 4, 440.0, 0.0, 0.5, 0.9)]

        resultado = detect_melody(vocals_path)

        assert resultado.success is True
        assert len(resultado.notes) == 1
        assert resultado.notes[0].note == "A"
        mock_load.assert_called_once()
        mock_pyin.assert_called_once()
        mock_freq.assert_called_once()
        mock_group.assert_called_once()

    @patch("app.pipeline.melody_detector.librosa.load")
    def test_retorna_erro_quando_falha(self, mock_load, vocals_path):
        """Deve retornar MelodyDetectionResult com success=False em caso de erro."""
        mock_load.side_effect = RuntimeError("Arquivo corrompido")

        resultado = detect_melody(vocals_path)

        assert resultado.success is False
        assert resultado.error is not None
        assert "Arquivo corrompido" in resultado.error
        assert resultado.notes == []

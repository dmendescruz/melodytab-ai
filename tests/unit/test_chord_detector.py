"""
test_chord_detector.py
Testes unitários para o módulo chord_detector.py
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.pipeline.chord_detector import (
    CHORD_TEMPLATES,
    ChordEvent,
    _detect_chords_from_chroma,
    _group_consecutive_chords,
    detect_chords,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def harmonic_path(tmp_path):
    """Cria um arquivo de áudio falso para os testes."""
    audio = tmp_path / "harmonic.wav"
    audio.write_bytes(b"fake audio content")
    return audio


@pytest.fixture
def chroma_c_major():
    """
    Retorna um frame de chroma que representa claramente um acorde de Dó maior.
    Notas C, E, G têm energia máxima (índices 0, 4, 7).
    """
    chroma = np.zeros(12)
    chroma[0] = 1.0  # C
    chroma[4] = 1.0  # E
    chroma[7] = 1.0  # G
    return chroma


@pytest.fixture
def chroma_a_minor():
    """
    Retorna um frame de chroma que representa claramente um acorde de Lá menor.
    Notas A, C, E têm energia máxima (índices 9, 0, 4).
    """
    chroma = np.zeros(12)
    chroma[9] = 1.0  # A
    chroma[0] = 1.0  # C
    chroma[4] = 1.0  # E
    return chroma


@pytest.fixture
def chroma_sequence_4_frames():
    """
    Retorna uma sequência de 4 frames de chroma:
    2 frames de C maior seguidos de 2 frames de Am.
    Shape: (12, 4)
    """
    c = np.zeros(12)
    c[0], c[4], c[7] = 1.0, 1.0, 1.0  # C maior

    am = np.zeros(12)
    am[9], am[0], am[4] = 1.0, 1.0, 1.0  # A menor

    return np.column_stack([c, c, am, am])


# ── Testes: CHORD_TEMPLATES ───────────────────────────────────────────────────


class TestChordTemplates:

    def test_todos_os_templates_tem_12_notas(self):
        """Cada template deve ter exatamente 12 notas."""
        for nome, template in CHORD_TEMPLATES.items():
            assert (
                len(template) == 12
            ), f"Template '{nome}' tem {len(template)} notas, esperado 12"

    def test_templates_contem_apenas_zeros_e_uns(self):
        """Templates devem conter apenas valores 0 ou 1."""
        for nome, template in CHORD_TEMPLATES.items():
            for valor in template:
                assert valor in (
                    0,
                    1,
                ), f"Template '{nome}' contém valor inválido: {valor}"

    def test_acordes_maiores_presentes(self):
        """Todos os 12 acordes maiores devem estar presentes."""
        maiores = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for acorde in maiores:
            assert acorde in CHORD_TEMPLATES, f"Acorde '{acorde}' ausente"

    def test_acordes_menores_presentes(self):
        """Todos os 12 acordes menores devem estar presentes."""
        menores = [
            "Cm",
            "C#m",
            "Dm",
            "D#m",
            "Em",
            "Fm",
            "F#m",
            "Gm",
            "G#m",
            "Am",
            "A#m",
            "Bm",
        ]
        for acorde in menores:
            assert acorde in CHORD_TEMPLATES, f"Acorde '{acorde}' ausente"

    def test_c_maior_tem_notas_corretas(self):
        """C maior deve ter notas C(0), E(4), G(7) ativas."""
        template = CHORD_TEMPLATES["C"]
        assert template[0] == 1  # C
        assert template[4] == 1  # E
        assert template[7] == 1  # G
        assert template[1] == 0  # C# inativo
        assert template[3] == 0  # D# inativo

    def test_am_tem_notas_corretas(self):
        """Am deve ter notas A(9), C(0), E(4) ativas."""
        template = CHORD_TEMPLATES["Am"]
        assert template[9] == 1  # A
        assert template[0] == 1  # C
        assert template[4] == 1  # E


# ── Testes: _detect_chords_from_chroma ───────────────────────────────────────


class TestDetectChordsFromChroma:

    def test_detecta_c_maior(self, chroma_c_major):
        """Deve detectar C maior a partir de um chroma com C, E, G."""
        chroma = chroma_c_major.reshape(12, 1)
        resultado = _detect_chords_from_chroma(chroma, sr=22050)

        assert len(resultado) == 1
        assert resultado[0].chord == "C"

    def test_detecta_a_menor(self, chroma_a_minor):
        """Deve detectar Am a partir de um chroma com A, C, E."""
        chroma = chroma_a_minor.reshape(12, 1)
        resultado = _detect_chords_from_chroma(chroma, sr=22050)

        assert len(resultado) == 1
        assert resultado[0].chord == "Am"

    def test_retorna_um_evento_por_frame(self, chroma_sequence_4_frames):
        """Deve retornar exatamente um ChordEvent por frame."""
        resultado = _detect_chords_from_chroma(chroma_sequence_4_frames, sr=22050)
        assert len(resultado) == 4

    def test_confianca_entre_zero_e_um(self, chroma_c_major):
        """Confiança deve estar entre 0.0 e 1.0."""
        chroma = chroma_c_major.reshape(12, 1)
        resultado = _detect_chords_from_chroma(chroma, sr=22050)

        assert 0.0 <= resultado[0].confidence <= 1.0

    def test_timestamps_sequenciais(self, chroma_sequence_4_frames):
        """Timestamps devem ser sequenciais e sem sobreposição."""
        resultado = _detect_chords_from_chroma(chroma_sequence_4_frames, sr=22050)

        for i in range(len(resultado) - 1):
            assert resultado[i].end_time == pytest.approx(
                resultado[i + 1].start_time, abs=0.001
            )


# ── Testes: _group_consecutive_chords ────────────────────────────────────────


class TestGroupConsecutiveChords:

    def test_agrupa_acordes_consecutivos_iguais(self):
        """Dois frames do mesmo acorde devem virar um único evento."""
        chords = [
            ChordEvent(chord="C", start_time=0.0, end_time=0.5, confidence=0.9),
            ChordEvent(chord="C", start_time=0.5, end_time=1.0, confidence=0.8),
            ChordEvent(chord="Am", start_time=1.0, end_time=1.5, confidence=0.85),
        ]
        resultado = _group_consecutive_chords(chords, min_duration=0.0)

        assert len(resultado) == 2
        assert resultado[0].chord == "C"
        assert resultado[0].start_time == pytest.approx(0.0)
        assert resultado[0].end_time == pytest.approx(1.0)
        assert resultado[1].chord == "Am"

    def test_remove_acordes_muito_curtos(self):
        """Acordes com duração menor que min_duration devem ser removidos."""
        chords = [
            ChordEvent(chord="C", start_time=0.0, end_time=0.1, confidence=0.9),
            ChordEvent(chord="Am", start_time=0.1, end_time=1.0, confidence=0.85),
        ]
        resultado = _group_consecutive_chords(chords, min_duration=0.5)

        assert len(resultado) == 1
        assert resultado[0].chord == "Am"

    def test_lista_vazia_retorna_lista_vazia(self):
        """Lista vazia deve retornar lista vazia."""
        assert _group_consecutive_chords([]) == []

    def test_confianca_e_media_dos_frames(self):
        """Confiança do acorde agrupado deve ser a média dos frames."""
        chords = [
            ChordEvent(chord="G", start_time=0.0, end_time=0.5, confidence=0.8),
            ChordEvent(chord="G", start_time=0.5, end_time=1.0, confidence=0.6),
        ]
        resultado = _group_consecutive_chords(chords, min_duration=0.0)

        assert len(resultado) == 1
        assert resultado[0].confidence == pytest.approx(0.7, abs=0.01)


# ── Testes: detect_chords (fluxo principal) ───────────────────────────────────


class TestDetectChords:

    @patch("app.pipeline.chord_detector.librosa.load")
    @patch("app.pipeline.chord_detector._extract_chroma")
    @patch("app.pipeline.chord_detector._detect_chords_from_chroma")
    @patch("app.pipeline.chord_detector._group_consecutive_chords")
    def test_fluxo_completo_com_sucesso(
        self,
        mock_group,
        mock_detect,
        mock_chroma,
        mock_load,
        harmonic_path,
    ):
        """Deve chamar todas as etapas em ordem e retornar sucesso."""
        mock_load.return_value = (np.zeros(22050), 22050)
        mock_chroma.return_value = np.zeros((12, 10))
        mock_detect.return_value = []
        mock_group.return_value = [
            ChordEvent(chord="C", start_time=0.0, end_time=2.0, confidence=0.9)
        ]

        resultado = detect_chords(harmonic_path)

        assert resultado.success is True
        assert len(resultado.chords) == 1
        assert resultado.chords[0].chord == "C"
        mock_load.assert_called_once()
        mock_chroma.assert_called_once()
        mock_detect.assert_called_once()
        mock_group.assert_called_once()

    @patch("app.pipeline.chord_detector.librosa.load")
    def test_retorna_erro_quando_falha(self, mock_load, harmonic_path):
        """Deve retornar ChordDetectionResult com success=False em caso de erro."""
        mock_load.side_effect = RuntimeError("Arquivo corrompido")

        resultado = detect_chords(harmonic_path)

        assert resultado.success is False
        assert resultado.error is not None
        assert "Arquivo corrompido" in resultado.error
        assert resultado.chords == []

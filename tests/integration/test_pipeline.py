"""
test_pipeline.py
Testes de integração do pipeline completo do MelodyTab AI.

Valida o fluxo do alinhamento até a formatação usando dados
reais sem mocks, garantindo que os módulos funcionam em conjunto.

Nota: Os testes de separação, detecção de acordes, melodia e
transcrição não são incluídos aqui pois dependem de arquivos
de áudio reais e APIs externas — validados manualmente.
"""

import pytest

from app.pipeline.aligner import align
from app.pipeline.chord_detector import ChordDetectionResult, ChordEvent
from app.pipeline.melody_detector import MelodyDetectionResult, NoteEvent
from app.pipeline.transcriber import TranscriptionResult, WordEvent


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def transcricao_completa():
    """Transcrição realista de um trecho musical."""
    return TranscriptionResult(
        words=[
            WordEvent("quando", 0.0, 0.5, 1.0),
            WordEvent("o", 0.5, 0.7, 1.0),
            WordEvent("sol", 0.7, 1.0, 1.0),
            WordEvent("se", 1.0, 1.2, 1.0),
            WordEvent("pôr", 1.2, 1.8, 1.0),
            WordEvent("estarei", 2.0, 2.6, 1.0),
            WordEvent("aqui", 2.6, 3.0, 1.0),
            WordEvent("pra", 3.0, 3.3, 1.0),
            WordEvent("te", 3.3, 3.5, 1.0),
            WordEvent("esperar", 3.5, 4.2, 1.0),
        ],
        full_text="quando o sol se pôr estarei aqui pra te esperar",
        language="pt",
        success=True,
    )


@pytest.fixture
def acordes_completos():
    """Acordes realistas para o trecho musical."""
    return ChordDetectionResult(
        chords=[
            ChordEvent("C", 0.0, 2.0, 0.92),
            ChordEvent("Am", 2.0, 4.2, 0.88),
        ],
        success=True,
    )


@pytest.fixture
def melodia_completa():
    """Melodia realista para o trecho musical."""
    return MelodyDetectionResult(
        notes=[
            NoteEvent("E", 4, 329.63, 0.0, 0.5, 0.9),
            NoteEvent("G", 4, 392.00, 0.5, 1.0, 0.88),
            NoteEvent("A", 4, 440.00, 1.0, 1.8, 0.92),
            NoteEvent("C", 5, 523.25, 2.0, 2.6, 0.87),
            NoteEvent("A", 4, 440.00, 2.6, 3.0, 0.85),
            NoteEvent("G", 4, 392.00, 3.0, 3.5, 0.9),
            NoteEvent("E", 4, 329.63, 3.5, 4.2, 0.88),
        ],
        success=True,
    )


# ── Testes: Alinhamento → Formatação ─────────────────────────────────────────


class TestAlignmentToFormatter:

    def test_pipeline_alinhamento_completo(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Pipeline de alinhamento deve retornar linhas com palavras."""
        resultado = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        assert resultado.success is True
        assert len(resultado.lines) > 0

        todas_palavras = [w for linha in resultado.lines for w in linha.words]
        assert len(todas_palavras) == len(transcricao_completa.words)

    def test_palavras_tem_acordes_corretos(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Palavras antes de 2.0s devem ter acorde C, após devem ter Am."""
        resultado = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        todas_palavras = [w for linha in resultado.lines for w in linha.words]

        palavras_c = [w for w in todas_palavras if w.start_time < 2.0 and w.chord]
        palavras_am = [w for w in todas_palavras if w.start_time > 2.0 and w.chord]

        assert all(w.chord == "C" for w in palavras_c)
        assert all(w.chord == "Am" for w in palavras_am)

    def test_mudancas_de_acorde_marcadas(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Deve haver exatamente 2 mudanças de acorde (C e Am)."""
        resultado = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        todas_palavras = [w for linha in resultado.lines for w in linha.words]
        mudancas = [w for w in todas_palavras if w.chord_change]

        assert len(mudancas) == 2
        assert mudancas[0].chord == "C"
        assert mudancas[1].chord == "Am"

    def test_notas_melodia_presentes(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Palavras com timestamp coincidente devem ter notas melódicas."""
        resultado = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        todas_palavras = [w for line in resultado.lines for w in line.words]
        com_notas = [w for w in todas_palavras if w.notes]

        assert len(com_notas) > 0

    def test_pipeline_sem_melodia(self, transcricao_completa, acordes_completos):
        """Pipeline deve funcionar sem melodia."""
        sem_melodia = MelodyDetectionResult(notes=[], success=False)

        resultado = align(transcricao_completa, acordes_completos, sem_melodia)

        assert resultado.success is True
        todas_palavras = [w for line in resultado.lines for w in line.words]
        assert all(w.notes == [] for w in todas_palavras)

    def test_pipeline_sem_acordes(self, transcricao_completa, melodia_completa):
        """Pipeline deve funcionar sem acordes."""
        sem_acordes = ChordDetectionResult(chords=[], success=False)

        resultado = align(transcricao_completa, sem_acordes, melodia_completa)

        assert resultado.success is True
        todas_palavras = [w for linha in resultado.lines for w in linha.words]
        assert all(w.chord is None for w in todas_palavras)


# ── Testes: Formatação da Cifra Bruta ────────────────────────────────────────


class TestCifraFormatting:

    def test_cifra_bruta_contem_acordes(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Cifra bruta deve conter os acordes detectados."""
        alignment = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        from app.pipeline.formatter import _build_raw_chord

        cifra_bruta = _build_raw_chord(alignment.lines)

        assert "[C]" in cifra_bruta
        assert "[Am]" in cifra_bruta

    def test_cifra_bruta_contem_letra(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Cifra bruta deve conter as palavras da letra."""
        alignment = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        from app.pipeline.formatter import _build_raw_chord

        cifra_bruta = _build_raw_chord(alignment.lines)

        assert "quando" in cifra_bruta
        assert "estarei" in cifra_bruta

    def test_cifra_bruta_contem_notas(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Cifra bruta deve conter as notas da melodia."""
        alignment = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        from app.pipeline.formatter import _build_raw_chord

        cifra_bruta = _build_raw_chord(alignment.lines)

        assert "E" in cifra_bruta or "G" in cifra_bruta


# ── Testes: PDF Export ────────────────────────────────────────────────────────


class TestPdfExport:

    def test_pipeline_completo_ate_pdf(
        self, transcricao_completa, acordes_completos, melodia_completa
    ):
        """Pipeline completo até PDF deve gerar bytes válidos."""
        from app.utils.pdf_export import export_to_pdf

        alignment = align(
            transcricao_completa,
            acordes_completos,
            melodia_completa,
        )

        from app.pipeline.formatter import _build_raw_chord

        cifra_bruta = _build_raw_chord(alignment.lines)

        pdf = export_to_pdf(
            cifra_bruta,
            titulo="Quando o Sol se Pôr",
            artista="Artista Teste",
        )

        assert isinstance(pdf, bytes)
        assert pdf[:4] == b"%PDF"
        assert len(pdf) > 1024

"""
test_formatter.py
Testes unitários para o módulo formatter.py
"""

from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.aligner import AlignedLine, AlignedWord, AlignmentResult
from app.pipeline.formatter import (
    _build_chord_line,
    _build_lyric_line,
    _build_melody_line,
    _build_raw_chord,
    _polish_with_llama,
    format_chord,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def words_linha_1():
    """Primeira linha: 'quando o sol' com acordes e notas."""
    return [
        AlignedWord("quando", 0.0, 0.5, chord="C", notes=["E"], chord_change=True),
        AlignedWord("o", 0.5, 0.7, chord="C", notes=["G"], chord_change=False),
        AlignedWord("sol", 0.7, 1.0, chord="Am", notes=["A"], chord_change=True),
    ]


@pytest.fixture
def words_sem_acorde():
    """Palavras sem acorde nem nota."""
    return [
        AlignedWord("palavra", 0.0, 0.5),
        AlignedWord("simples", 0.5, 1.0),
    ]


@pytest.fixture
def linha_1(words_linha_1):
    """AlignedLine com palavras da primeira linha."""
    return AlignedLine(
        words=words_linha_1,
        start_time=0.0,
        end_time=1.0,
    )


@pytest.fixture
def alignment_result(linha_1):
    """AlignmentResult com uma linha."""
    return AlignmentResult(lines=[linha_1], success=True)


# ── Testes: _build_lyric_line ─────────────────────────────────────────────────


class TestBuildLyricLine:

    def test_junta_palavras_com_espacos(self, words_linha_1):
        """Deve juntar todas as palavras com espaços."""
        resultado = _build_lyric_line(words_linha_1)
        assert resultado == "quando o sol"

    def test_lista_vazia_retorna_string_vazia(self):
        """Lista vazia deve retornar string vazia."""
        assert _build_lyric_line([]) == ""

    def test_uma_palavra(self):
        """Lista com uma palavra deve retornar só ela."""
        words = [AlignedWord("olá", 0.0, 0.5)]
        assert _build_lyric_line(words) == "olá"


# ── Testes: _build_chord_line ─────────────────────────────────────────────────


class TestBuildChordLine:

    def test_insere_acordes_em_colchetes(self, words_linha_1):
        """Acordes devem aparecer entre colchetes."""
        resultado = _build_chord_line(words_linha_1)
        assert "[C]" in resultado
        assert "[Am]" in resultado

    def test_sem_acorde_retorna_string_vazia(self, words_sem_acorde):
        """Palavras sem acorde devem retornar linha de acordes vazia."""
        resultado = _build_chord_line(words_sem_acorde)
        assert resultado.strip() == ""

    def test_acorde_posicionado_antes_da_palavra(self, words_linha_1):
        """Acorde [C] deve aparecer antes de [Am] na linha."""
        resultado = _build_chord_line(words_linha_1)
        pos_c = resultado.find("[C]")
        pos_am = resultado.find("[Am]")
        assert pos_c < pos_am

    def test_apenas_chord_change_gera_acorde(self):
        """Apenas palavras com chord_change=True devem gerar acorde na linha."""
        words = [
            AlignedWord("a", 0.0, 0.5, chord="G", chord_change=True),
            AlignedWord("b", 0.5, 1.0, chord="G", chord_change=False),
            AlignedWord("c", 1.0, 1.5, chord="G", chord_change=False),
        ]
        resultado = _build_chord_line(words)
        assert resultado.count("[G]") == 1


# ── Testes: _build_melody_line ────────────────────────────────────────────────


class TestBuildMelodyLine:

    def test_insere_notas_nas_posicoes_corretas(self, words_linha_1):
        """Notas devem aparecer na linha de melodia."""
        resultado = _build_melody_line(words_linha_1)
        assert "E" in resultado
        assert "G" in resultado
        assert "A" in resultado

    def test_sem_nota_retorna_string_vazia(self, words_sem_acorde):
        """Palavras sem nota devem retornar linha de melodia vazia."""
        resultado = _build_melody_line(words_sem_acorde)
        assert resultado.strip() == ""

    def test_notas_em_ordem(self, words_linha_1):
        """Notas devem aparecer na ordem das palavras."""
        resultado = _build_melody_line(words_linha_1)
        pos_e = resultado.find("E")
        pos_g = resultado.find("G")
        pos_a = resultado.find("A")
        assert pos_e < pos_g < pos_a

    def test_multiplas_notas_por_palavra(self):
        """Palavra com múltiplas notas deve exibir todas."""
        words = [
            AlignedWord("quando", 0.0, 0.5, notes=["E", "G", "A"]),
        ]
        resultado = _build_melody_line(words)
        assert "E" in resultado
        assert "G" in resultado
        assert "A" in resultado


# ── Testes: _build_raw_chord ──────────────────────────────────────────────────


class TestBuildRawChord:

    def test_gera_cifra_com_tres_linhas(self, linha_1):
        """Deve gerar linha de acordes, melodia e letra."""
        resultado = _build_raw_chord([linha_1])
        assert "[C]" in resultado
        assert "E" in resultado
        assert "quando" in resultado

    def test_linhas_separadas_por_quebra(self, linha_1):
        """Acordes, melodia e letra devem estar em linhas separadas."""
        resultado = _build_raw_chord([linha_1])
        linhas = resultado.split("\n")
        assert len(linhas) >= 3

    def test_lista_vazia_retorna_string_vazia(self):
        """Lista vazia deve retornar string vazia."""
        assert _build_raw_chord([]) == ""


# ── Testes: _polish_with_llama ────────────────────────────────────────────────


class TestPolishWithLlama:

    @patch("app.pipeline.formatter.Groq")
    def test_retorna_cifra_polida(self, mock_groq_class):
        """Deve retornar o texto retornado pelo LLaMA."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  cifra polida  "))]
        )
        mock_groq_class.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test"}):
            resultado = _polish_with_llama("cifra bruta")

        assert resultado == "cifra polida"

    @patch("app.pipeline.formatter.Groq")
    def test_chama_llama_com_temperatura_baixa(self, mock_groq_class):
        """Deve usar temperatura baixa para saída consistente."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="cifra"))]
        )
        mock_groq_class.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test"}):
            _polish_with_llama("cifra bruta")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == pytest.approx(0.1)
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_erro_sem_api_key(self):
        """Deve lançar RuntimeError sem GROQ_API_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                _polish_with_llama("cifra bruta")


# ── Testes: format_chord (fluxo principal) ────────────────────────────────────


class TestFormatChord:

    @patch("app.pipeline.formatter._polish_with_llama")
    def test_fluxo_completo_com_sucesso(self, mock_polish, alignment_result):
        """Deve retornar FormattedChord com sucesso."""
        mock_polish.return_value = "cifra polida pelo llama"

        resultado = format_chord(alignment_result)

        assert resultado.success is True
        assert resultado.polished == "cifra polida pelo llama"
        assert resultado.raw != ""
        mock_polish.assert_called_once()

    @patch("app.pipeline.formatter._polish_with_llama")
    def test_raw_contem_acordes_e_letra(self, mock_polish, alignment_result):
        """Cifra bruta deve conter acordes e letra."""
        mock_polish.return_value = "polida"

        resultado = format_chord(alignment_result)

        assert "[C]" in resultado.raw
        assert "quando" in resultado.raw

    @patch("app.pipeline.formatter._polish_with_llama")
    def test_retorna_erro_quando_polish_falha(self, mock_polish, alignment_result):
        """Deve retornar FormattedChord com success=False quando LLaMA falha."""
        mock_polish.side_effect = RuntimeError("LLaMA indisponível")

        resultado = format_chord(alignment_result)

        assert resultado.success is False
        assert "LLaMA indisponível" in resultado.error

    @patch("app.pipeline.formatter._polish_with_llama")
    def test_polish_recebe_cifra_bruta(self, mock_polish, alignment_result):
        """O LLaMA deve receber a cifra bruta como entrada."""
        mock_polish.return_value = "polida"

        resultado = format_chord(alignment_result)

        args = mock_polish.call_args.args
        assert args[0] == resultado.raw

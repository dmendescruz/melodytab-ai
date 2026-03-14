"""
test_transcriber.py
Testes unitários para o módulo transcriber.py
"""

from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.transcriber import (
    MAX_FILE_SIZE_BYTES,
    _parse_words,
    _validate_api_key,
    _validate_file,
    transcribe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vocals_path(tmp_path):
    """Cria um arquivo de áudio vocal falso de tamanho válido."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"x" * 1024)
    return audio


@pytest.fixture
def vocals_path_grande(tmp_path):
    """Cria um arquivo de áudio que excede o limite de 25 MB."""
    audio = tmp_path / "vocals_grande.wav"
    audio.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
    return audio


@pytest.fixture
def mock_word():
    """Retorna um dicionário simulando uma palavra da resposta da API Groq v1.1.1."""
    return {"word": "olá", "start": 0.5, "end": 0.9}


@pytest.fixture
def mock_response(mock_word):
    """Retorna um objeto simulando a resposta completa da API Groq."""
    response = MagicMock()
    response.text = "olá mundo"
    response.words = [mock_word]
    response.language = "pt"
    return response


# ── Testes: _validate_api_key ─────────────────────────────────────────────────


class TestValidateApiKey:

    def test_passa_quando_chave_configurada(self):
        """Não deve lançar exceção quando a chave está configurada."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
            _validate_api_key()  # não deve lançar exceção

    def test_erro_quando_chave_ausente(self):
        """Deve lançar RuntimeError quando a chave não está configurada."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                _validate_api_key()


# ── Testes: _validate_file ────────────────────────────────────────────────────


class TestValidateFile:

    def test_passa_com_arquivo_valido(self, vocals_path):
        """Não deve lançar exceção para arquivo válido."""
        _validate_file(vocals_path)  # não deve lançar exceção

    def test_erro_arquivo_inexistente(self, tmp_path):
        """Deve lançar FileNotFoundError para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            _validate_file(tmp_path / "nao_existe.wav")

    def test_erro_arquivo_muito_grande(self, vocals_path_grande):
        """Deve lançar ValueError para arquivo acima de 25 MB."""
        with pytest.raises(ValueError, match="25 MB"):
            _validate_file(vocals_path_grande)


# ── Testes: _parse_words ──────────────────────────────────────────────────────


class TestParseWords:

    def test_parse_palavras_com_timestamps(self, mock_response, mock_word):
        """Deve extrair palavras e timestamps corretamente da resposta."""
        resultado = _parse_words(mock_response)

        assert len(resultado) == 1
        assert resultado[0].word == "olá"
        assert resultado[0].start_time == pytest.approx(0.5, abs=0.001)
        assert resultado[0].end_time == pytest.approx(0.9, abs=0.001)

    def test_fallback_sem_timestamps(self):
        """Deve retornar o texto completo quando API não retorna timestamps."""
        response = MagicMock()
        response.text = "texto sem timestamps"
        response.words = None

        resultado = _parse_words(response)

        assert len(resultado) == 1
        assert resultado[0].word == "texto sem timestamps"
        assert resultado[0].start_time == pytest.approx(0.0)
        assert resultado[0].end_time == pytest.approx(0.0)

    def test_ignora_palavras_vazias(self):
        """Deve ignorar itens com palavra vazia na resposta."""
        palavra_vazia = MagicMock()
        palavra_vazia.word = "  "
        palavra_vazia.start = 0.0
        palavra_vazia.end = 0.1

        response = MagicMock()
        response.words = [palavra_vazia]

        resultado = _parse_words(response)

        assert len(resultado) == 0

    def test_timestamps_arredondados(self, mock_response):
        """Timestamps devem ser arredondados para 3 casas decimais."""
        mock_response.words[0]["start"] = 0.123456
        mock_response.words[0]["end"]   = 0.987654

        resultado = _parse_words(mock_response)

        assert resultado[0].start_time == pytest.approx(0.123, abs=0.001)
        assert resultado[0].end_time == pytest.approx(0.988, abs=0.001)


# ── Testes: transcribe (fluxo principal) ─────────────────────────────────────


class TestTranscribe:

    @patch("app.pipeline.transcriber.Groq")
    def test_fluxo_completo_com_sucesso(
        self, mock_groq_class, vocals_path, mock_response
    ):
        """Deve retornar TranscriptionResult com sucesso."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
            resultado = transcribe(vocals_path)

        assert resultado.success is True
        assert resultado.full_text == "olá mundo"
        assert resultado.language == "pt"
        assert len(resultado.words) == 1
        assert resultado.words[0].word == "olá"

    @patch("app.pipeline.transcriber.Groq")
    def test_chama_api_com_parametros_corretos(
        self, mock_groq_class, vocals_path, mock_response
    ):
        """Deve chamar a API com o modelo e formato corretos."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
            transcribe(vocals_path)

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-large-v3"
        assert call_kwargs["response_format"] == "verbose_json"
        assert "word" in call_kwargs["timestamp_granularities"]

    def test_retorna_erro_sem_api_key(self, vocals_path):
        """Deve retornar TranscriptionResult com success=False sem API key."""
        with patch.dict("os.environ", {}, clear=True):
            resultado = transcribe(vocals_path)

        assert resultado.success is False
        assert resultado.error is not None
        assert "GROQ_API_KEY" in resultado.error

    def test_retorna_erro_arquivo_inexistente(self, tmp_path):
        """Deve retornar TranscriptionResult com success=False para arquivo inexistente."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
            resultado = transcribe(tmp_path / "nao_existe.wav")

        assert resultado.success is False
        assert resultado.error is not None

    @patch("app.pipeline.transcriber.Groq")
    def test_retorna_erro_quando_api_falha(self, mock_groq_class, vocals_path):
        """Deve retornar TranscriptionResult com success=False quando API lança exceção."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = Exception(
            "API indisponível"
        )
        mock_groq_class.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
            resultado = transcribe(vocals_path)

        assert resultado.success is False
        assert "API indisponível" in resultado.error

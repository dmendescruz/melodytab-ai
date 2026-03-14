"""
test_groq_client.py
Testes unitários para o módulo groq_client.py
"""

from unittest.mock import MagicMock, patch
import pytest

from groq import RateLimitError, APIStatusError

from app.api.groq_client import GroqClient, MAX_RETRIES


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Instancia GroqClient com chave de API mockada."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
        return GroqClient()


@pytest.fixture
def mock_transcription_response():
    """Simula resposta de transcrição da API Groq."""
    word = MagicMock()
    word.word = "olá"
    word.start = 0.5
    word.end = 0.9

    response = MagicMock()
    response.text = "olá mundo"
    response.words = [word]
    response.language = "pt"
    return response


# ── Testes: inicialização ─────────────────────────────────────────────────────


class TestGroqClientInit:

    def test_inicializa_com_chave_valida(self):
        """Deve inicializar com sucesso quando a chave está configurada."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test123"}):
            client = GroqClient()
            assert client is not None

    def test_erro_sem_api_key(self):
        """Deve lançar RuntimeError quando a chave não está configurada."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                GroqClient()


# ── Testes: transcribe ────────────────────────────────────────────────────────


class TestGroqClientTranscribe:

    def test_transcricao_bem_sucedida(
        self, client, mock_transcription_response, tmp_path
    ):
        """Deve retornar resposta da API em caso de sucesso."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"fake audio")

        client._client.audio.transcriptions.create = MagicMock(
            return_value=mock_transcription_response
        )

        resultado = client.transcribe(str(audio))

        assert resultado.text == "olá mundo"
        client._client.audio.transcriptions.create.assert_called_once()

    def test_chama_com_modelo_correto(
        self, client, mock_transcription_response, tmp_path
    ):
        """Deve chamar a API com whisper-large-v3."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"fake audio")

        client._client.audio.transcriptions.create = MagicMock(
            return_value=mock_transcription_response
        )

        client.transcribe(str(audio))

        call_kwargs = client._client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-large-v3"
        assert call_kwargs["response_format"] == "verbose_json"


# ── Testes: complete ──────────────────────────────────────────────────────────


class TestGroqClientComplete:

    def test_complete_bem_sucedido(self, client):
        """Deve retornar texto gerado pelo LLaMA."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  cifra formatada  "

        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        resultado = client.complete("formate esta cifra")

        assert resultado == "cifra formatada"

    def test_chama_com_modelo_correto(self, client):
        """Deve chamar a API com llama-3.3-70b-versatile."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "resposta"

        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        client.complete("prompt")

        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert call_kwargs["temperature"] == pytest.approx(0.1)


# ── Testes: retry ─────────────────────────────────────────────────────────────


class TestGroqClientRetry:

    @patch("app.api.groq_client.time.sleep")
    def test_retenta_em_rate_limit(self, mock_sleep, client, tmp_path):
        """Deve tentar novamente quando ocorre RateLimitError."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"fake audio")

        mock_response = MagicMock()
        mock_response.text = "texto"
        mock_response.words = []

        client._client.audio.transcriptions.create = MagicMock(
            side_effect=[
                RateLimitError.__new__(RateLimitError),
                mock_response,
            ]
        )

        resultado = client.transcribe(str(audio))

        assert resultado.text == "texto"
        assert mock_sleep.called

    @patch("app.api.groq_client.time.sleep")
    def test_falha_apos_max_retries(self, mock_sleep, client, tmp_path):
        """Deve lançar RuntimeError após esgotar todas as tentativas."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"fake audio")

        client._client.audio.transcriptions.create = MagicMock(
            side_effect=RateLimitError.__new__(RateLimitError)
        )

        with pytest.raises(RuntimeError, match=f"{MAX_RETRIES} tentativas"):
            client.transcribe(str(audio))

        assert mock_sleep.call_count == MAX_RETRIES - 1

    @patch("app.api.groq_client.time.sleep")
    def test_nao_retenta_erro_400(self, mock_sleep, client):
        """Não deve retentar erros 4xx."""
        erro_400 = APIStatusError.__new__(APIStatusError)
        erro_400.status_code = 400

        client._client.chat.completions.create = MagicMock(side_effect=erro_400)

        with pytest.raises(RuntimeError, match="erro 400"):
            client.complete("prompt")

        mock_sleep.assert_not_called()

"""
test_huggingface_client.py
Testes unitários para o módulo huggingface_client.py
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.api.huggingface_client import HuggingFaceClient


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Instancia HuggingFaceClient com token mockado."""
    with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "hf_test123"}):
        return HuggingFaceClient()


@pytest.fixture
def audio_path(tmp_path):
    """Cria um arquivo de áudio falso."""
    audio = tmp_path / "harmonic.wav"
    audio.write_bytes(b"fake audio content")
    return audio


@pytest.fixture
def mock_chords_response():
    """Simula resposta de detecção de acordes da API."""
    return [
        {"label": "C", "score": 0.92, "start": 0.0, "end": 1.5},
        {"label": "Am", "score": 0.87, "start": 1.5, "end": 3.0},
    ]


# ── Testes: inicialização ─────────────────────────────────────────────────────


class TestHuggingFaceClientInit:

    def test_inicializa_com_token_valido(self):
        """Deve inicializar com sucesso quando o token está configurado."""
        with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "hf_test"}):
            client = HuggingFaceClient()
            assert client is not None

    def test_erro_sem_token(self):
        """Deve lançar RuntimeError quando o token não está configurado."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN"):
                HuggingFaceClient()

    def test_headers_contem_authorization(self):
        """Headers devem conter o token de autorização."""
        with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "hf_test"}):
            client = HuggingFaceClient()
            assert "Authorization" in client._headers
            assert "hf_test" in client._headers["Authorization"]


# ── Testes: detect_chords ─────────────────────────────────────────────────────


class TestHuggingFaceClientDetectChords:

    @patch("app.api.huggingface_client.requests.post")
    def test_deteccao_bem_sucedida(
        self, mock_post, client, audio_path, mock_chords_response
    ):
        """Deve retornar lista de acordes em caso de sucesso."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_chords_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        resultado = client.detect_chords(audio_path)

        assert len(resultado) == 2
        assert resultado[0]["label"] == "C"
        assert resultado[1]["label"] == "Am"

    @patch("app.api.huggingface_client.time.sleep")
    @patch("app.api.huggingface_client.requests.post")
    def test_aguarda_quando_modelo_carregando(
        self, mock_post, mock_sleep, client, audio_path, mock_chords_response
    ):
        """Deve aguardar e tentar novamente quando modelo está carregando (503)."""
        resposta_503 = MagicMock()
        resposta_503.status_code = 503
        resposta_503.json.return_value = {"estimated_time": 5}

        resposta_200 = MagicMock()
        resposta_200.status_code = 200
        resposta_200.json.return_value = mock_chords_response
        resposta_200.raise_for_status.return_value = None

        mock_post.side_effect = [resposta_503, resposta_200]

        resultado = client.detect_chords(audio_path)

        assert len(resultado) == 2
        assert mock_sleep.called

    @patch("app.api.huggingface_client.requests.post")
    def test_erro_401_nao_retentado(self, mock_post, client, audio_path):
        """Erro 401 não deve ser retentado."""
        erro_http = requests.HTTPError("401 Unauthorized")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        erro_http.response = mock_resp

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = erro_http
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="401"):
            client.detect_chords(audio_path)

        assert mock_post.call_count == 1


# ── Testes: is_available ──────────────────────────────────────────────────────


class TestHuggingFaceClientIsAvailable:

    @patch("app.api.huggingface_client.requests.get")
    def test_disponivel_quando_status_200(self, mock_get, client):
        """Deve retornar True quando API retorna 200."""
        mock_get.return_value = MagicMock(status_code=200)
        assert client.is_available() is True

    @patch("app.api.huggingface_client.requests.get")
    def test_disponivel_quando_modelo_carregando(self, mock_get, client):
        """Deve retornar True quando modelo está carregando (503)."""
        mock_get.return_value = MagicMock(status_code=503)
        assert client.is_available() is True

    @patch("app.api.huggingface_client.requests.get")
    def test_indisponivel_quando_excecao(self, mock_get, client):
        """Deve retornar False quando ocorre exceção de conexão."""
        mock_get.side_effect = requests.ConnectionError("sem conexão")
        assert client.is_available() is False

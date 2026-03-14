"""
test_replicate_client.py
Testes unitários para o módulo replicate_client.py
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.api.replicate_client import ReplicateClient, _download_file


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Instancia ReplicateClient com token mockado."""
    with patch.dict("os.environ", {"REPLICATE_API_TOKEN": "r8_test123"}):
        return ReplicateClient()


@pytest.fixture
def audio_path(tmp_path):
    """Cria um arquivo de áudio falso."""
    audio = tmp_path / "musica.mp3"
    audio.write_bytes(b"fake audio content")
    return audio


@pytest.fixture
def mock_stems_output(tmp_path):
    """Simula output do Replicate com URLs das faixas."""
    return {
        "vocals": "https://replicate.com/vocals.wav",
        "bass": "https://replicate.com/bass.wav",
        "drums": "https://replicate.com/drums.wav",
        "other": "https://replicate.com/other.wav",
    }


# ── Testes: inicialização ─────────────────────────────────────────────────────


class TestReplicateClientInit:

    def test_inicializa_com_token_valido(self):
        """Deve inicializar com sucesso quando o token está configurado."""
        with patch.dict("os.environ", {"REPLICATE_API_TOKEN": "r8_test"}):
            client = ReplicateClient()
            assert client is not None

    def test_erro_sem_token(self):
        """Deve lançar RuntimeError quando o token não está configurado."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="REPLICATE_API_TOKEN"):
                ReplicateClient()


# ── Testes: separate_stems ────────────────────────────────────────────────────


class TestReplicateClientSeparateStems:

    @patch("app.api.replicate_client._download_file")
    @patch("app.api.replicate_client.replicate.run")
    def test_separacao_bem_sucedida(
        self, mock_run, mock_download, client, audio_path, mock_stems_output
    ):
        """Deve retornar dicionário com caminhos das faixas."""
        mock_run.return_value = mock_stems_output
        mock_download.return_value = None

        resultado = client.separate_stems(audio_path)

        assert "vocals" in resultado
        assert "bass" in resultado
        assert "drums" in resultado
        assert "other" in resultado

    @patch("app.api.replicate_client._download_file")
    @patch("app.api.replicate_client.replicate.run")
    def test_chama_modelo_correto(
        self, mock_run, mock_download, client, audio_path, mock_stems_output
    ):
        """Deve chamar o modelo Demucs correto."""
        mock_run.return_value = mock_stems_output

        client.separate_stems(audio_path)

        call_args = mock_run.call_args
        assert "cjwbw/demucs:" in call_args.args[0]

    @patch("app.api.replicate_client.replicate.run")
    def test_erro_quando_url_ausente(self, mock_run, client, audio_path):
        """Deve lançar RuntimeError quando URL de faixa está ausente."""
        mock_run.return_value = {
            "vocals": None,
            "bass": None,
            "drums": None,
            "other": None,
        }

        with pytest.raises(RuntimeError, match="vocals"):
            client.separate_stems(audio_path)


# ── Testes: _download_file ────────────────────────────────────────────────────


class TestDownloadFile:

    def test_download_bem_sucedido(self, tmp_path):
        """Deve salvar o arquivo no destino correto."""
        destino = tmp_path / "faixa.wav"

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"audio", b"data"]
        mock_response.raise_for_status.return_value = None

        with patch("app.api.replicate_client.requests.get", return_value=mock_response):
            _download_file("https://exemplo.com/faixa.wav", destino)

        assert destino.exists()

    def test_erro_em_download_falho(self, tmp_path):
        """Deve lançar exceção quando o download falha."""
        destino = tmp_path / "faixa.wav"

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("app.api.replicate_client.requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                _download_file("https://exemplo.com/faixa.wav", destino)

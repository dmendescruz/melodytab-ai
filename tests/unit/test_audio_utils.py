"""
test_audio_utils.py
Testes unitários para o módulo audio.py
"""

from unittest.mock import MagicMock, patch

import pytest

from app.utils.audio import (
    GROQ_MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_FORMATS,
    format_duration,
    get_audio_info,
    is_too_large_for_groq,
    needs_conversion_for_groq,
    save_uploaded_file,
    validate_audio_file,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def audio_mp3(tmp_path):
    """Cria um arquivo MP3 falso de tamanho válido."""
    audio = tmp_path / "musica.mp3"
    audio.write_bytes(b"x" * 1024)
    return audio


@pytest.fixture
def audio_wav(tmp_path):
    """Cria um arquivo WAV falso de tamanho válido."""
    audio = tmp_path / "musica.wav"
    audio.write_bytes(b"x" * 1024)
    return audio


@pytest.fixture
def audio_grande(tmp_path):
    """Cria um arquivo maior que o limite máximo."""
    audio = tmp_path / "grande.mp3"
    audio.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
    return audio


@pytest.fixture
def audio_groq_grande(tmp_path):
    """Cria um arquivo maior que o limite da API Groq."""
    audio = tmp_path / "groq_grande.mp3"
    audio.write_bytes(b"x" * (GROQ_MAX_FILE_SIZE_BYTES + 1))
    return audio


# ── Testes: validate_audio_file ───────────────────────────────────────────────


class TestValidateAudioFile:

    def test_valida_mp3_valido(self, audio_mp3):
        """Deve retornar True para MP3 válido."""
        valido, msg = validate_audio_file(audio_mp3)
        assert valido is True
        assert msg == "OK"

    def test_valida_wav_valido(self, audio_wav):
        """Deve retornar True para WAV válido."""
        valido, msg = validate_audio_file(audio_wav)
        assert valido is True
        assert msg == "OK"

    def test_rejeita_arquivo_inexistente(self, tmp_path):
        """Deve retornar False para arquivo inexistente."""
        valido, msg = validate_audio_file(tmp_path / "nao_existe.mp3")
        assert valido is False
        assert "não encontrado" in msg

    def test_rejeita_formato_invalido(self, tmp_path):
        """Deve retornar False para formato não suportado."""
        arquivo = tmp_path / "video.mp4"
        arquivo.write_bytes(b"x" * 1024)
        valido, msg = validate_audio_file(arquivo)
        assert valido is False
        assert "não suportado" in msg

    def test_rejeita_arquivo_vazio(self, tmp_path):
        """Deve retornar False para arquivo vazio."""
        arquivo = tmp_path / "vazio.mp3"
        arquivo.write_bytes(b"")
        valido, msg = validate_audio_file(arquivo)
        assert valido is False
        assert "vazio" in msg

    def test_rejeita_arquivo_muito_grande(self, audio_grande):
        """Deve retornar False para arquivo acima do limite."""
        valido, msg = validate_audio_file(audio_grande)
        assert valido is False
        assert "grande" in msg

    def test_formatos_suportados_sao_validos(self, tmp_path):
        """Todos os formatos suportados devem ser aceitos."""
        for fmt in SUPPORTED_FORMATS:
            arquivo = tmp_path / f"audio{fmt}"
            arquivo.write_bytes(b"x" * 1024)
            valido, _ = validate_audio_file(arquivo)
            assert valido is True, f"Formato {fmt} deveria ser válido"


# ── Testes: format_duration ───────────────────────────────────────────────────


class TestFormatDuration:

    def test_formata_minutos_e_segundos(self):
        """225 segundos deve formatar como '3:45'."""
        assert format_duration(225) == "3:45"

    def test_formata_menos_de_um_minuto(self):
        """45 segundos deve formatar como '0:45'."""
        assert format_duration(45) == "0:45"

    def test_formata_exatamente_um_minuto(self):
        """60 segundos deve formatar como '1:00'."""
        assert format_duration(60) == "1:00"

    def test_formata_horas(self):
        """3690 segundos deve formatar como '1:01:30'."""
        assert format_duration(3690) == "1:01:30"

    def test_formata_zero(self):
        """0 segundos deve formatar como '0:00'."""
        assert format_duration(0) == "0:00"

    def test_formata_segundos_com_zero_a_esquerda(self):
        """65 segundos deve formatar como '1:05'."""
        assert format_duration(65) == "1:05"


# ── Testes: needs_conversion_for_groq ────────────────────────────────────────


class TestNeedsConversionForGroq:

    def test_mp3_nao_precisa_conversao(self, tmp_path):
        """MP3 não precisa de conversão para Groq."""
        assert needs_conversion_for_groq(tmp_path / "audio.mp3") is False

    def test_wav_nao_precisa_conversao(self, tmp_path):
        """WAV não precisa de conversão para Groq."""
        assert needs_conversion_for_groq(tmp_path / "audio.wav") is False

    def test_m4a_nao_precisa_conversao(self, tmp_path):
        """M4A não precisa de conversão para Groq."""
        assert needs_conversion_for_groq(tmp_path / "audio.m4a") is False

    def test_flac_precisa_conversao(self, tmp_path):
        """FLAC precisa de conversão para Groq."""
        assert needs_conversion_for_groq(tmp_path / "audio.flac") is True

    def test_ogg_precisa_conversao(self, tmp_path):
        """OGG precisa de conversão para Groq."""
        assert needs_conversion_for_groq(tmp_path / "audio.ogg") is True


# ── Testes: is_too_large_for_groq ─────────────────────────────────────────────


class TestIsTooLargeForGroq:

    def test_arquivo_pequeno_nao_e_grande(self, audio_mp3):
        """Arquivo pequeno não deve exceder limite da Groq."""
        assert is_too_large_for_groq(audio_mp3) is False

    def test_arquivo_grande_excede_limite(self, audio_groq_grande):
        """Arquivo acima de 25MB deve exceder limite da Groq."""
        assert is_too_large_for_groq(audio_groq_grande) is True


# ── Testes: get_audio_info ────────────────────────────────────────────────────


class TestGetAudioInfo:

    @patch("app.utils.audio.sf.info")
    def test_retorna_info_correta(self, mock_sf_info, audio_mp3):
        """Deve retornar dicionário com informações do áudio."""
        mock_info = MagicMock()
        mock_info.duration = 180.5
        mock_info.samplerate = 44100
        mock_info.channels = 2
        mock_sf_info.return_value = mock_info

        info = get_audio_info(audio_mp3)

        assert info["duration"] == pytest.approx(180.5)
        assert info["sample_rate"] == 44100
        assert info["channels"] == 2
        assert info["format"] == "mp3"

    @patch("app.utils.audio.sf.info")
    def test_retorna_zeros_em_caso_de_erro(self, mock_sf_info, audio_mp3):
        """Deve retornar zeros quando não consegue ler o arquivo."""
        mock_sf_info.side_effect = Exception("Arquivo corrompido")

        info = get_audio_info(audio_mp3)

        assert info["duration"] == 0.0
        assert info["sample_rate"] == 0
        assert info["channels"] == 0


# ── Testes: save_uploaded_file ────────────────────────────────────────────────


class TestSaveUploadedFile:

    def test_salva_arquivo_corretamente(self, tmp_path):
        """Deve salvar o arquivo no diretório temporário."""
        uploaded = MagicMock()
        uploaded.name = "musica.mp3"
        uploaded.getbuffer.return_value = b"fake audio data"

        with patch("app.utils.audio.tempfile.mkdtemp", return_value=str(tmp_path)):
            resultado = save_uploaded_file(uploaded)

        assert resultado.exists()
        assert resultado.suffix == ".mp3"
        assert resultado.read_bytes() == b"fake audio data"

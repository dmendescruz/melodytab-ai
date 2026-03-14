"""
test_separator.py
Testes unitários para o módulo separator.py
"""

from unittest.mock import patch

import pytest

from app.pipeline.separator import (
    SeparationMode,
    SeparationResult,
    _has_audio_content,
    _select_harmonic,
    separate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def audio_path(tmp_path):
    """Cria um arquivo de áudio falso para os testes."""
    audio = tmp_path / "musica.mp3"
    audio.write_bytes(b"fake audio content")
    return audio


@pytest.fixture
def stem_dir(tmp_path):
    """Cria um diretório com faixas separadas falsas."""
    stems = tmp_path / "htdemucs" / "musica"
    stems.mkdir(parents=True)

    for nome in ["vocals", "bass", "drums", "other"]:
        faixa = stems / f"{nome}.wav"
        faixa.write_bytes(
            b"x" * 2048
        )  # conteúdo suficiente para passar _has_audio_content

    return stems


@pytest.fixture
def stem_dir_6s(tmp_path):
    """Cria um diretório com faixas do modelo htdemucs_6s."""
    stems = tmp_path / "htdemucs_6s" / "musica"
    stems.mkdir(parents=True)

    for nome in ["vocals", "bass", "drums", "other", "guitar", "piano"]:
        faixa = stems / f"{nome}.wav"
        faixa.write_bytes(b"x" * 2048)

    return stems


# ── Testes: _has_audio_content ────────────────────────────────────────────────


class TestHasAudioContent:

    def test_arquivo_com_conteudo_suficiente(self, tmp_path):
        """Deve retornar True para arquivo maior que o mínimo."""
        arquivo = tmp_path / "audio.wav"
        arquivo.write_bytes(b"x" * 2048)
        assert _has_audio_content(arquivo) is True

    def test_arquivo_muito_pequeno(self, tmp_path):
        """Deve retornar False para arquivo menor que 1024 bytes."""
        arquivo = tmp_path / "audio.wav"
        arquivo.write_bytes(b"x" * 10)
        assert _has_audio_content(arquivo) is False

    def test_arquivo_inexistente(self, tmp_path):
        """Deve retornar False para arquivo que não existe."""
        arquivo = tmp_path / "nao_existe.wav"
        assert _has_audio_content(arquivo) is False


# ── Testes: _select_harmonic ──────────────────────────────────────────────────


class TestSelectHarmonic:

    def test_prioriza_guitarra_quando_disponivel(self, tmp_path):
        """Deve selecionar guitarra quando disponível e com conteúdo."""
        guitar = tmp_path / "guitar.wav"
        piano = tmp_path / "piano.wav"
        other = tmp_path / "other.wav"

        guitar.write_bytes(b"x" * 2048)
        piano.write_bytes(b"x" * 2048)
        other.write_bytes(b"x" * 2048)

        caminho, fonte = _select_harmonic(guitar=guitar, piano=piano, other=other)

        assert caminho == guitar
        assert fonte == "guitar"

    def test_usa_piano_quando_guitarra_ausente(self, tmp_path):
        """Deve selecionar piano quando guitarra não está disponível."""
        piano = tmp_path / "piano.wav"
        other = tmp_path / "other.wav"

        piano.write_bytes(b"x" * 2048)
        other.write_bytes(b"x" * 2048)

        caminho, fonte = _select_harmonic(guitar=None, piano=piano, other=other)

        assert caminho == piano
        assert fonte == "piano"

    def test_usa_other_quando_guitarra_e_piano_ausentes(self, tmp_path):
        """Deve selecionar other quando guitarra e piano não estão disponíveis."""
        other = tmp_path / "other.wav"
        other.write_bytes(b"x" * 2048)

        caminho, fonte = _select_harmonic(guitar=None, piano=None, other=other)

        assert caminho == other
        assert fonte == "other"

    def test_ignora_guitarra_sem_conteudo(self, tmp_path):
        """Deve ignorar guitarra com arquivo muito pequeno e usar piano."""
        guitar = tmp_path / "guitar.wav"
        piano = tmp_path / "piano.wav"
        other = tmp_path / "other.wav"

        guitar.write_bytes(b"x" * 10)  # muito pequeno — inválido
        piano.write_bytes(b"x" * 2048)
        other.write_bytes(b"x" * 2048)

        caminho, fonte = _select_harmonic(guitar=guitar, piano=piano, other=other)

        assert caminho == piano
        assert fonte == "piano"


# ── Testes: separate (fluxo principal) ───────────────────────────────────────


class TestSeparate:

    @patch("app.pipeline.separator._separate_local")
    def test_usa_local_quando_disponivel(self, mock_local, audio_path, tmp_path):
        """Deve usar Demucs local e retornar modo LOCAL."""
        mock_result = SeparationResult(
            vocals=tmp_path / "vocals.wav",
            harmonic=tmp_path / "other.wav",
            bass=tmp_path / "bass.wav",
            drums=tmp_path / "drums.wav",
            other=tmp_path / "other.wav",
            mode=SeparationMode.LOCAL,
            success=True,
            harmonic_source="other",
        )
        mock_local.return_value = mock_result

        result = separate(audio_path)

        assert result.success is True
        assert result.mode == SeparationMode.LOCAL
        mock_local.assert_called_once_with(audio_path)

    @patch("app.pipeline.separator._separate_replicate")
    @patch("app.pipeline.separator._separate_local")
    def test_usa_fallback_quando_local_falha(
        self, mock_local, mock_replicate, audio_path, tmp_path
    ):
        """Deve usar Replicate quando Demucs local lançar exceção."""
        mock_local.side_effect = RuntimeError("Demucs falhou")

        mock_result = SeparationResult(
            vocals=tmp_path / "vocals.wav",
            harmonic=tmp_path / "other.wav",
            bass=tmp_path / "bass.wav",
            drums=tmp_path / "drums.wav",
            other=tmp_path / "other.wav",
            mode=SeparationMode.REPLICATE,
            success=True,
            harmonic_source="other",
        )
        mock_replicate.return_value = mock_result

        result = separate(audio_path)

        assert result.success is True
        assert result.mode == SeparationMode.REPLICATE
        mock_local.assert_called_once()
        mock_replicate.assert_called_once()

    @patch("app.pipeline.separator._separate_replicate")
    @patch("app.pipeline.separator._separate_local")
    def test_retorna_erro_quando_tudo_falha(
        self, mock_local, mock_replicate, audio_path
    ):
        """Deve retornar SeparationResult com success=False quando ambos falham."""
        mock_local.side_effect = RuntimeError("Demucs falhou")
        mock_replicate.side_effect = RuntimeError("Replicate falhou")

        result = separate(audio_path)

        assert result.success is False
        assert result.error is not None
        assert "Replicate falhou" in result.error

    @patch("app.pipeline.separator._separate_local")
    def test_nao_chama_replicate_quando_local_funciona(
        self, mock_local, audio_path, tmp_path
    ):
        """Não deve chamar Replicate quando Demucs local funciona."""
        mock_local.return_value = SeparationResult(
            vocals=tmp_path / "vocals.wav",
            harmonic=tmp_path / "other.wav",
            bass=tmp_path / "bass.wav",
            drums=tmp_path / "drums.wav",
            other=tmp_path / "other.wav",
            mode=SeparationMode.LOCAL,
            success=True,
            harmonic_source="other",
        )

        with patch("app.pipeline.separator._separate_replicate") as mock_replicate:
            result = separate(audio_path)
            mock_replicate.assert_not_called()

        assert result.success is True


# ── Testes: modelo htdemucs_6s ────────────────────────────────────────────────


class TestModelo6Stems:

    def test_seleciona_guitarra_no_modelo_6s(self, tmp_path):
        """Com htdemucs_6s, deve priorizar guitarra sobre other."""
        guitar = tmp_path / "guitar.wav"
        piano = tmp_path / "piano.wav"
        other = tmp_path / "other.wav"

        guitar.write_bytes(b"x" * 2048)
        piano.write_bytes(b"x" * 2048)
        other.write_bytes(b"x" * 2048)

        caminho, fonte = _select_harmonic(guitar=guitar, piano=piano, other=other)

        assert fonte == "guitar"
        assert caminho == guitar

    def test_seleciona_piano_quando_guitarra_vazia_no_modelo_6s(self, tmp_path):
        """Com htdemucs_6s, deve usar piano quando guitarra está vazia."""
        guitar = tmp_path / "guitar.wav"
        piano = tmp_path / "piano.wav"
        other = tmp_path / "other.wav"

        guitar.write_bytes(b"x" * 10)  # silêncio — inválido
        piano.write_bytes(b"x" * 2048)
        other.write_bytes(b"x" * 2048)

        caminho, fonte = _select_harmonic(guitar=guitar, piano=piano, other=other)

        assert fonte == "piano"
        assert caminho == piano

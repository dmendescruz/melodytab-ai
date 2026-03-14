"""
test_pdf_export.py
Testes unitários para o módulo pdf_export.py
"""

from pathlib import Path

import pytest

from app.utils.pdf_export import (
    _build_elements,
    _build_styles,
    export_to_pdf,
    export_to_pdf_file,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cifra_simples():
    """Retorna uma cifra simples para testes."""
    return "[C] quando o sol se [Am] pôr\n   mi  sol  lá      mi ré dó\nquando o sol se pôr"


@pytest.fixture
def cifra_vazia():
    """Retorna uma cifra vazia."""
    return ""


# ── Testes: export_to_pdf ─────────────────────────────────────────────────────


class TestExportToPdf:

    def test_retorna_bytes(self, cifra_simples):
        """Deve retornar bytes do PDF gerado."""
        resultado = export_to_pdf(cifra_simples, titulo="Teste")
        assert isinstance(resultado, bytes)

    def test_pdf_nao_vazio(self, cifra_simples):
        """PDF gerado não deve ser vazio."""
        resultado = export_to_pdf(cifra_simples, titulo="Teste")
        assert len(resultado) > 0

    def test_pdf_começa_com_header_correto(self, cifra_simples):
        """PDF deve começar com o header correto (%PDF)."""
        resultado = export_to_pdf(cifra_simples, titulo="Teste")
        assert resultado[:4] == b"%PDF"

    def test_gera_pdf_com_artista(self, cifra_simples):
        """Deve gerar PDF com artista sem erros."""
        resultado = export_to_pdf(
            cifra_simples,
            titulo="Minha Música",
            artista="Artista Teste",
        )
        assert isinstance(resultado, bytes)
        assert len(resultado) > 0

    def test_gera_pdf_sem_artista(self, cifra_simples):
        """Deve gerar PDF sem artista sem erros."""
        resultado = export_to_pdf(cifra_simples, titulo="Minha Música")
        assert isinstance(resultado, bytes)
        assert len(resultado) > 0

    def test_gera_pdf_com_cifra_vazia(self, cifra_vazia):
        """Deve gerar PDF mesmo com cifra vazia."""
        resultado = export_to_pdf(cifra_vazia, titulo="Vazia")
        assert isinstance(resultado, bytes)
        assert len(resultado) > 0

    def test_pdf_maior_com_cifra_longa(self):
        """PDF com cifra longa deve ser maior que com cifra curta."""
        cifra_curta = "[C] verso curto"
        cifra_longa = "\n".join([f"[C] linha {i}" for i in range(50)])

        pdf_curto = export_to_pdf(cifra_curta, titulo="Curta")
        pdf_longo = export_to_pdf(cifra_longa, titulo="Longa")

        assert len(pdf_longo) > len(pdf_curto)


# ── Testes: export_to_pdf_file ────────────────────────────────────────────────


class TestExportToPdfFile:

    def test_cria_arquivo_pdf(self, cifra_simples, tmp_path):
        """Deve criar o arquivo PDF no caminho especificado."""
        output = tmp_path / "cifra.pdf"
        resultado = export_to_pdf_file(cifra_simples, output, titulo="Teste")

        assert resultado == output
        assert output.exists()

    def test_arquivo_pdf_nao_vazio(self, cifra_simples, tmp_path):
        """Arquivo PDF criado não deve ser vazio."""
        output = tmp_path / "cifra.pdf"
        export_to_pdf_file(cifra_simples, output, titulo="Teste")

        assert output.stat().st_size > 0

    def test_retorna_caminho_correto(self, cifra_simples, tmp_path):
        """Deve retornar o caminho do arquivo criado."""
        output = tmp_path / "cifra.pdf"
        resultado = export_to_pdf_file(cifra_simples, output, titulo="Teste")

        assert isinstance(resultado, Path)
        assert resultado == output


# ── Testes: _build_styles ─────────────────────────────────────────────────────


class TestBuildStyles:

    def test_retorna_dicionario(self):
        """Deve retornar um dicionário de estilos."""
        estilos = _build_styles()
        assert isinstance(estilos, dict)

    def test_contem_estilos_obrigatorios(self):
        """Deve conter todos os estilos necessários."""
        estilos = _build_styles()
        esperados = ["app_name", "titulo", "artista", "metadata", "cifra", "rodape"]
        for estilo in esperados:
            assert estilo in estilos, f"Estilo '{estilo}' ausente"

    def test_cifra_usa_fonte_monospace(self):
        """Estilo da cifra deve usar fonte Courier (monospace)."""
        estilos = _build_styles()
        assert estilos["cifra"].fontName == "Courier"

    def test_titulo_usa_fonte_bold(self):
        """Estilo do título deve usar fonte bold."""
        estilos = _build_styles()
        assert "Bold" in estilos["titulo"].fontName


# ── Testes: _build_elements ───────────────────────────────────────────────────


class TestBuildElements:

    def test_retorna_lista_nao_vazia(self, cifra_simples):
        """Deve retornar uma lista de elementos não vazia."""
        elementos = _build_elements(cifra_simples, "Título", "Artista")
        assert isinstance(elementos, list)
        assert len(elementos) > 0

    def test_mais_elementos_com_artista(self, cifra_simples):
        """Com artista deve ter mais elementos que sem artista."""
        com_artista = _build_elements(cifra_simples, "Título", "Artista")
        sem_artista = _build_elements(cifra_simples, "Título", "")
        assert len(com_artista) > len(sem_artista)

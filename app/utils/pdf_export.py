"""
pdf_export.py
Geração de PDF da cifra melódica usando ReportLab.

Gera um PDF formatado com a cifra melódica, preservando
o alinhamento monospace entre acordes, melodia e letra,
com título, metadados e estilo visual limpo.
"""

import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    HRFlowable,
    Preformatted,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

# Configurações da página
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2 * cm

# Cores
COLOR_TITLE = colors.HexColor("#1A4B8C")
COLOR_SUBTITLE = colors.HexColor("#2E75B6")
COLOR_TEXT = colors.HexColor("#333333")
COLOR_DIVIDER = colors.HexColor("#DDDDDD")
COLOR_BACKGROUND = colors.HexColor("#F5F8FF")


# ── Função Principal ──────────────────────────────────────────────────────────


def export_to_pdf(
    cifra: str,
    titulo: str = "Cifra Melódica",
    artista: str = "",
) -> bytes:
    """
    Gera um PDF com a cifra melódica formatada.

    Args:
        cifra:   Texto da cifra a ser exportada.
        titulo:  Título da música.
        artista: Nome do artista (opcional).

    Returns:
        Bytes do PDF gerado.
    """
    logger.info("Gerando PDF para: %s", titulo)

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=titulo,
        author="MelodyTab AI",
    )

    elementos = _build_elements(cifra, titulo, artista)
    doc.build(elementos)

    pdf_bytes = buffer.getvalue()
    logger.info("PDF gerado com sucesso: %.1f KB", len(pdf_bytes) / 1024)

    return pdf_bytes


def export_to_pdf_file(
    cifra: str,
    output_path: Path,
    titulo: str = "Cifra Melódica",
    artista: str = "",
) -> Path:
    """
    Gera um PDF e salva em um arquivo.

    Args:
        cifra:       Texto da cifra a ser exportada.
        output_path: Caminho para o arquivo PDF de saída.
        titulo:      Título da música.
        artista:     Nome do artista (opcional).

    Returns:
        Caminho para o arquivo PDF gerado.
    """
    pdf_bytes = export_to_pdf(cifra, titulo, artista)

    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

    logger.info("PDF salvo em: %s", output_path)
    return output_path


# ── Construção do Documento ───────────────────────────────────────────────────


def _build_elements(
    cifra: str,
    titulo: str,
    artista: str,
) -> list:
    """
    Constrói a lista de elementos do PDF.

    Args:
        cifra:   Texto da cifra.
        titulo:  Título da música.
        artista: Nome do artista.

    Returns:
        Lista de elementos ReportLab para o documento.
    """
    estilos = _build_styles()
    elements = []

    # Cabeçalho
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph("🎸 MelodyTab AI", estilos["app_name"]))
    elements.append(Spacer(1, 0.3 * cm))

    # Título da música
    elements.append(Paragraph(titulo, estilos["titulo"]))

    # Artista (se fornecido)
    if artista:
        elements.append(Paragraph(artista, estilos["artista"]))

    # Metadados
    data_geracao = datetime.now().strftime("%d/%m/%Y às %H:%M")
    elements.append(Paragraph(f"Gerado em {data_geracao}", estilos["metadata"]))

    elements.append(Spacer(1, 0.3 * cm))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=COLOR_DIVIDER,
        )
    )
    elements.append(Spacer(1, 0.5 * cm))

    # Cifra em fonte monospace
    elements.append(Preformatted(cifra, estilos["cifra"]))

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=COLOR_DIVIDER,
        )
    )
    elements.append(Spacer(1, 0.3 * cm))

    # Rodapé
    elements.append(
        Paragraph(
            "Gerado automaticamente pelo MelodyTab AI",
            estilos["rodape"],
        )
    )

    return elements


def _build_styles() -> dict:
    """
    Define os estilos tipográficos do documento.

    Returns:
        Dicionário com os estilos ReportLab.
    """
    return {
        "app_name": ParagraphStyle(
            "app_name",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=COLOR_SUBTITLE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "titulo": ParagraphStyle(
            "titulo",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=COLOR_TITLE,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "artista": ParagraphStyle(
            "artista",
            fontName="Helvetica",
            fontSize=14,
            textColor=COLOR_SUBTITLE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "metadata": ParagraphStyle(
            "metadata",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "cifra": ParagraphStyle(
            "cifra",
            fontName="Courier",
            fontSize=11,
            textColor=COLOR_TEXT,
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=8,
        ),
        "rodape": ParagraphStyle(
            "rodape",
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
    }

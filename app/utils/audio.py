"""
audio.py
Funções auxiliares para manipulação de arquivos de áudio.

Fornece utilitários para validação, conversão e informações
sobre arquivos de áudio usados no pipeline do MelodyTab AI.
"""

import logging
import tempfile
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

# Formatos de áudio suportados para upload
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Tamanho máximo de arquivo em bytes (50 MB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Tamanho máximo para envio à API Groq (25 MB)
GROQ_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


# ── Validação ─────────────────────────────────────────────────────────────────


def validate_audio_file(path: Path) -> tuple[bool, str]:
    """
    Valida um arquivo de áudio verificando formato e tamanho.

    Args:
        path: Caminho para o arquivo de áudio.

    Returns:
        Tupla (válido, mensagem):
            válido:    True se o arquivo for válido.
            mensagem:  Mensagem de erro ou 'OK'.
    """
    if not path.exists():
        return False, f"Arquivo não encontrado: {path}"

    sufixo = path.suffix.lower()
    if sufixo not in SUPPORTED_FORMATS:
        return False, (
            f"Formato '{sufixo}' não suportado. "
            f"Formatos aceitos: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    tamanho = path.stat().st_size
    if tamanho == 0:
        return False, "Arquivo de áudio está vazio."

    if tamanho > MAX_FILE_SIZE_BYTES:
        return False, (
            f"Arquivo muito grande: {tamanho / 1024 / 1024:.1f} MB. "
            f"Limite máximo: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )

    logger.info(
        "Arquivo validado: %s (%.1f MB)",
        path.name,
        tamanho / 1024 / 1024,
    )
    return True, "OK"


# ── Informações do Áudio ──────────────────────────────────────────────────────


def get_audio_info(path: Path) -> dict:
    """
    Retorna informações técnicas de um arquivo de áudio.

    Args:
        path: Caminho para o arquivo de áudio.

    Returns:
        Dicionário com informações do áudio:
        {
            'duration':    duração em segundos,
            'sample_rate': taxa de amostragem em Hz,
            'channels':    número de canais,
            'format':      formato do arquivo,
            'size_mb':     tamanho em MB,
        }
    """
    try:
        info = sf.info(str(path))
        return {
            "duration": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": path.suffix.lower().lstrip("."),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        }
    except Exception as e:
        logger.warning("Não foi possível ler info do áudio %s: %s", path, e)
        return {
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "format": path.suffix.lower().lstrip("."),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        }


def format_duration(seconds: float) -> str:
    """
    Formata uma duração em segundos para string legível.

    Args:
        seconds: Duração em segundos.

    Returns:
        String formatada (ex: '3:45', '1:02:30').
    """
    segundos = int(seconds)
    minutos = segundos // 60
    horas = minutos // 60

    if horas > 0:
        return f"{horas}:{minutos % 60:02d}:{segundos % 60:02d}"
    return f"{minutos}:{segundos % 60:02d}"


# ── Conversão e Preparação ────────────────────────────────────────────────────


def save_uploaded_file(uploaded_file) -> Path:
    """
    Salva um arquivo enviado pelo Streamlit em um diretório temporário.

    Args:
        uploaded_file: Objeto UploadedFile do Streamlit.

    Returns:
        Caminho para o arquivo salvo no diretório temporário.
    """
    sufixo = Path(uploaded_file.name).suffix.lower()
    tmp_dir = Path(tempfile.mkdtemp(prefix="melodytab_upload_"))
    destino = tmp_dir / f"audio{sufixo}"

    with open(destino, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(
        "Arquivo salvo em: %s (%.1f MB)",
        destino,
        destino.stat().st_size / 1024 / 1024,
    )
    return destino


def needs_conversion_for_groq(path: Path) -> bool:
    """
    Verifica se o arquivo precisa ser convertido para envio à API Groq.

    A API Groq aceita MP3, MP4, MPEG, MPGA, M4A, WAV e WEBM.
    Arquivos FLAC precisam ser convertidos para WAV.

    Args:
        path: Caminho para o arquivo de áudio.

    Returns:
        True se o arquivo precisar de conversão.
    """
    formatos_groq = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    return path.suffix.lower() not in formatos_groq


def is_too_large_for_groq(path: Path) -> bool:
    """
    Verifica se o arquivo excede o limite de tamanho da API Groq (25 MB).

    Args:
        path: Caminho para o arquivo de áudio.

    Returns:
        True se o arquivo exceder o limite.
    """
    return path.stat().st_size > GROQ_MAX_FILE_SIZE_BYTES

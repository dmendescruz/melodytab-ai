"""
transcriber.py
Transcrição da letra com timestamps usando Groq API (Whisper-large-v3).

A faixa vocal isolada pelo Demucs é enviada para a API da Groq, que
utiliza o modelo Whisper-large-v3 para transcrever a letra com
timestamps precisos por palavra — essenciais para o alinhamento
com acordes e melodia.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# Tamanho máximo de arquivo aceito pela API da Groq em bytes (25 MB)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class WordEvent:
    """
    Representa uma palavra transcrita com seu timestamp.

    Atributos:
        word:       Texto da palavra.
        start_time: Tempo de início em segundos.
        end_time:   Tempo de fim em segundos.
        confidence: Confiança da transcrição entre 0.0 e 1.0.
    """

    word: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class TranscriptionResult:
    """
    Resultado completo da transcrição.

    Atributos:
        words:    Lista de palavras com timestamps.
        full_text: Texto completo transcrito.
        language: Idioma detectado pelo Whisper.
        success:  Indica se a transcrição foi bem-sucedida.
        error:    Mensagem de erro, se houver.
    """

    words: list[WordEvent]
    full_text: str
    language: str
    success: bool
    error: str | None = None


# ── Função Principal ──────────────────────────────────────────────────────────


def transcribe(vocals_path: Path) -> TranscriptionResult:
    """
    Transcreve a letra de uma faixa vocal usando Groq Whisper-large-v3.

    Args:
        vocals_path: Caminho para a faixa vocal isolada pelo Demucs.

    Returns:
        TranscriptionResult com palavras, timestamps e texto completo.
    """
    logger.info("Iniciando transcrição: %s", vocals_path)

    try:
        _validate_api_key()
        _validate_file(vocals_path)

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        language = os.getenv("WHISPER_LANGUAGE", "pt")

        logger.info("Enviando áudio para Groq API (Whisper-large-v3)...")
        logger.info("Idioma configurado: %s", language)

        with open(vocals_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language=language,
            )

        words = _parse_words(response)
        full_text = response.text.strip()
        detected = getattr(response, "language", language)

        logger.info("✅ Transcrição concluída: %d palavras detectadas.", len(words))
        logger.info("Idioma detectado: %s", detected)
        logger.info(
            "Texto: %s", full_text[:100] + "..." if len(full_text) > 100 else full_text
        )

        return TranscriptionResult(
            words=words,
            full_text=full_text,
            language=detected,
            success=True,
        )

    except Exception as e:
        logger.error("❌ Transcrição falhou: %s", str(e))
        return TranscriptionResult(
            words=[],
            full_text="",
            language="",
            success=False,
            error=str(e),
        )


# ── Parsing da Resposta ───────────────────────────────────────────────────────


def _parse_words(response) -> list[WordEvent]:
    """
    Extrai a lista de palavras com timestamps da resposta da API.

    A API retorna os timestamps no campo `words` da resposta
    verbose_json. Cada item contém word, start e end.

    Args:
        response: Objeto de resposta da API Groq.

    Returns:
        Lista de WordEvent com timestamps por palavra.
    """
    words = []

    raw_words = getattr(response, "words", None)
    if not raw_words:
        logger.warning(
            "API não retornou timestamps por palavra. "
            "Usando texto completo sem timestamps."
        )
        # Fallback: cria um único WordEvent com o texto completo
        words.append(
            WordEvent(
                word=response.text.strip(),
                start_time=0.0,
                end_time=0.0,
                confidence=1.0,
            )
        )
        return words

    for item in raw_words:
        word = getattr(item, "word", "").strip()
        start_time = float(getattr(item, "start", 0.0))
        end_time = float(getattr(item, "end", 0.0))

        if not word:
            continue

        words.append(
            WordEvent(
                word=word,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                confidence=1.0,  # Whisper não retorna confiança por palavra
            )
        )

    return words


# ── Validações ────────────────────────────────────────────────────────────────


def _validate_api_key() -> None:
    """
    Verifica se a chave da API Groq está configurada.

    Raises:
        RuntimeError: Se a chave não estiver configurada no .env.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY não configurada. " "Adicione sua chave ao arquivo .env."
        )


def _validate_file(path: Path) -> None:
    """
    Verifica se o arquivo de áudio existe e está dentro do limite de tamanho.

    Args:
        path: Caminho para o arquivo de áudio.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo exceder o limite de 25 MB da API.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    tamanho = path.stat().st_size
    if tamanho > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Arquivo muito grande: {tamanho / 1024 / 1024:.1f} MB. "
            f"Limite da API Groq: 25 MB."
        )

    logger.info(
        "Arquivo validado: %s (%.1f MB)",
        path.name,
        tamanho / 1024 / 1024,
    )

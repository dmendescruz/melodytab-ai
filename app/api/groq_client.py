"""
groq_client.py
Cliente centralizado para a Groq API.

Centraliza todas as chamadas para a Groq API em um único lugar,
incluindo Whisper para transcrição e LLaMA para formatação de texto.
Implementa retry automático com backoff exponencial para lidar com
erros temporários de rede e rate limiting.
"""

import logging
import time
from typing import Any

from groq import Groq, RateLimitError, APIConnectionError, APIStatusError
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

# Configurações de retry
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # segundos — dobra a cada tentativa (backoff exponencial)
MAX_DELAY = 30.0  # delay máximo em segundos


# ── Cliente Principal ─────────────────────────────────────────────────────────


class GroqClient:
    """
    Cliente centralizado para a Groq API.

    Gerencia autenticação, retry automático e logging de todas
    as chamadas para os modelos Whisper e LLaMA da Groq.

    Uso:
        client = GroqClient()
        resultado = client.transcribe(audio_path)
        cifra = client.complete(prompt)
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY não configurada. " "Adicione sua chave ao arquivo .env."
            )
        self._client = Groq(api_key=api_key)
        logger.info("GroqClient inicializado com sucesso.")

    def transcribe(
        self,
        audio_path: str,
        language: str = "pt",
        response_format: str = "verbose_json",
    ) -> Any:
        """
        Transcreve um arquivo de áudio usando Whisper-large-v3.

        Args:
            audio_path:      Caminho para o arquivo de áudio.
            language:        Código do idioma (ex: 'pt', 'en').
            response_format: Formato da resposta ('verbose_json' retorna timestamps).

        Returns:
            Objeto de resposta da API com texto e timestamps.

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        logger.info("Transcrevendo áudio: %s", audio_path)

        def _call():
            with open(audio_path, "rb") as f:
                return self._client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    response_format=response_format,
                    timestamp_granularities=["word"],
                    language=language,
                )

        return self._with_retry(_call, context="transcribe")

    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Gera uma completion de texto usando LLaMA 3.3 70B.

        Args:
            prompt:      Prompt de entrada para o modelo.
            temperature: Temperatura da geração (0.0 a 1.0).
            max_tokens:  Número máximo de tokens na resposta.

        Returns:
            Texto gerado pelo modelo.

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        logger.info("Gerando completion com LLaMA (%.0f tokens max).", max_tokens)

        def _call():
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        return self._with_retry(_call, context="complete")

    # ── Retry com Backoff Exponencial ─────────────────────────────────────────

    def _with_retry(self, func, context: str = "") -> Any:
        """
        Executa uma função com retry automático e backoff exponencial.

        Tenta novamente em caso de:
        - RateLimitError: API com muitas requisições
        - APIConnectionError: Falha de conexão temporária

        Não tenta novamente em caso de:
        - APIStatusError com status 4xx (erro do cliente)

        Args:
            func:    Função a ser executada.
            context: Nome da operação para logging.

        Returns:
            Resultado da função.

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        delay = RETRY_DELAY

        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                resultado = func()
                if tentativa > 1:
                    logger.info(
                        "✅ %s bem-sucedido na tentativa %d.", context, tentativa
                    )
                return resultado

            except RateLimitError:
                logger.warning(
                    "⚠️ Rate limit atingido (%s), tentativa %d/%d. "
                    "Aguardando %.1fs...",
                    context,
                    tentativa,
                    MAX_RETRIES,
                    delay,
                )

            except APIConnectionError:
                logger.warning(
                    "⚠️ Erro de conexão (%s), tentativa %d/%d. " "Aguardando %.1fs...",
                    context,
                    tentativa,
                    MAX_RETRIES,
                    delay,
                )

            except APIStatusError as e:
                # Erros 4xx não devem ser retentados
                if 400 <= e.status_code < 500:
                    logger.error(
                        "❌ Erro %d na Groq API (%s): %s",
                        e.status_code,
                        context,
                        str(e),
                    )
                    raise RuntimeError(
                        f"Groq API retornou erro {e.status_code}: {str(e)}"
                    )
                logger.warning(
                    "⚠️ Erro %d na Groq API (%s), tentativa %d/%d.",
                    e.status_code,
                    context,
                    tentativa,
                    MAX_RETRIES,
                )

            except Exception as e:
                logger.error("❌ Erro inesperado (%s): %s", context, str(e))
                raise

            if tentativa < MAX_RETRIES:
                time.sleep(min(delay, MAX_DELAY))
                delay *= 2  # backoff exponencial

        raise RuntimeError(
            f"Groq API falhou após {MAX_RETRIES} tentativas ({context})."
        )

"""
huggingface_client.py
Cliente centralizado para a HuggingFace Inference API.

Usado como fallback para detecção de acordes quando o
processamento local com librosa falha. Gerencia autenticação,
chamadas à API e retry automático.
"""

import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configurações de retry
MAX_RETRIES = 3
RETRY_DELAY = 3.0
MAX_DELAY = 30.0

# URL base da HuggingFace Inference API
HF_API_BASE = "https://api-inference.huggingface.co/models"

# Modelo para detecção de acordes
CHORD_MODEL = "christianlschubert/tiny-chord-recognition-model"

# Timeout para requisições em segundos
REQUEST_TIMEOUT = 60


# ── Cliente Principal ─────────────────────────────────────────────────────────


class HuggingFaceClient:
    """
    Cliente centralizado para a HuggingFace Inference API.

    Gerencia autenticação, chamadas à API e retry automático
    para modelos de análise musical.

    Uso:
        client = HuggingFaceClient()
        acordes = client.detect_chords(audio_path)
    """

    def __init__(self):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN não configurado. "
                "Adicione seu token ao arquivo .env."
            )
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }
        logger.info("HuggingFaceClient inicializado com sucesso.")

    def detect_chords(self, audio_path: Path) -> list[dict]:
        """
        Detecta acordes em um arquivo de áudio usando HuggingFace.

        Args:
            audio_path: Caminho para o arquivo de áudio.

        Returns:
            Lista de dicionários com acordes e timestamps:
            [{'label': 'Am', 'score': 0.95, 'start': 0.0, 'end': 1.5}, ...]

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        logger.info("Detectando acordes via HuggingFace: %s", audio_path)

        def _call():
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            url = f"{HF_API_BASE}/{CHORD_MODEL}"
            response = requests.post(
                url,
                headers=self._headers,
                data=audio_data,
                timeout=REQUEST_TIMEOUT,
            )

            # Modelo pode estar carregando — aguarda e tenta novamente
            if response.status_code == 503:
                estimated_time = response.json().get("estimated_time", 20)
                logger.info("Modelo carregando, aguardando %.0fs...", estimated_time)
                time.sleep(min(estimated_time, 30))
                raise RuntimeError("Modelo ainda carregando, tentando novamente...")

            response.raise_for_status()
            return response.json()

        resultado = self._with_retry(_call, context="detect_chords")
        logger.info("✅ %d acordes detectados via HuggingFace.", len(resultado))
        return resultado

    def is_available(self) -> bool:
        """
        Verifica se a API está disponível e o modelo está acessível.

        Returns:
            True se a API estiver disponível, False caso contrário.
        """
        try:
            url = f"{HF_API_BASE}/{CHORD_MODEL}"
            response = requests.get(
                url,
                headers=self._headers,
                timeout=10,
            )
            disponivel = response.status_code in (200, 503)
            logger.info(
                "HuggingFace API disponível: %s (status: %d)",
                disponivel,
                response.status_code,
            )
            return disponivel

        except Exception as e:
            logger.warning("HuggingFace API indisponível: %s", str(e))
            return False

    # ── Retry com Backoff Exponencial ─────────────────────────────────────────

    def _with_retry(self, func, context: str = ""):
        """
        Executa uma função com retry automático e backoff exponencial.

        Tenta novamente em caso de:
        - RuntimeError (modelo carregando)
        - requests.ConnectionError (falha de rede)
        - requests.Timeout (timeout da requisição)

        Não tenta novamente em caso de:
        - requests.HTTPError com status 4xx (erro do cliente)

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

            except requests.HTTPError as e:
                # Erros 4xx não devem ser retentados
                if e.response is not None and 400 <= e.response.status_code < 500:
                    logger.error(
                        "❌ Erro %d na HuggingFace API (%s): %s",
                        e.response.status_code,
                        context,
                        str(e),
                    )
                    raise RuntimeError(
                        f"HuggingFace API retornou erro "
                        f"{e.response.status_code}: {str(e)}"
                    )
                logger.warning(
                    "⚠️ Erro HTTP (%s), tentativa %d/%d: %s",
                    context,
                    tentativa,
                    MAX_RETRIES,
                    str(e),
                )

            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(
                    "⚠️ Erro de rede (%s), tentativa %d/%d: %s",
                    context,
                    tentativa,
                    MAX_RETRIES,
                    str(e),
                )

            except RuntimeError as e:
                logger.warning(
                    "⚠️ %s (%s), tentativa %d/%d.",
                    str(e),
                    context,
                    tentativa,
                    MAX_RETRIES,
                )

            except Exception as e:
                logger.error("❌ Erro inesperado (%s): %s", context, str(e))
                raise

            if tentativa < MAX_RETRIES:
                logger.info("Aguardando %.1fs antes da próxima tentativa...", delay)
                time.sleep(min(delay, MAX_DELAY))
                delay *= 2

        raise RuntimeError(
            f"HuggingFace API falhou após {MAX_RETRIES} tentativas ({context})."
        )

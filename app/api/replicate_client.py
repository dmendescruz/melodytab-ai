"""
replicate_client.py
Cliente centralizado para a Replicate API.

Usado como fallback para separação de instrumentos (Demucs)
quando o processamento local falha. Gerencia autenticação,
download das faixas geradas e retry automático.
"""

import logging
import os
import time
from pathlib import Path

import replicate
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configurações de retry
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # segundos — maior que Groq pois modelos demoram mais
MAX_DELAY = 60.0

# Versão pinada do modelo Demucs no Replicate
DEMUCS_VERSION = "25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953"
DEMUCS_MODEL = f"cjwbw/demucs:{DEMUCS_VERSION}"


# ── Cliente Principal ─────────────────────────────────────────────────────────


class ReplicateClient:
    """
    Cliente centralizado para a Replicate API.

    Gerencia autenticação, execução de modelos e download
    dos arquivos gerados, com retry automático.

    Uso:
        client = ReplicateClient()
        faixas = client.separate_stems(audio_path)
    """

    def __init__(self):
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN não configurado. "
                "Adicione seu token ao arquivo .env."
            )
        os.environ["REPLICATE_API_TOKEN"] = token
        logger.info("ReplicateClient inicializado com sucesso.")

    def separate_stems(
        self,
        audio_path: Path,
        model: str = "htdemucs",
    ) -> dict[str, Path]:
        """
        Separa os instrumentos de um arquivo de áudio usando Demucs.

        Args:
            audio_path: Caminho para o arquivo de áudio.
            model:      Modelo Demucs a usar (htdemucs ou htdemucs_6s).

        Returns:
            Dicionário com os caminhos das faixas baixadas:
            {'vocals': Path, 'bass': Path, 'drums': Path, 'other': Path}

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        logger.info(
            "Iniciando separação via Replicate (modelo: %s): %s",
            model,
            audio_path,
        )

        def _call():
            with open(audio_path, "rb") as f:
                output = replicate.run(
                    DEMUCS_MODEL,
                    input={"audio": f, "model": model},
                )
            return output

        output = self._with_retry(_call, context="separate_stems")

        # Valida e baixa as faixas retornadas
        return self._download_stems(output, audio_path.parent)

    def _download_stems(
        self,
        output: dict,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Baixa as faixas geradas pelo Replicate para o diretório local.

        Args:
            output:     Dicionário com URLs das faixas retornadas pela API.
            output_dir: Diretório onde as faixas serão salvas.

        Returns:
            Dicionário com os caminhos locais das faixas baixadas.

        Raises:
            RuntimeError: Se alguma URL estiver ausente ou download falhar.
        """
        faixas_esperadas = ["vocals", "bass", "drums", "other"]
        caminhos = {}

        for nome in faixas_esperadas:
            url = output.get(nome)
            if not url:
                raise RuntimeError(f"Replicate não retornou URL para a faixa '{nome}'")

            destino = output_dir / f"{nome}.wav"
            logger.info("Baixando faixa '%s'...", nome)
            _download_file(url, destino)
            caminhos[nome] = destino
            logger.info("Faixa '%s' salva em: %s", nome, destino)

        return caminhos

    # ── Retry com Backoff Exponencial ─────────────────────────────────────────

    def _with_retry(self, func, context: str = ""):
        """
        Executa uma função com retry automático e backoff exponencial.

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

            except replicate.exceptions.ReplicateError as e:
                logger.warning(
                    "⚠️ Erro Replicate (%s), tentativa %d/%d: %s",
                    context,
                    tentativa,
                    MAX_RETRIES,
                    str(e),
                )

            except Exception as e:
                logger.error("❌ Erro inesperado (%s): %s", context, str(e))
                raise

            if tentativa < MAX_RETRIES:
                logger.info("Aguardando %.1fs antes da próxima tentativa...", delay)
                time.sleep(min(delay, MAX_DELAY))
                delay *= 2

        raise RuntimeError(
            f"Replicate API falhou após {MAX_RETRIES} tentativas ({context})."
        )


# ── Utilitários ───────────────────────────────────────────────────────────────


def _download_file(url: str, destino: Path) -> None:
    """
    Faz o download de um arquivo de uma URL para um caminho local.

    Args:
        url:     URL do arquivo a baixar.
        destino: Caminho local onde o arquivo será salvo.

    Raises:
        requests.HTTPError: Se o download falhar.
    """
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(destino, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(
        "Download concluído: %s (%.1f KB)",
        destino.name,
        destino.stat().st_size / 1024,
    )

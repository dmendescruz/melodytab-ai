"""
separator.py
Separação de instrumentos usando Demucs local com fallback para Replicate API.

Modelos suportados:
    htdemucs    → 4 faixas: vocals, drums, bass, other (padrão, mais rápido)
    htdemucs_6s → 6 faixas: vocals, drums, bass, guitar, piano, other (mais preciso)

A faixa harmônica (usada para detecção de acordes) é selecionada automaticamente
com base nas faixas disponíveis no modelo escolhido.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import demucs.separate
import torch
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Constantes ────────────────────────────────────────────────────────────────

# Modelos com 4 faixas: vocals, drums, bass, other
MODELS_4_STEMS = {"htdemucs", "mdx", "mdx_extra", "mdx_q"}

# Modelos com 6 faixas: vocals, drums, bass, guitar, piano, other
MODELS_6_STEMS = {"htdemucs_6s"}

# Prioridade de seleção da faixa harmônica para detecção de acordes
# Guitarra > Piano > Other (qualquer outro instrumento melódico)
HARMONIC_PRIORITY = ["guitar", "piano", "other"]


# ── Enums e Dataclasses ───────────────────────────────────────────────────────


class SeparationMode(Enum):
    LOCAL = "local"
    REPLICATE = "replicate"


@dataclass
class SeparationResult:
    """
    Resultado da separação de instrumentos.

    A faixa `harmonic` representa o instrumento melódico principal
    disponível — pode ser guitarra, piano ou qualquer outro instrumento
    dependendo do modelo usado e do conteúdo da música.

    Atributos:
        vocals:       Faixa vocal isolada.
        harmonic:     Melhor faixa harmônica disponível para detecção de acordes.
        bass:         Faixa de baixo isolada.
        drums:        Faixa de bateria isolada.
        guitar:       Faixa de guitarra isolada (apenas htdemucs_6s).
        piano:        Faixa de piano isolada (apenas htdemucs_6s).
        other:        Faixa com demais instrumentos.
        model:        Nome do modelo Demucs utilizado.
        harmonic_source: Nome da faixa que originou `harmonic` (guitar/piano/other).
        mode:         Modo de execução (local ou Replicate).
        success:      Indica se a separação foi bem-sucedida.
        error:        Mensagem de erro, se houver.
    """

    vocals: Path
    harmonic: Path
    bass: Path
    drums: Path
    other: Path
    mode: SeparationMode
    success: bool
    model: str = "htdemucs"
    harmonic_source: str = "other"
    guitar: Path | None = None
    piano: Path | None = None
    error: str | None = None


# ── Função Principal ──────────────────────────────────────────────────────────


def separate(audio_path: Path) -> SeparationResult:
    """
    Ponto de entrada principal.
    Tenta separar localmente com Demucs, com fallback para Replicate API.

    Args:
        audio_path: Caminho para o arquivo de áudio original.

    Returns:
        SeparationResult com os caminhos das faixas separadas.
    """
    logger.info("Iniciando separação de instrumentos: %s", audio_path)

    # Tentativa 1: Demucs local
    try:
        logger.info("Tentando separação com Demucs local...")
        result = _separate_local(audio_path)
        logger.info("✅ Separação local concluída com sucesso.")
        logger.info(
            "Faixa harmônica selecionada: '%s' (modelo: %s)",
            result.harmonic_source,
            result.model,
        )
        return result

    except Exception as e:
        logger.warning("⚠️ Demucs local falhou: %s", str(e))
        logger.info("Tentando fallback para Replicate API...")

    # Tentativa 2: Replicate API
    try:
        result = _separate_replicate(audio_path)
        logger.info("✅ Separação via Replicate concluída com sucesso.")
        logger.info(
            "Faixa harmônica selecionada: '%s' (modelo: %s)",
            result.harmonic_source,
            result.model,
        )
        return result

    except Exception as e:
        logger.error("❌ Replicate API também falhou: %s", str(e))
        return SeparationResult(
            vocals=Path(),
            harmonic=Path(),
            bass=Path(),
            drums=Path(),
            other=Path(),
            mode=SeparationMode.LOCAL,
            success=False,
            error=str(e),
        )


# ── Separação Local (Demucs) ──────────────────────────────────────────────────


def _separate_local(audio_path: Path) -> SeparationResult:
    """
    Executa a separação usando Demucs localmente.

    Args:
        audio_path: Caminho para o arquivo de áudio.

    Returns:
        SeparationResult com os caminhos das faixas separadas.

    Raises:
        RuntimeError: Se a separação falhar ou arquivos não forem gerados.
    """
    model = os.getenv("DEMUCS_MODEL", "htdemucs")
    output_dir = Path(tempfile.mkdtemp(prefix="melodytab_"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Modelo: %s | Dispositivo: %s | Saída: %s", model, device, output_dir)
    logger.info("Isso pode levar alguns minutos em CPU...")

    demucs.separate.main(
        [
            "--name",
            model,
            "--out",
            str(output_dir),
            "--device",
            device,
            str(audio_path),
        ]
    )

    stem_dir = output_dir / model / audio_path.stem

    # Faixas presentes em todos os modelos
    vocals = stem_dir / "vocals.wav"
    bass = stem_dir / "bass.wav"
    drums = stem_dir / "drums.wav"
    other = stem_dir / "other.wav"

    # Faixas exclusivas do htdemucs_6s
    guitar = stem_dir / "guitar.wav" if model in MODELS_6_STEMS else None
    piano = stem_dir / "piano.wav" if model in MODELS_6_STEMS else None

    # Valida faixas obrigatórias
    for nome, caminho in [
        ("vocals", vocals),
        ("bass", bass),
        ("drums", drums),
        ("other", other),
    ]:
        if not caminho.exists():
            raise RuntimeError(f"Arquivo não encontrado para '{nome}': {caminho}")

    # Seleciona a melhor faixa harmônica disponível
    harmonic, harmonic_source = _select_harmonic(
        guitar=guitar,
        piano=piano,
        other=other,
    )

    return SeparationResult(
        vocals=vocals,
        harmonic=harmonic,
        bass=bass,
        drums=drums,
        other=other,
        guitar=guitar,
        piano=piano,
        mode=SeparationMode.LOCAL,
        success=True,
        model=model,
        harmonic_source=harmonic_source,
    )


# ── Fallback: Replicate API ───────────────────────────────────────────────────


def _separate_replicate(audio_path: Path) -> SeparationResult:
    """
    Executa a separação usando a API do Replicate como fallback.
    Utiliza sempre o modelo htdemucs (4 faixas) no fallback.

    Args:
        audio_path: Caminho para o arquivo de áudio.

    Returns:
        SeparationResult com os caminhos das faixas separadas.

    Raises:
        RuntimeError: Se o token não estiver configurado ou a API falhar.
    """
    import replicate

    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN não configurado no .env")

    logger.info("Enviando áudio para Replicate API...")

    with open(audio_path, "rb") as f:
        output = replicate.run(
            "cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953",
            input={"audio": f, "model": "htdemucs"},
        )

    # Replicate retorna URLs para cada faixa
    output_dir = Path(tempfile.mkdtemp(prefix="melodytab_replicate_"))

    urls = {
        "vocals": output.get("vocals"),
        "other": output.get("other"),
        "bass": output.get("bass"),
        "drums": output.get("drums"),
    }

    # Valida e baixa todas as faixas
    caminhos = {}
    for nome, url in urls.items():
        if not url:
            raise RuntimeError(f"Replicate não retornou URL para a faixa '{nome}'")
        destino = output_dir / f"{nome}.wav"
        logger.info("Baixando faixa '%s'...", nome)
        _download_file(url, destino)
        caminhos[nome] = destino

    # No fallback (htdemucs 4 faixas), harmonic vem direto de other
    harmonic, harmonic_source = _select_harmonic(
        guitar=None,
        piano=None,
        other=caminhos["other"],
    )

    return SeparationResult(
        vocals=caminhos["vocals"],
        harmonic=harmonic,
        bass=caminhos["bass"],
        drums=caminhos["drums"],
        other=caminhos["other"],
        guitar=None,
        piano=None,
        mode=SeparationMode.REPLICATE,
        success=True,
        model="htdemucs",
        harmonic_source=harmonic_source,
    )


# ── Seleção da Faixa Harmônica ────────────────────────────────────────────────


def _select_harmonic(
    guitar: Path | None,
    piano: Path | None,
    other: Path,
) -> tuple[Path, str]:
    """
    Seleciona a melhor faixa harmônica para detecção de acordes.

    Prioridade: guitarra → piano → other.
    Uma faixa é considerada válida se existir e tiver conteúdo de áudio.

    Args:
        guitar: Caminho para a faixa de guitarra (None se modelo 4 faixas).
        piano:  Caminho para a faixa de piano (None se modelo 4 faixas).
        other:  Caminho para a faixa other (sempre presente).

    Returns:
        Tupla (caminho_da_faixa, nome_da_fonte).
    """
    candidatas = [
        ("guitar", guitar),
        ("piano", piano),
        ("other", other),
    ]

    for nome, caminho in candidatas:
        if caminho and caminho.exists() and _has_audio_content(caminho):
            logger.info("Faixa harmônica selecionada: '%s'", nome)
            return caminho, nome

    # Se tudo falhar, usa other sem verificação
    logger.warning(
        "Nenhuma faixa harmônica com conteúdo encontrada. Usando 'other' como fallback."
    )
    return other, "other"


def _has_audio_content(path: Path, min_size_bytes: int = 1024) -> bool:
    """
    Verifica se um arquivo de áudio tem conteúdo relevante.
    Usa o tamanho do arquivo como heurística rápida.

    Args:
        path:           Caminho para o arquivo de áudio.
        min_size_bytes: Tamanho mínimo em bytes para considerar válido.

    Returns:
        True se o arquivo tem conteúdo, False caso contrário.
    """
    try:
        return path.stat().st_size > min_size_bytes
    except OSError:
        return False


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
    import requests

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(destino, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info("Download concluído: %s (%d bytes)", destino, destino.stat().st_size)

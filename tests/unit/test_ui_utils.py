"""
test_ui_utils.py
Testes unitários para logger.py, progress.py e chord_display.py
"""

import logging

import pytest

from app.components.chord_display import _count_chords, _sanitize_filename
from app.components.progress import (
    StepMode,
    StepStatus,
    get_pipeline_steps,
    update_step,
)
from app.utils.logger import StreamlitLogHandler


# ── Testes: StreamlitLogHandler ───────────────────────────────────────────────


class TestStreamlitLogHandler:

    @pytest.fixture
    def handler(self):
        """Retorna um StreamlitLogHandler limpo."""
        return StreamlitLogHandler()

    def test_acumula_mensagens(self, handler):
        """Deve acumular mensagens de log."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="mensagem teste",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        assert len(handler.get_messages()) == 1

    def test_filtra_por_nivel(self, handler):
        """Deve filtrar mensagens por nível."""
        for level, msg in [
            (logging.INFO, "info msg"),
            (logging.WARNING, "warning msg"),
            (logging.ERROR, "error msg"),
        ]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        assert len(handler.get_messages_by_level("INFO")) == 1
        assert len(handler.get_messages_by_level("WARNING")) == 1
        assert len(handler.get_messages_by_level("ERROR")) == 1

    def test_detecta_erros(self, handler):
        """Deve retornar True quando há mensagens de erro."""
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="erro grave",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        assert handler.has_errors() is True

    def test_sem_erros_retorna_false(self, handler):
        """Deve retornar False quando não há erros."""
        assert handler.has_errors() is False

    def test_clear_limpa_mensagens(self, handler):
        """Deve limpar todas as mensagens acumuladas."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="msg",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        handler.clear()
        assert len(handler.get_messages()) == 0


# ── Testes: get_pipeline_steps ────────────────────────────────────────────────


class TestGetPipelineSteps:

    def test_retorna_seis_etapas(self):
        """Deve retornar exatamente 6 etapas."""
        steps = get_pipeline_steps()
        assert len(steps) == 6

    def test_etapas_comecam_pendentes(self):
        """Todas as etapas devem começar com status PENDING."""
        steps = get_pipeline_steps()
        assert all(s.status == StepStatus.PENDING for s in steps)

    def test_etapas_tem_nomes_unicos(self):
        """Todas as etapas devem ter nomes únicos."""
        steps = get_pipeline_steps()
        nomes = [s.name for s in steps]
        assert len(nomes) == len(set(nomes))

    def test_ordem_das_etapas(self):
        """Etapas devem estar na ordem correta do pipeline."""
        steps = get_pipeline_steps()
        nomes = [s.name for s in steps]
        assert nomes == [
            "separator",
            "chords",
            "melody",
            "transcriber",
            "aligner",
            "formatter",
        ]


# ── Testes: update_step ───────────────────────────────────────────────────────


class TestUpdateStep:

    def test_atualiza_status(self):
        """Deve atualizar o status da etapa correta."""
        steps = get_pipeline_steps()
        steps = update_step(steps, "chords", StepStatus.RUNNING)

        chord_step = next(s for s in steps if s.name == "chords")
        assert chord_step.status == StepStatus.RUNNING

    def test_atualiza_mensagem(self):
        """Deve atualizar a mensagem da etapa."""
        steps = get_pipeline_steps()
        steps = update_step(
            steps, "chords", StepStatus.SUCCESS, message="10 acordes detectados"
        )

        chord_step = next(s for s in steps if s.name == "chords")
        assert chord_step.message == "10 acordes detectados"

    def test_atualiza_modo(self):
        """Deve atualizar o modo de execução da etapa."""
        steps = get_pipeline_steps()
        steps = update_step(steps, "separator", StepStatus.SUCCESS, mode=StepMode.LOCAL)

        sep_step = next(s for s in steps if s.name == "separator")
        assert sep_step.mode == StepMode.LOCAL

    def test_acumula_historico_de_mensagens(self):
        """Deve acumular mensagens no histórico."""
        steps = get_pipeline_steps()
        steps = update_step(steps, "melody", StepStatus.RUNNING, message="iniciando...")
        steps = update_step(steps, "melody", StepStatus.SUCCESS, message="concluído!")

        melody_step = next(s for s in steps if s.name == "melody")
        assert len(melody_step.messages) == 2
        assert "iniciando..." in melody_step.messages
        assert "concluído!" in melody_step.messages

    def test_nao_altera_outras_etapas(self):
        """Atualizar uma etapa não deve alterar as demais."""
        steps = get_pipeline_steps()
        steps = update_step(steps, "chords", StepStatus.SUCCESS)

        outras = [s for s in steps if s.name != "chords"]
        assert all(s.status == StepStatus.PENDING for s in outras)


# ── Testes: _count_chords ─────────────────────────────────────────────────────


class TestCountChords:

    def test_conta_acordes_simples(self):
        """Deve contar acordes em colchetes corretamente."""
        cifra = "[C] quando o sol [Am] se pôr\n[F] estarei [G] aqui"
        assert _count_chords(cifra) == 4

    def test_conta_acordes_unicos(self):
        """Deve contar apenas acordes únicos."""
        cifra = "[C] verso um\n[C] verso dois\n[Am] refrão"
        assert _count_chords(cifra) == 2

    def test_cifra_sem_acordes(self):
        """Deve retornar 0 para cifra sem acordes."""
        assert _count_chords("apenas letra sem acordes") == 0

    def test_conta_acordes_com_sustenido(self):
        """Deve contar acordes com sustenido."""
        cifra = "[C#] nota [F#m] menor"
        assert _count_chords(cifra) == 2

    def test_conta_acordes_com_setima(self):
        """Deve contar acordes com sétima."""
        cifra = "[G7] acorde [C7] com sétima"
        assert _count_chords(cifra) == 2


# ── Testes: _sanitize_filename ────────────────────────────────────────────────


class TestSanitizeFilename:

    def test_remove_caracteres_invalidos(self):
        """Deve remover caracteres inválidos para nome de arquivo."""
        assert _sanitize_filename('musica<>:"/\\|?*') == "musica"

    def test_substitui_espacos_por_underscore(self):
        """Deve substituir espaços por underscores."""
        assert _sanitize_filename("minha musica") == "minha_musica"

    def test_nome_valido_nao_muda(self):
        """Nome já válido não deve ser alterado."""
        assert _sanitize_filename("musica_legal") == "musica_legal"

    def test_nome_vazio_retorna_cifra(self):
        """Nome vazio deve retornar 'cifra' como padrão."""
        assert _sanitize_filename("") == "cifra"

    def test_apenas_caracteres_invalidos_retorna_cifra(self):
        """Nome com apenas caracteres inválidos deve retornar 'cifra'."""
        assert _sanitize_filename('<>:"/\\|?*') == "cifra"

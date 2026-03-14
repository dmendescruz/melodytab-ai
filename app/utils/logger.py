"""
logger.py
Configuração centralizada de logging para o MelodyTab AI.

Configura o logging da aplicação com formatação consistente,
níveis adequados para desenvolvimento e produção, e handlers
para console e arquivo de log.
"""

import logging
import os

# Nível de log padrão — pode ser sobrescrito pela variável LOG_LEVEL no .env
DEFAULT_LOG_LEVEL = "INFO"

# Formato das mensagens de log
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging() -> None:
    """
    Configura o sistema de logging da aplicação.

    Define nível, formato e handlers para console.
    Deve ser chamado uma única vez na inicialização do app.
    """
    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    # Remove handlers existentes para evitar duplicação
    root = logging.getLogger()
    root.handlers.clear()

    # Handler de console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root.setLevel(level)
    root.addHandler(console_handler)

    # Silencia bibliotecas muito verbosas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configurado: nível=%s", level_name)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado para o módulo especificado.

    Args:
        name: Nome do módulo (geralmente __name__).

    Returns:
        Logger configurado.
    """
    return logging.getLogger(name)


class StreamlitLogHandler(logging.Handler):
    """
    Handler de logging que acumula mensagens para exibição no Streamlit.

    Permite capturar logs do pipeline em tempo real e exibi-los
    na interface do usuário durante o processamento.

    Uso:
        handler = StreamlitLogHandler()
        logging.getLogger().addHandler(handler)

        # Durante o processamento:
        for msg in handler.get_messages():
            st.write(msg)
    """

    def __init__(self):
        super().__init__()
        self._messages: list[dict] = []
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        """Captura um registro de log e armazena na lista interna."""
        self._messages.append(
            {
                "level": record.levelname,
                "message": self.format(record),
                "name": record.name,
            }
        )

    def get_messages(self) -> list[dict]:
        """Retorna todas as mensagens capturadas."""
        return self._messages.copy()

    def get_messages_by_level(self, level: str) -> list[dict]:
        """
        Retorna mensagens filtradas por nível.

        Args:
            level: Nível de log ('INFO', 'WARNING', 'ERROR').

        Returns:
            Lista de mensagens do nível especificado.
        """
        return [m for m in self._messages if m["level"] == level.upper()]

    def clear(self) -> None:
        """Limpa todas as mensagens acumuladas."""
        self._messages.clear()

    def has_errors(self) -> bool:
        """Retorna True se houver mensagens de erro."""
        return any(m["level"] == "ERROR" for m in self._messages)

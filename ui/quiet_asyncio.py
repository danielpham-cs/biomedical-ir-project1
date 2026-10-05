"""Hide a harmless Windows asyncio log line.

When a browser tab closes or reloads, the Proactor event loop on Windows logs
"Exception in callback _ProactorBasePipeTransport._call_connection_lost" with
ConnectionResetError [WinError 10054]. The app is unaffected; only the terminal
gets noisy. This filter drops exactly that record and leaves every other error.
"""
import logging


class _ClosedConnectionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        message = record.getMessage()
        harmless = isinstance(error, (ConnectionResetError, ConnectionAbortedError)) and '_call_connection_lost' in message
        return not harmless


def install() -> None:
    """Idempotent: Streamlit re-runs scripts, but the filter is added to the asyncio logger once."""
    logger = logging.getLogger('asyncio')
    if not any(isinstance(f, _ClosedConnectionFilter) for f in logger.filters):
        logger.addFilter(_ClosedConnectionFilter())

"""The asyncio log filter hides only the harmless Windows closed-connection record."""
import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.quiet_asyncio import _ClosedConnectionFilter, install


def record(message, error):
    exc_info = (type(error), error, None) if error else None
    return logging.LogRecord('asyncio', logging.ERROR, __file__, 1, message, None, exc_info)


class QuietAsyncioTests(unittest.TestCase):
    def test_drops_only_closed_connection_noise(self):
        keep = _ClosedConnectionFilter().filter
        noise = 'Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)'
        self.assertFalse(keep(record(noise, ConnectionResetError(10054, 'forcibly closed'))))
        self.assertTrue(keep(record(noise, ValueError('real bug'))))            # other errors still shown
        self.assertTrue(keep(record('Task exception was never retrieved', ConnectionResetError())))
        self.assertTrue(keep(record('plain message', None)))

    def test_install_is_idempotent(self):
        install()
        install()
        filters = [f for f in logging.getLogger('asyncio').filters if isinstance(f, _ClosedConnectionFilter)]
        self.assertEqual(len(filters), 1)


if __name__ == '__main__':
    unittest.main()

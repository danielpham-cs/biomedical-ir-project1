"""Download latency/error policy and identity-safe local XML reuse."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pubmed import (EFETCH, ID_CONVERTER, PubMedError, _http_session,
                        _request_ncbi, load_saved_article, save_article)
from test_pubmed import PUBMED, parse_pubmed_xml


def response(status=200, *, headers=None):
    return Mock(status_code=status, headers=headers or {}, content=PUBMED)


@patch("src.pubmed.time.sleep")
@patch("src.pubmed.requests.Session.get")
class DownloadRequestTests(unittest.TestCase):
    def test_permanent_http_errors_are_not_retried(self, get, sleep):
        for code in (400, 401, 403, 404):
            get.reset_mock()
            reply = response(code)
            get.return_value = reply
            with self.subTest(code=code), self.assertRaisesRegex(PubMedError, f"HTTP {code}"):
                _request_ncbi(EFETCH, {"db": "pubmed", "id": "12345"})
            self.assertEqual(get.call_count, 1)
            reply.close.assert_called_once()

    def test_short_retry_after_and_transient_recovery(self, get, sleep):
        replies = [response(503, headers={"Retry-After": "2"}), response()]
        get.side_effect = replies
        self.assertEqual(_request_ncbi(EFETCH, {"db": "pubmed", "id": "12345"}), PUBMED)
        self.assertEqual(get.call_count, 2)
        sleep.assert_any_call(2)
        self.assertTrue(all(reply.close.call_count == 1 for reply in replies))

    def test_long_retry_after_does_not_leave_ui_sleeping(self, get, sleep):
        get.return_value = response(429, headers={"Retry-After": "120"})
        with self.assertRaisesRegex(PubMedError, "120 seconds"):
            _request_ncbi(EFETCH, {"db": "pubmed", "id": "12345"})
        get.assert_called_once()
        self.assertTrue(all(call.args[0] <= 0.35 for call in sleep.call_args_list))

    def test_read_timeouts_are_shorter_for_interactive_lookups(self, get, sleep):
        get.return_value = response()
        for endpoint, params, expected in (
            (EFETCH, {"db": "pubmed", "id": "12345"}, (5, 12)),
            (ID_CONVERTER, {"ids": "PMC67890"}, (5, 12)),
            (EFETCH, {"db": "pmc", "id": "67890"}, (5, 30)),
            (EFETCH, {"db": "pubmed", "id": "12345,56789"}, (5, 30)),
        ):
            with self.subTest(params=params):
                _request_ncbi(endpoint, params)
                self.assertEqual(get.call_args.kwargs["timeout"], expected)

    def test_timeout_retry_count_is_bounded_and_error_identifies_stage(self, get, sleep):
        get.side_effect = requests.Timeout()
        with self.assertRaisesRegex(PubMedError, "PMC ID conversion: NCBI timed out after 2 attempts"):
            _request_ncbi(ID_CONVERTER, {"ids": "PMC67890"})
        self.assertEqual(get.call_count, 2)

    def test_tls_errors_fail_immediately_without_disabling_verification(self, get, sleep):
        get.side_effect = requests.exceptions.SSLError()
        with self.assertRaisesRegex(PubMedError, "verification failed"):
            _request_ncbi(EFETCH, {"db": "pubmed", "id": "12345"})
        get.assert_called_once()
        self.assertNotIn("verify", get.call_args.kwargs)

    def test_sessions_are_reused_and_keys_only_go_to_eutilities(self, get, sleep):
        self.assertIs(_http_session(), _http_session())
        get.return_value = response()
        with patch.dict("os.environ", {"NCBI_API_KEY": "test-only-key"}):
            _request_ncbi("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "pubmed"})
            self.assertEqual(get.call_args.kwargs["params"]["api_key"], "test-only-key")
            _request_ncbi(ID_CONVERTER, {"ids": "PMC67890"})
            self.assertNotIn("api_key", get.call_args.kwargs["params"])


class SavedDownloadTests(unittest.TestCase):
    @patch("src.pubmed.requests.Session.get")
    def test_pmid_and_pmcid_reuse_saved_xml_without_network(self, get):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_article(parse_pubmed_xml(PUBMED), root)
            for identifier in ("12345", "PMID: 12345", "PMC67890"):
                with self.subTest(identifier=identifier):
                    article = load_saved_article(identifier, root)
                    self.assertEqual(article.pmid, "12345")
                    self.assertEqual(article.xml, PUBMED)
            get.assert_not_called()

    def test_missing_corrupt_and_wrong_id_files_do_not_return_wrong_paper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(load_saved_article("12345", root))
            for xml in (b"not xml", PUBMED.replace(b"12345", b"99999")):
                (root / "PMID12345.xml").write_bytes(xml)
                self.assertIsNone(load_saved_article("12345", root))
                self.assertIsNone(load_saved_article("PMC67890", root))
            with self.assertRaises(ValueError):
                load_saved_article("../../12345", root)


if __name__ == "__main__":
    unittest.main()

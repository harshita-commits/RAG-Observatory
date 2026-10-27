import io
import os
import unittest
from unittest.mock import patch
from observatory.__main__ import main


class ConsoleTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows console regression")
    def test_cli_allows_unicode_mlflow_run_links_with_legacy_pipe_encoding(self):
        buffer = io.BytesIO()
        output = io.TextIOWrapper(buffer, encoding="cp1252")
        try:
            with patch("sys.stdout", output), patch("sys.argv", ["observatory", "--help"]):
                with self.assertRaises(SystemExit) as exit_info:
                    main()
                self.assertEqual(exit_info.exception.code, 0)
                output.write("\U0001f3c3 View run at: http://127.0.0.1:5000\n")
                output.flush()
                self.assertIn("\U0001f3c3 View run", buffer.getvalue().decode("utf-8"))
        finally:
            output.close()


if __name__ == "__main__": unittest.main()

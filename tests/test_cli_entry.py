"""Regression check for the module entry point, without running training."""
import subprocess
import sys
import unittest


class TestModuleEntry(unittest.TestCase):
    def test_module_help(self):
        result = subprocess.run([sys.executable, "-m", "riskselrul", "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("train", "audit", "deploy"):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()

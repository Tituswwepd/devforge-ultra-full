# tools/tests.py
def suggest_tests_for_code(language: str, code: str) -> str:
    language = (language or "").lower()
    if language == "python":
        return f'''# tests/test_basic.py
import unittest

class TestGenerated(unittest.TestCase):
    def test_import(self):
        import generated  # rename to your module
        self.assertTrue(True)

    def test_sanity(self):
        self.assertEqual(2+2, 4)

if __name__ == "__main__":
    unittest.main()
'''
    # fallback generic
    return "// TODO: add unit tests for this code"

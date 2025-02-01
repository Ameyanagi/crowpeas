# tests/test_core.py
import unittest
from src.crowpeas.core import main

class TestCore(unittest.TestCase):
    def test_main(self):
        main()

if __name__ == '__main__':
    unittest.main()
import sys
import unittest
sys.path.append("src/")
from main import main

class MainArgs:
    def __init__(self, attrDict):
        self.__dict__ = attrDict

class MainTests(unittest.TestCase):
    mainArgs = MainArgs({"show": False})

    def test_main(self):
        try:
            ret = main(self.mainArgs)
        except Exception as e:
            ret = e
        self.assertIsNone(ret)

if __name__ == "__main__":
    unittest.main()

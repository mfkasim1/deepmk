import unittest

def main():
    test_modules = [
        "deepmk.criteria.test.iou_test",
    ]
    suite = unittest.TestSuite()
    for t in test_modules:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    # run the test
    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    main()

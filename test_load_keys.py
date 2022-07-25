import unittest
import load_keys


class MyTestCase(unittest.TestCase):
	def test_thesaurus(self):
		got_dict = load_keys.load_keys("test_thesaurus.xlsx")
		expected_dict = {
			"a": {"a1": {"a11": {}, "a12": {}}, "a2": {"a21": {}, "a22": {}, "a23": {}}},
			"b": {"b1": {"b11": {}, "b12": {}}, "b2": {}, "b3": {}}}
		self.assertEqual(got_dict,expected_dict)  # add assertion here


if __name__ == '__main__':
	unittest.main()

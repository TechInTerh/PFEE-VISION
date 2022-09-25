import unittest
from graph_to_tree import graph_to_tree


class MyTestCase(unittest.TestCase):
	def test_simple(self):
		d_in = {
			"A": {"B": 0.8},
			"B": {"A": 0.8},
			"C": {"D": 0.9, "E": 0.7},
			"D": {"C": 0.9, "E": 0.8},
			"E": {"C": 0.7, "D": 0.8},
		}
		expected = {
			"A": {"B,0.8": {}},
			"C": {"D,0.9": {}, "E,0.7": {}}
		}
		got = graph_to_tree(d_in)
		
		self.assertEqual(expected, got)


if __name__ == '__main__':
	unittest.main()

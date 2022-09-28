import unittest
from graph_to_tree import dict_to_graph
import graphviz as gv


class MyTestCase(unittest.TestCase):
	def test_simple(self):
		d_in = {
			"A": {"B": 0.8},
			"B": {"A": 0.8},
			"C": {"D": 0.9, "E": 0.7},
			"D": {"C": 0.9, "E": 0.8},
			"E": {"C": 0.7, "D": 0.8},
		}
		got: gv.Digraph = dict_to_graph(d_in)
		got.render("test.png", view=True, format="png")
	
	# self.assertEqual(expected, got)
	
	def test_empty(self):
		d_in = {
			"A": {},
			"B": {"C": 0.8},
			"C": {"B": 0.8},
		}
		expected = {"B": {"C,0.8": {}}}
		got = graph_to_tree(d_in)
		
		self.assertEqual(expected, got)
	
	def test_multiple(self):
		d_in = {
			"A": {"B": 0.8},
			"B": {"A": 0.8, "C": 0.7},
			"C": {"B": 0.7},
		}
		expected = {
			"B": {"B,0.8": {}, "C,0.9": {}}
		}
		got = graph_to_tree(d_in)
		
		self.assertEqual(expected, got)


if __name__ == '__main__':
	unittest.main()

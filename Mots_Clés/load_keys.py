# function that load excel file
import json
import sys

import pandas as pd


def load_excel(file_name):
	df = pd.read_excel(file_name)
	return df


def get_keys(line: str):
	return line.split("\\")[1:]


def list_to_dict(l: list):
	ret_json = {}
	if len(l) > 1:
		current_head = l[0][0]
		ret_json[current_head] = []
		for body in l[1:]:
			if len(body) == 1:
				current_head = body[0]
				ret_json[current_head] = []
			else:
				ret_json[current_head].append(body[1:])
		apply_change(ret_json)
	return ret_json


def apply_change(d: dict):
	for i in d:
		if type(d[i]) is list:
			d[i] = list_to_dict(d[i])
	return d


def load_keys(file_name: str) -> dict:
	df = load_excel(file_name)
	
	res = df.drop_duplicates().applymap(get_keys)
	tmp_json = res.to_dict()
	d = tmp_json.get("MotsClés")
	ret_json = {}
	for i in d:
		val = d[i]
		if len(val) > 1:
			if val[0] not in ret_json:
				ret_json[val[0]] = []
			ret_json[val[0]].append(val[1:])
	return apply_change(ret_json)


if __name__ == '__main__':
	new_dict = load_keys(sys.argv[1])
	with open(sys.argv[2], "w") as f:
		json.dump(new_dict, f, indent=4, sort_keys=False,ensure_ascii=False)

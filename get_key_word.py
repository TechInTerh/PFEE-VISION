import pandas as pd
import sys
from tqdm import tqdm


def delete_quote(l: [str]):
	for i, s in enumerate(l):  # type: str
		l[i] = s.replace('"', "").replace(" ", "")
	return l


def split_word(text: str):
	return text.split(";")


def get_key_word(file_input: str, n: int = 100, file_out=None):
	tqdm.pandas()
	print("Reading Excel...")
	df = pd.read_excel(file_input)
	print("Droping...")
	df: pd.DataFrame = df.drop(["Unnamed: 3", "Unnamed: 4"], axis=1).astype("str")
	df_key_word: pd.DataFrame = df["MOTCLE"].progress_apply(split_word).map(delete_quote).explode().value_counts().head(
		n)
	if file_out is not None:
		print("Writing into", file_out)
		df_key_word.to_json(file_out)
	return df_key_word


if __name__ == '__main__':
	file_input = sys.argv[1]
	if len(sys.argv) > 2:
		n = int(sys.argv[2])
	else:
		n = 100
	if len(sys.argv) > 3:
		file_out = sys.argv[3]
	else:
		file_out = None
	print(get_key_word(file_input, n, file_out=file_out))

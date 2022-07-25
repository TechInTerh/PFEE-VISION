from google_images_search import GoogleImagesSearch
from tqdm import tqdm
from load_keys import load_keys

def clean_dir(path: str):
	import os
	import shutil
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


if __name__ == '__main__':
	gis = GoogleImagesSearch(developer_key="AIzaSyB2ZD01hQseKjLb1Smp-k4VAOkZLN-19ow",
	                         custom_search_cx="76710f952443c49ed")
	_search_params = {
		'imgType': 'photo',
		"fileType": "jpg",
		'num': 10,
		#'rights': "cc_publicdomain",
		'imgSize': "medium",
		"imgColorType": "gray"
	}
	path = "./images/"
	clean_dir(path)
	list_keys = load_keys("export_thesaurus_new.xlsx")
	list_keys = list_keys["Objet"]
	annotations = {}
	for key in list_keys:
		_search_params['q'] = key
		gis.search(_search_params, path_to_dir=path)
		for image in tqdm(gis.results()):
			image.download(path)

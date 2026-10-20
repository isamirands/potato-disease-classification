from pathlib import Path
from campis.datasets.mappings import load_crops,class_id
def test_ids_stable():
 c=load_crops(Path(__file__).parents[1]/"configs/crops.yaml"); assert class_id(c,"tomato","healthy")==0 and class_id(c,"rice","bacterial_leaf_blight")==3

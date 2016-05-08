import prepare_features as pf
import dirs
from dataset import DataSet

#features = pf.get_image_features("/Users/rmencis/RUG/Handwriting_Recognition/char_boxes_processed/KNMP/a/KNMP-VIII_F_69______2C2O_0094_1116_1696_1152_1743_a.png")

#print(pf.get_classes(dirs.KNMP_CHAR_BOXES_DIR_PATH))

dataset = DataSet(dirs.KNMP_CHAR_BOXES_DIR_PATH)

#print(dataset.get_all_item_count())
#print(dataset.get_train_item_count())
#print(dataset.get_test_item_count())
dataset.prepare_next_batch(10)

print(dataset.get_batch_one_hot_labels())
import os

# Top dir
BASE_DIR_PATH = "/Users/rmencis/RUG/Handwriting_Recognition"

# Labels (.words files)
LABELS_DIR_PATH = os.path.join(BASE_DIR_PATH,"code/ZeroAccuracySystems/labels")

# Original page images (jpg)
PAGES_DIR_PATH = os.path.join(BASE_DIR_PATH,"pages")
STANFORD_PAGES_DIR_PATH = os.path.join(PAGES_DIR_PATH,"Stanford")
KNMP_PAGES_DIR_PATH = os.path.join(PAGES_DIR_PATH,"KNMP")

# Cut out character boxes
CHAR_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH,"char_boxes")
STANFORD_CHAR_BOXES_DIR_PATH = os.path.join(CHAR_BOXES_DIR_PATH,"Stanford")
KNMP_CHAR_BOXES_DIR_PATH = os.path.join(CHAR_BOXES_DIR_PATH,"KNMP")

# Cut out word boxes
WORD_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH,"word_boxes")
STANFORD_WORD_BOXES_DIR_PATH = os.path.join(WORD_BOXES_DIR_PATH,"Stanford")
KNMP_WORD_BOXES_DIR_PATH = os.path.join(WORD_BOXES_DIR_PATH,"KNMP")

# Word boxes manually cut out from pages (with labels)
ADDITIONAL_WORD_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH,"additional_word_boxes")
STANFORD_ADDITIONAL_WORD_BOXES_DIR_PATH = os.path.join(ADDITIONAL_WORD_BOXES_DIR_PATH,"Stanford")
KNMP_ADDITIONAL_WORD_BOXES_DIR_PATH = os.path.join(ADDITIONAL_WORD_BOXES_DIR_PATH,"KNMP")

# Processed word boxes
PROCESSED_WORD_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH,"word_boxes_processed")
STANFORD_PROCESSED_WORD_BOXES_DIR_PATH = os.path.join(PROCESSED_WORD_BOXES_DIR_PATH,"Stanford")
KNMP_PROCESSED_WORD_BOXES_DIR_PATH = os.path.join(PROCESSED_WORD_BOXES_DIR_PATH,"KNMP")

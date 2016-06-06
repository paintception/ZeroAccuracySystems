import os

# Top dir
GIT_DIR_PATH = os.path.dirname(os.path.join(os.path.dirname(__file__), "../"))
BASE_DIR_PATH = GIT_DIR_PATH
DROPBOX_DIR_PATH = os.path.join(BASE_DIR_PATH, "HWR_Share")


# Labels (.words files)
LABELS_DIR_PATH = os.path.join(GIT_DIR_PATH, "labels")

# Original page images (jpg)
PAGES_DIR_PATH = os.path.join(BASE_DIR_PATH, "data", "pages")
STANFORD_PAGES_DIR_PATH = os.path.join(PAGES_DIR_PATH, "Stanford")
KNMP_PAGES_DIR_PATH = os.path.join(PAGES_DIR_PATH, "KNMP")

# Cut out character boxes
CHAR_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH, "char_boxes")
STANFORD_CHAR_BOXES_DIR_PATH = os.path.join(CHAR_BOXES_DIR_PATH, "Stanford")
KNMP_CHAR_BOXES_DIR_PATH = os.path.join(CHAR_BOXES_DIR_PATH, "KNMP")

# Cut out word boxes
WORD_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH, "word_boxes")
STANFORD_WORD_BOXES_DIR_PATH = os.path.join(WORD_BOXES_DIR_PATH, "Stanford")
KNMP_WORD_BOXES_DIR_PATH = os.path.join(WORD_BOXES_DIR_PATH, "KNMP")

# Word boxes manually cut out from pages (with labels)
ADDITIONAL_WORD_BOXES_DIR_PATH = os.path.join(DROPBOX_DIR_PATH, "additional_word_labels")
STANFORD_ADDITIONAL_WORD_BOXES_DIR_PATH = os.path.join(ADDITIONAL_WORD_BOXES_DIR_PATH, "Stanford")
KNMP_ADDITIONAL_WORD_BOXES_DIR_PATH = os.path.join(ADDITIONAL_WORD_BOXES_DIR_PATH, "KNMP")

# Processed word boxes
PROCESSED_WORD_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH, "word_boxes_processed")
STANFORD_PROCESSED_WORD_BOXES_DIR_PATH = os.path.join(PROCESSED_WORD_BOXES_DIR_PATH, "Stanford")
KNMP_PROCESSED_WORD_BOXES_DIR_PATH = os.path.join(PROCESSED_WORD_BOXES_DIR_PATH, "KNMP")

# Processed char boxes
PROCESSED_CHAR_BOXES_DIR_PATH = os.path.join(BASE_DIR_PATH,"char_boxes_processed")
STANFORD_PROCESSED_CHAR_BOXES_DIR_PATH = os.path.join(PROCESSED_CHAR_BOXES_DIR_PATH,"Stanford")
KNMP_PROCESSED_CHAR_BOXES_DIR_PATH = os.path.join(PROCESSED_CHAR_BOXES_DIR_PATH,"KNMP")

# Saved models
MODEL_DIR_PATH = os.path.join(BASE_DIR_PATH,"models")
STANFORD_MODEL_DIR_PATH = os.path.join(MODEL_DIR_PATH,"Stanford")
KNMP_MODEL_DIR_PATH = os.path.join(MODEL_DIR_PATH,"KNMP")



import matplotlib.pyplot as plt
import numpy as np


def plot_words_with_labels(images, target_labels, predicted_labels):
    fig, splots = plt.subplots(int(round(len(images) / 8)), 8)
    for spl, img, tl, pl in zip(splots.flat, images, target_labels, predicted_labels):
        spl.set_title(tl)
        spl.imshow(np.transpose(img), cmap=plt.get_cmap('Greys_r'))
        spl.get_xaxis().set_ticks([])
        spl.get_yaxis().set_ticks([])
        spl.set_xlabel(pl)

    plt.show()


if __name__ == "__main__":
    import blstm_ctc_net.word_dataset_with_timesteps as wd
    import dirs

    word_dataset = wd.WordDataSet(dirs.STANFORD_PROCESSED_WORD_BOXES_DIR_PATH)
    plot_words_with_labels(word_dataset.get_test_data()[:17], word_dataset.get_test_labels()[:17], word_dataset.get_test_labels()[:17])

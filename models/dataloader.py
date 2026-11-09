import os
import numpy as np
import cv2
import nibabel as nib
from skimage import transform


def read_data(image_size=(50, 75, 1), debug=False):
    # CHANGED: strictly AD vs CO only. The original script merged CO and YA
    # together as class 0 (age-matched controls + young adults treated as
    # the same "HC" class), which folds in an age confound. It also had no
    # branch for AQ files at all, so any AQ folder crashed the whole run via
    # `raise _file` (itself broken -- Python 3 can't raise a plain string,
    # which is why you saw `TypeError: exceptions must derive from
    # BaseException` instead of a clear error). This version explicitly
    # SKIPS both YA and AQ files (not included in the dataset at all),
    # keeping only true AD/CO -- matching what the paper's methods section
    # actually describes (14 AD vs 14 age-matched HC, YA excluded).
    folder_list = [item[0] for item in os.walk('.')]
    folder_list = folder_list[1:]

    octs = []
    oct_label = []
    label_counter = 0
    oct_size = (image_size[0], image_size[1], 5, image_size[2])
    skipped_ya = 0
    skipped_aq = 0
    skipped_other = 0

    for _folder in folder_list:
        file_list = os.listdir(os.path.join(os.getcwd(), _folder))
        for _file in file_list:
            if _file[-4:] == '.img':
                # CHANGED: check class membership BEFORE loading/resizing the
                # volume, so skipped files don't waste time on nibabel/resize.
                if _file.find('YA') != -1:
                    skipped_ya += 1
                    continue
                elif _file.find('AQ') != -1:
                    skipped_aq += 1
                    continue
                elif _file.find('CO') != -1:
                    label = 0
                elif _file.find('AD') != -1:
                    label = 1
                else:
                    # CHANGED: the raw data folder also contains non-subject
                    # utility/calibration files (e.g.
                    # "raw_DARK_vertical_above_horizontal.img") that don't
                    # carry any AD/CO/YA/AQ code at all. These aren't a
                    # different class -- they're not patient data -- so skip
                    # and count them rather than treat as an error.
                    skipped_other += 1
                    continue

                oct_image = nib.load(os.path.join(os.getcwd(), _folder, _file)).get_fdata()
                oct_image /= np.max(oct_image)
                oct_image_standard_size = transform.resize(oct_image, oct_size, order=1, preserve_range=True)

                octs.append(oct_image_standard_size)
                oct_label.append(label)
                label_counter += 1

        if debug and label_counter > 1:
            break

    print(f"Skipped {skipped_ya} YA files, {skipped_aq} AQ files, and {skipped_other} "
          f"non-subject/calibration files (AD-vs-CO only run).")

    images = []
    labels = []
    for i, d in enumerate(octs):
        for j in range(5):
            images.append(d[:, :, j, :])
            labels.append(oct_label[i])

    print("Fraction of AD images:", sum(oct_label) / len(oct_label))

    return np.asarray(images), np.asarray(labels)


if __name__ == '__main__':
    x, y = read_data(image_size=(224, 224, 1), debug=False)
    os.makedirs('./induced_data/hc', exist_ok=True)
    os.makedirs('./induced_data/ad', exist_ok=True)
    for i in range(x.shape[0]):
        if y[i] == 0:
            cv2.imwrite('./induced_data/hc/data_' + str(i) + '.png', 255 * x[i])
        else:
            cv2.imwrite('./induced_data/ad/data_' + str(i) + '.png', 255 * x[i])

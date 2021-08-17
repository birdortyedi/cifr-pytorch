from configs.default import get_cfg_defaults

import os
import cv2
import numpy as np

if __name__ == '__main__':
    cfg = get_cfg_defaults()

    filtered = cv2.imread(os.path.join(cfg.DATASET.ROOT, "1", "1_1977.jpg"))
    org = cv2.imread(os.path.join(cfg.DATASET.ROOT, "1", "1_Original.jpg"))

    print(filtered.shape, org.shape)

    loc1 = np.array([540, 210])
    loc2 = np.array([120, 140])
    loc3 = np.array([40, 580])
    loc4 = np.array([820, 662])
    loc5 = np.array([740, 104])
    loc6 = np.array([510, 712])

    crop1_f = filtered[loc1[1]:loc1[1]+128, loc1[0]:loc1[0]+128]
    crop1_o = org[loc1[1]:loc1[1] + 128, loc1[0]:loc1[0] + 128]
    crop2 = filtered[loc2[1]:loc2[1] + 128, loc2[0]:loc2[0] + 128]
    crop3 = filtered[loc3[1]:loc3[1]+128, loc3[0]:loc3[0]+128]
    crop4 = filtered[loc4[1]:loc4[1] + 128, loc4[0]:loc4[0] + 128]
    crop5 = filtered[loc5[1]:loc5[1] + 128, loc5[0]:loc5[0] + 128]
    crop6_f = filtered[loc6[1]:loc6[1] + 128, loc6[0]:loc6[0] + 128]
    crop6_o = org[loc6[1]:loc6[1] + 128, loc6[0]:loc6[0] + 128]

    filtered = cv2.rectangle(filtered, tuple(loc1), tuple(loc1+128), color=(255, 0, 0), thickness=8)
    org = cv2.rectangle(org, tuple(loc1), tuple(loc1 + 128), color=(255, 0, 0), thickness=8)
    filtered = cv2.rectangle(filtered, tuple(loc2), tuple(loc2 + 128), color=(0, 0, 255), thickness=8)
    filtered = cv2.rectangle(filtered, tuple(loc3), tuple(loc3 + 128), color=(0, 0, 255), thickness=8)
    filtered = cv2.rectangle(filtered, tuple(loc4), tuple(loc4 + 128), color=(0, 0, 255), thickness=8)
    filtered = cv2.rectangle(filtered, tuple(loc5), tuple(loc5 + 128), color=(0, 0, 255), thickness=8)
    filtered = cv2.rectangle(filtered, tuple(loc6), tuple(loc6 + 128), color=(0, 255, 0), thickness=8)
    org = cv2.rectangle(org, tuple(loc6), tuple(loc6 + 128), color=(0, 255, 0), thickness=8)

    cv2.imwrite("../../Downloads/filtered.jpg", filtered)
    cv2.imwrite("../../Downloads/org.jpg", org)
    # cv2.imwrite("../../Downloads/crop_org.jpg", crop1_o)
    # cv2.imwrite("../../Downloads/crop_filtered.jpg", crop1_f)
    # cv2.imwrite("../../Downloads/crop2_filtered.jpg", crop2)
    # cv2.imwrite("../../Downloads/crop3_filtered.jpg", crop3)
    # cv2.imwrite("../../Downloads/crop4_filtered.jpg", crop4)
    # cv2.imwrite("../../Downloads/crop5_filtered.jpg", crop5)
    # cv2.imwrite("../../Downloads/crop6_filtered.jpg", crop6_f)
    # cv2.imwrite("../../Downloads/crop6_org.jpg", crop6_o)
from typing import List
import random

import cv2
import numpy as np
import matplotlib.colors as mplc


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr

color_proposals = [hex_to_bgr(colour) for colour in mplc.CSS4_COLORS.values()]


def draw_binary_masks(image: np.ndarray, masks: List[np.ndarray], write_numbers=True):
    image_ = image.copy()
    for idx, mask in enumerate(masks):
        colour = random.choice(color_proposals)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue

        cv2.drawContours(image_, contours, -1, colour, 2)

        if write_numbers:
            font_scale = min(image.shape[0], image.shape[1]) / 1000

            number = str(idx)

            text_size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x, text_y = cx - text_size[0] // 2, cy + text_size[1] // 2

            # cv2.rectangle(image_, (text_x - 5, text_y - text_size[1] - 5), 
            #             (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

            text_bg = image_.copy()
            cv2.rectangle(text_bg, (text_x - 5, text_y - text_size[1] - 5), 
                        (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                        
            cv2.addWeighted(text_bg, 0.5, image_, 0.5, 0, image_)
                        
            cv2.putText(image_, number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image_
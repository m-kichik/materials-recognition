import sys
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import webcolors


IGNORE_TOKEN = 1000
sys.setrecursionlimit(100000000)

default_colours_map = {
    "aliceblue": "pale blue",
    "antiquewhite": "off-white",
    "cyan": "bright blue",
    "aquamarine": "blue-green",
    "azure": "light blue",
    "beige": "light tan",
    "bisque": "pale orange",
    "black": "black",
    "blanchedalmond": "creamy beige",
    "blue": "blue",
    "blueviolet": "purple-blue",
    "brown": "brown",
    "burlywood": "light brown",
    "cadetblue": "greyish blue",
    "chartreuse": "yellow-green",
    "chocolate": "rich brown",
    "coral": "pink-orange",
    "cornflowerblue": "soft blue",
    "cornsilk": "pale yellow",
    "crimson": "deep red",
    "darkblue": "navy blue",
    "darkcyan": "deep teal",
    "darkgoldenrod": "dark yellow-brown",
    "darkgrey": "charcoal",
    "darkgreen": "deep green",
    "darkkhaki": "dull yellow-green",
    "darkmagenta": "deep purple",
    "darkolivegreen": "deep olive",
    "darkorange": "burnt orange",
    "darkorchid": "deep violet",
    "darkred": "maroon",
    "darksalmon": "muted salmon",
    "darkseagreen": "muted green",
    "darkslateblue": "greyish blue",
    "darkslategrey": "deep grey",
    "darkturquoise": "deep cyan",
    "darkviolet": "deep purple",
    "deeppink": "hot pink",
    "deepskyblue": "bright blue",
    "dimgrey": "muted grey",
    "dodgerblue": "vivid blue",
    "firebrick": "reddish brown",
    "floralwhite": "off-white",
    "forestgreen": "rich green",
    "magenta": "bright pink",
    "gainsboro": "light grey",
    "ghostwhite": "very pale blue",
    "gold": "golden yellow",
    "goldenrod": "warm yellow",
    "grey": "grey",
    "green": "green",
    "greenyellow": "lime yellow",
    "honeydew": "pale green",
    "hotpink": "vibrant pink",
    "indianred": "muted red",
    "indigo": "deep blue-purple",
    "ivory": "creamy white",
    "khaki": "dusty yellow",
    "lavender": "soft purple",
    "lavenderblush": "pinkish lavender",
    "lawngreen": "bright green",
    "lemonchiffon": "pale yellow",
    "lightblue": "soft blue",
    "lightcoral": "soft red",
    "lightcyan": "pale cyan",
    "lightgoldenrodyellow": "soft yellow",
    "lightgrey": "pale grey",
    "lightgreen": "soft green",
    "lightpink": "pale pink",
    "lightsalmon": "soft salmon",
    "lightseagreen": "soft teal",
    "lightskyblue": "pastel blue",
    "lightslategrey": "soft grey-blue",
    "lightsteelblue": "soft bluish grey",
    "lightyellow": "pale yellow",
    "lime": "neon green",
    "limegreen": "bright green",
    "linen": "off-white",
    "maroon": "dark red",
    "mediumaquamarine": "moderate blue-green",
    "mediumblue": "true blue",
    "mediumorchid": "muted violet",
    "mediumpurple": "moderate purple",
    "mediumseagreen": "balanced green",
    "mediumslateblue": "blueish purple",
    "mediumspringgreen": "bright teal-green",
    "mediumturquoise": "muted turquoise",
    "mediumvioletred": "rich pink-purple",
    "midnightblue": "deep navy",
    "mintcream": "pale mint",
    "mistyrose": "soft pink",
    "moccasin": "pale orange-tan",
    "navajowhite": "warm beige",
    "navy": "dark blue",
    "oldlace": "creamy white",
    "olive": "dark yellow-green",
    "olivedrab": "muted olive",
    "orange": "orange",
    "orangered": "red-orange",
    "orchid": "soft purple-pink",
    "palegoldenrod": "muted yellow",
    "palegreen": "soft green",
    "paleturquoise": "light blue-green",
    "palevioletred": "soft red-purple",
    "papayawhip": "light peach",
    "peachpuff": "soft peach",
    "peru": "earthy brown",
    "pink": "pink",
    "plum": "muted purple",
    "powderblue": "dusty blue",
    "purple": "purple",
    "red": "red",
    "rosybrown": "pinkish brown",
    "royalblue": "deep blue",
    "saddlebrown": "rich brown",
    "salmon": "pink-orange",
    "sandybrown": "warm tan",
    "seagreen": "teal green",
    "seashell": "pale pinkish white",
    "sienna": "reddish brown",
    "silver": "metallic grey",
    "skyblue": "light blue",
    "slateblue": "greyish blue",
    "slategrey": "bluish grey",
    "snow": "very pale white",
    "springgreen": "vivid green",
    "steelblue": "cool blue-grey",
    "tan": "warm beige",
    "teal": "blue-green",
    "thistle": "soft purple",
    "tomato": "reddish orange",
    "turquoise": "bright blue-green",
    "violet": "purple",
    "wheat": "pale beige",
    "white": "white",
    "whitesmoke": "off-white",
    "yellow": "yellow",
    "yellowgreen": "greenish yellow",
}


def find_colours(
    img: np.ndarray, mask: np.ndarray = None, max_distance: int | float = 10
) -> List[Tuple[np.ndarray, float]]:
    """
    Simple BFS for defining colours in the given region. Note that the colour encoding does
    not change during processing.

    Args:
        img (np.ndarray): image of shape HxWxC, typical PIL Image converted to numpy array;
        mask (np.ndarray): mask of shape HxW, numpy array of bool type;
        max_distance (int | float): maximum euclidean distance colours of the neighboring
            pixels that allows to associate them with the similar colour domain.

    Returns:
        List[Tuple[np.ndarray, float]]: list of colours and corresponding areas for the region
            of the image inside the mask.
    """
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    h, w, c = img.shape

    masked_img = img.copy().astype(np.int16)
    start_i, end_i = 0, h
    start_j, end_j = 0, w

    if mask is not None:
        masked_img[np.logical_not(mask)] = [IGNORE_TOKEN] * c

        i_indices, j_indices = np.where(mask)
        start_i, end_i = i_indices.min(), i_indices.max()
        start_j, end_j = j_indices.min(), j_indices.max()

        mask_area = np.sum(mask)
    else:
        mask_area = h * w

    def recursive_step(i, j, prev_colour, domain_colours):
        if i < 0 or j < 0 or i >= h or j >= w:
            return

        if masked_img[i, j, 0] == IGNORE_TOKEN:
            return

        colour = masked_img[i, j, :].copy()

        if np.linalg.norm(prev_colour - colour) > max_distance:
            return

        domain_colours.append(colour)
        masked_img[i, j, :] = [IGNORE_TOKEN] * c

        recursive_step(i - 1, j, colour, domain_colours)
        recursive_step(i, j - 1, colour, domain_colours)
        recursive_step(i + 1, j, colour, domain_colours)
        recursive_step(i, j + 1, colour, domain_colours)

    colours = []
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            if masked_img[i, j, 0] != IGNORE_TOKEN:
                domain_colours = []
                recursive_step(i, j, masked_img[i, j, :], domain_colours)

                domain_colours = np.array(domain_colours)

                colour = np.round(np.mean(domain_colours, axis=0)).astype(np.int16)
                colours.append((colour, float(domain_colours.shape[0] / mask_area)))

    return colours


def name_colour(colour: np.ndarray, palette: Dict[Tuple[int], str] = None) -> str:
    """
    Returns the closest (by l2 norm) colour name for the given colour using the given palette.

    Args:
        colour (np.ndarray): RGB colour;
        palette (Dict[np.ndarray, str]): dictionary with RGB colours keys and their names.

    Returns:
        str: name of the given colour.
    """
    if palette is None:
        palette = {
            tuple(webcolors.name_to_rgb(name)): name for name in webcolors.names("css3")
        }

        palette = {k: default_colours_map[v] for k, v in palette.items()}

    idx = np.argmin(
        list(
            map(
                lambda x: np.linalg.norm(colour.astype(np.int16) - np.array(x)),
                list(palette.keys()),
            )
        )
    )
    return palette[list(palette.keys())[idx]]


def main():
    imname = "/home/m-kichik/Downloads/Flag_of_Russia.jpg"
    img = np.array(Image.open(imname))

    mask = np.zeros(img.shape[:-1], dtype=np.bool)
    mask[-10:, -10:] = True

    colours = find_colours(img, mask)

    for colour, _ in colours:
        print(name_colour(colour))  # , palette))


if __name__ == "__main__":
    main()

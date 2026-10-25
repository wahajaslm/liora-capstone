"""Readable names for the 27 product type codes.

Rakuten does not publish these names. We inferred each one by reading sample
product titles from the class, so treat them as labels for plots only.
"""

CATEGORY_NAMES = {
    10: "Used Books",
    40: "Games",
    50: "Game Accessories",
    60: "Consoles",
    1140: "Figurines",
    1160: "Cards",
    1180: "Wargames",
    1280: "Toys",
    1281: "Board Games",
    1300: "RC Models",
    1301: "Baby Socks",
    1302: "Outdoor Toys",
    1320: "Baby",
    1560: "Furniture",
    1920: "Textiles",
    1940: "Food",
    2060: "Decor",
    2220: "Pets",
    2280: "Magazines",
    2403: "Book Lots",
    2462: "Used Games",
    2522: "Stationery",
    2582: "Garden Furniture",
    2583: "Pool",
    2585: "Garden Tools",
    2705: "New Books",
    2905: "PC Games",
}


def category_label(code):
    """Return 'Name (code)' for a product type code."""
    return f"{CATEGORY_NAMES.get(code, 'Unknown')} ({code})"

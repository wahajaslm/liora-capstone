"""Readable names for the 27 product type codes.

Rakuten does not publish these names. We inferred each one by reading sample
product titles from the class, so treat them as labels for plots only.
"""

CATEGORY_NAMES = {
    10: "Used Books",
    40: "Video Games",
    50: "Gaming Accessories",
    60: "Retro / Handheld Consoles",
    1140: "Figurines",
    1160: "Trading Cards",
    1180: "Tabletop / Wargame Figures",
    1280: "Kids Toys",
    1281: "Board Games / Kids Activities",
    1300: "Model Kits / RC / Drones",
    1301: "Baby Socks / Bar Games",
    1302: "Outdoor Toys",
    1320: "Baby Care",
    1560: "Furniture",
    1920: "Home Textiles",
    1940: "Groceries",
    2060: "Home Decor",
    2220: "Pet Supplies",
    2280: "Magazines / Press",
    2403: "Book & Magazine Lots",
    2462: "Used Games & Console Bundles",
    2522: "Stationery",
    2582: "Garden Furniture",
    2583: "Pool Accessories",
    2585: "Tools & Garden",
    2705: "New Books",
    2905: "PC Games (Download)",
}


def category_label(code):
    """Return 'Name (code)' for a product type code."""
    return f"{CATEGORY_NAMES.get(code, 'Unknown')} ({code})"

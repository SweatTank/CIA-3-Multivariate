"""Approximate CS:GO (2017-era) in-game buy prices, keyed by the `wp` column of mm_master_demos.csv.

Used only for the Loadout Value KPI. These are NOT in the Kaggle dataset; they are a hand-entered
lookup of public in-game prices and are approximate (a few weapons changed price across patches).
Weapons absent from this table (grenades, Bomb, Knife, Unknown) are excluded from the KPI.
"""

WEAPON_PRICES = {
    # Pistols
    "Glock": 200, "USP": 200, "P2000": 200, "P250": 300, "DualBarettas": 400,
    "Tec9": 500, "FiveSeven": 500, "CZ": 500, "Deagle": 700,
    # SMGs
    "Mac10": 1050, "MP9": 1250, "UMP": 1200, "Bizon": 1400, "MP7": 1500, "P90": 2350,
    # Rifles
    "Scout": 1700, "Gallil": 2000, "Famas": 2050, "AK47": 2700, "M4A1": 3100,
    "M4A4": 3100, "SG556": 3000, "AUG": 3300, "AWP": 4750, "G3SG1": 5000, "Scar20": 5000,
    # Heavy
    "Nova": 1200, "SawedOff": 1200, "Swag7": 1800, "XM1014": 2000, "M249": 5200, "Negev": 5700,
    # Equipment
    "Zeus": 200,
}

"""
Consumer-friendly spending taxonomy + semantic anchors.

The dataset's `category` column is just the verbose ISO-18245 MCC description
(396 distinct strings like "Record Shops" or "Non-durable Goods, Not Elsewhere
Classified"). That is the *broken baseline* Revolut describes: too granular,
occasionally wrong, useless for budgeting. This module defines the clean ~16
category taxonomy users actually understand, plus a rich natural-language
"anchor" per category. The embedding tier (engine/embed.py) embeds each MCC
description and assigns it to the nearest anchor by cosine similarity — so the
432 MCCs map into the taxonomy *semantically* rather than via a hand-maintained
lookup table, and the same mechanism generalises to MCCs we have never seen.
"""

# Canonical category -> anchor sentence describing what belongs in it.
# Anchors are written densely so the sentence-transformer has strong signal.
TAXONOMY: dict[str, str] = {
    "Groceries":
        "Supermarkets, grocery stores, convenience stores, food markets, "
        "bakeries, butchers, greengrocers and everyday food and drink shopping "
        "for the home.",
    "Restaurants & Takeaway":
        "Restaurants, eating places, fast food, snack bars, cafes, bistros, "
        "diners, food delivery, caterers, bars, pubs and drinking places.",
    "Transport":
        "Local transport: trains, trams, buses, metro, commuter rail, taxis, "
        "ride hailing, fuel and petrol stations, parking, tolls, car washes "
        "and bicycle shops.",
    "Travel":
        "Travel and holidays: airlines, flights, hotels, lodging, resorts, "
        "car rental, travel agencies, tour operators, cruise lines and tourist "
        "attractions.",
    "Shopping":
        "Retail shopping: clothing, fashion, shoes, electronics, department "
        "stores, furniture, homeware, books, toys, gifts, sporting goods and "
        "general merchandise.",
    "Bills & Utilities":
        "Recurring bills and utilities: electricity, gas, water, telecoms, "
        "mobile, internet, cable and pay television, insurance premiums.",
    "Entertainment & Digital":
        "Entertainment and digital goods: streaming media, music, movies, "
        "games, apps, software, digital subscriptions, cinemas, theatres, "
        "concerts and recreation.",
    "Health & Pharmacy":
        "Health and medical: pharmacies, drug stores, doctors, dentists, "
        "opticians, hospitals, clinics and medical services.",
    "Personal Care":
        "Personal care and beauty: hairdressers, barbers, salons, spas, "
        "cosmetics and health and beauty shops.",
    "Cash & ATM":
        "Cash withdrawals from ATMs and cash machines, currency exchange and "
        "money orders.",
    "Fees & Charges":
        "Bank and card fees, service charges, account charges and interest.",
    "Income":
        "Money received: salary, wages, transfers in, top ups, credits and "
        "deposits into the account.",
    "Refunds & Reversals":
        "Refunds, returns, chargebacks and reversed transactions where money "
        "comes back to the account.",
    "Gambling":
        "Betting, lottery, casinos, gaming, wagering and gambling.",
    "Services":
        "Professional and personal services: consulting, advertising, legal, "
        "accounting, education, schools, government, charities, repairs and "
        "business services not elsewhere classified.",
    "Other":
        "Miscellaneous or unknown purchases that do not fit any other "
        "category.",
}

CATEGORIES: list[str] = list(TAXONOMY.keys())

# ---------------------------------------------------------------------------
# Rule layer 1: transaction TYPE -> category (for the ~3.3% of rows that are
# not merchant purchases and carry no MCC: fees, ATM, credits, chargebacks).
# These are deterministic and high-confidence.
# ---------------------------------------------------------------------------
TYPE_TO_CATEGORY: dict[str, str] = {
    "FEE": "Fees & Charges",
    "CHARGE": "Fees & Charges",
    "ATM": "Cash & ATM",
    "CARD_CREDIT": "Income",
    "CARD_REFUND": "Refunds & Reversals",
    "CARD_CHARGEBACK": "Refunds & Reversals",
    # CARD_PAYMENT is intentionally absent -> falls through to merchant logic.
}

# ---------------------------------------------------------------------------
# Rule layer 2: merchant-NAME head words -> category.
#
# The synthetic merchant names use a controlled Dutch grammar. ~11% carry an
# informative "type word" (Snackbar, Supermarkt, Tankstation...) that pins the
# category with 68-100% purity regardless of MCC. We use these for two things:
#   (a) to fill the category when the MCC is Unknown, and
#   (b) to cross-check the MCC mapping and surface likely miscategorisations
#       (the "supermarket tagged as shopping" scenario from the brief).
# Keys are lowercase substrings matched against the merchant name.
# ---------------------------------------------------------------------------
# Only high-precision head words are kept. Generic tokens that the synthetic
# grammar reuses across unrelated MCCs (e.g. "markt", "bar", "garage") were
# deliberately removed — they produced false correction flags (a generic
# "Zilveren Markt" on a Digital-Goods MCC is not a miscategorised grocer).
NAME_KEYWORDS: dict[str, str] = {
    # Groceries
    "supermarkt": "Groceries", "kruidenier": "Groceries",
    "bakkerij": "Groceries", "slagerij": "Groceries", "versmarkt": "Groceries",
    "buurtwinkel": "Groceries", "delicatessen": "Groceries",
    # Restaurants & Takeaway
    "eetcafe": "Restaurants & Takeaway", "restaurant": "Restaurants & Takeaway",
    "snackbar": "Restaurants & Takeaway", "frituur": "Restaurants & Takeaway",
    "pizzeria": "Restaurants & Takeaway", "bistro": "Restaurants & Takeaway",
    "grillroom": "Restaurants & Takeaway",
    "broodjeszaak": "Restaurants & Takeaway",
    # Transport
    "tankstation": "Transport", "spoorvervoer": "Transport",
    "arriva": "Transport", "parkeer": "Transport", "taxi": "Transport",
    "benzine": "Transport",
    # Travel
    "hotel": "Travel", "reisbureau": "Travel", "airlines": "Travel",
    "luchtvaart": "Travel", "camping": "Travel",
    # Health & Pharmacy
    "apotheek": "Health & Pharmacy", "drogist": "Health & Pharmacy",
    "tandarts": "Health & Pharmacy", "huisarts": "Health & Pharmacy",
    "opticien": "Health & Pharmacy",
    # Personal Care
    "kapper": "Personal Care", "kapsalon": "Personal Care",
    "schoonheid": "Personal Care",
    # Shopping
    "kledingwinkel": "Shopping", "schoenen": "Shopping",
    "boekhandel": "Shopping", "elektronica": "Shopping",
    "warenhuis": "Shopping", "speelgoed": "Shopping",
    # Entertainment & Digital
    ".app": "Entertainment & Digital", "bioscoop": "Entertainment & Digital",
    "theater": "Entertainment & Digital",
}

# ---------------------------------------------------------------------------
# Rule layer 0: explicit ISO-18245 MCC pins.
#
# The embedding tier maps ~90% of MCC descriptions correctly, but a handful of
# high-volume codes have descriptions whose *wording* misleads cosine
# similarity — e.g. "Service Stations (with or without ancillary services)"
# (petrol) is pulled to "Services" by the repeated word "service", and "Car
# Rental Companies" lands in Services rather than Travel. These codes are
# unambiguous in the ISO standard, so we pin them deterministically and let the
# embedding tier handle everything else. This is the standard "pin the known
# critical cases, embed the long tail" pattern.
# ---------------------------------------------------------------------------
MCC_OVERRIDES: dict[str, str] = {
    # Fuel / transport (wording trap: "...services")
    "5541": "Transport", "5542": "Transport",
    # Travel (wording trap: "...services"/"companies")
    "7512": "Travel", "4722": "Travel", "7011": "Travel",
    "4511": "Travel", "3246": "Travel", "3000": "Travel",
    # Health & pharmacy
    "5912": "Health & Pharmacy", "5122": "Health & Pharmacy",
    "8011": "Health & Pharmacy", "8021": "Health & Pharmacy",
    "8043": "Health & Pharmacy", "8062": "Health & Pharmacy",
    "8099": "Health & Pharmacy",
    # Personal care
    "7230": "Personal Care", "7298": "Personal Care",
    # Bills & utilities (telecom / cable / insurance)
    "4814": "Bills & Utilities", "4812": "Bills & Utilities",
    "4816": "Bills & Utilities", "4899": "Bills & Utilities",
    "4900": "Bills & Utilities", "6300": "Bills & Utilities",
    # Cash / money movement
    "6010": "Cash & ATM", "6011": "Cash & ATM", "6051": "Cash & ATM",
    "4829": "Cash & ATM",
}

# MCC codes that mean "no usable category" -> route to name/LLM fallback.
UNKNOWN_MCCS = {"0000", "0763", "9999", "", None}

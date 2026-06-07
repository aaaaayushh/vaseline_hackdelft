"""
Taxonomy + deterministic MCC -> PFC map (Spending IQ categorization engine, Component 1).

We collapse the Plaid Personal Finance Categories (PFC) scheme into a 12-category
budgeting taxonomy (see DESIGN.md s4.1). The MCC -> PFC map below is the deterministic
*prior* used by the cascade (Layer 1 feature) and the *label* source for capability
evaluation (on synthetic data MCC is the ground truth -- DESIGN.md s8).

Design choices worth knowing:
  * MCCs follow ISO 18245 ranges. We map by specific override first, then by range.
  * The "digital domain" MCCs (Record Stores 5735, Computer Software 5734, Computer
    Programming 7372, Direct-Marketing Subscription 5967/5968, Digital Goods 5815-5818)
    are routed to DIGITAL because in this dataset they are dominated by digital-domain
    merchants (gracht.app, tulp.nl/abonnement, ...). This is the normalization the
    design calls out (DESIGN.md s2, s4.2): one budgeting concept spread across many MCCs.
"""

from __future__ import annotations

# ---- The 12-category budgeting taxonomy -------------------------------------
GROCERIES = "Groceries"
DINING = "Dining & Takeaway"
TRANSPORT = "Transport"
SHOPPING = "Shopping"
DIGITAL = "Digital & Subscriptions"
BILLS = "Bills & Utilities"
ENTERTAINMENT = "Entertainment"
HEALTH = "Health"
TRAVEL = "Travel"
CASH = "Cash & ATM"
FEES = "Fees & Charges"
INCOME = "Income & Refunds"

TAXONOMY = [
    GROCERIES, DINING, TRANSPORT, SHOPPING, DIGITAL, BILLS,
    ENTERTAINMENT, HEALTH, TRAVEL, CASH, FEES, INCOME,
]

# ---- Layer 0 structural rules: transaction TYPE -> category -----------------
# These bypass the ML cascade entirely (no merchant to learn from). DESIGN.md s4.2.
TYPE_RULES = {
    "ATM": CASH,
    "FEE": FEES,
    "CHARGE": FEES,
    "CARD_CREDIT": INCOME,
    "CARD_CHARGEBACK": INCOME,
    # NOTE: CARD_REFUND carries a real MCC/category in this data, so it is NOT a pure
    # structural type -- it flows through the ML cascade like a merchant txn, but its
    # negative amount makes it money-in. We still treat it as INCOME at money-flow time.
}

# Specific MCC overrides (take precedence over range rules below).
_MCC_OVERRIDES = {
    # --- Digital & Subscriptions (the normalization story) ---
    "5734": DIGITAL,   # Computer Software Stores
    "5735": DIGITAL,   # Record Stores  (gracht.* digital domain)
    "5815": DIGITAL,   # Digital Goods - Books/Movies
    "5816": DIGITAL,   # Digital Goods - Games
    "5817": DIGITAL,   # Digital Goods - Applications (excl games)
    "5818": DIGITAL,   # Digital Goods - Large Digital Goods Merchant
    "5967": DIGITAL,   # Direct Marketing - Inbound Telemarketing
    "5968": DIGITAL,   # Direct Marketing - Subscription
    "7372": DIGITAL,   # Computer Programming / Data Processing
    "4816": DIGITAL,   # Computer Network / Information Services
    # --- Bills & Utilities ---
    "4814": BILLS,     # Telecommunication Services
    "4899": BILLS,     # Cable, Satellite, Pay TV/Radio
    "6300": BILLS,     # Insurance Underwriting, Premiums
    # --- Dining ---
    "5811": DINING,    # Caterers
    "5813": DINING,    # Drinking Places (bars)
    # --- Transport (fuel + parking + tolls) ---
    "5541": TRANSPORT, "5542": TRANSPORT,  # Service Stations / Automated Fuel
    "7523": TRANSPORT,                       # Parking Lots, Garages
    "4784": TRANSPORT,                       # Tolls / Bridge Fees
    # --- Travel ---
    "4411": TRAVEL,    # Cruise Lines
    "4722": TRAVEL,    # Travel Agencies, Tour Operators
    # --- Health ---
    "5912": HEALTH,    # Drug Stores and Pharmacies
    "5977": HEALTH,    # Cosmetic Stores
    # --- Entertainment ---
    "7995": ENTERTAINMENT,  # Betting / Casino Gambling
    # --- Groceries (food specialty) ---
    "5499": GROCERIES, # Misc Food Stores - Convenience / Specialty
    "5462": GROCERIES, # Bakeries
}


def mcc_to_pfc(mcc: str | None) -> str:
    """Map a raw MCC string to one of the 12 budgeting categories.

    Returns SHOPPING as a conservative catch-all for unknown retail MCCs so the
    function is total (never raises / never returns None) -- the ML layer can
    override. Non-merchant / missing MCC ('None') returns SHOPPING too; those rows
    are handled by Layer 0 TYPE_RULES before they ever reach this map.
    """
    if mcc is None:
        return SHOPPING
    m = str(mcc).strip()
    if m in _MCC_OVERRIDES:
        return _MCC_OVERRIDES[m]
    if not m.isdigit():
        return SHOPPING
    code = int(m)

    # --- ISO 18245 range rules -------------------------------------------------
    # Airlines / passenger air travel
    if 3000 <= code <= 3299:
        return TRAVEL
    # Car rental agencies
    if 3300 <= code <= 3499:
        return TRAVEL
    # Lodging / hotels / resorts
    if 3500 <= code <= 3999 or code == 7011:
        return TRAVEL

    # Transportation services (rail, ferries, taxis, bus, freight, toll, etc.)
    if 4000 <= code <= 4799:
        # carve-outs handled by overrides above (4411 cruise, 4722 agencies -> TRAVEL)
        return TRANSPORT
    # Utilities & telecom
    if 4800 <= code <= 4999:
        return BILLS  # (4816 -> DIGITAL handled by override)

    # Wholesale / building / hardware / home supply
    if 5000 <= code <= 5399:
        return SHOPPING
    # Grocery & food stores
    if code == 5411 or 5420 <= code <= 5499:
        return GROCERIES
    # Automotive (dealers, parts, tires, fuel)
    if 5500 <= code <= 5599:
        return TRANSPORT
    # Apparel & accessories
    if 5600 <= code <= 5699:
        return SHOPPING
    # Furniture / home / appliances / music stores
    if 5700 <= code <= 5799:
        return SHOPPING  # (5734/5735 -> DIGITAL via override)
    # Eating & drinking places
    if 5800 <= code <= 5814:
        return DINING
    # Digital goods / direct marketing / misc retail
    if 5815 <= code <= 5818:
        return DIGITAL
    if code == 5912 or code == 5977:
        return HEALTH
    if 5960 <= code <= 5969:
        return DIGITAL  # direct marketing -> subscriptions/digital
    if 5900 <= code <= 5999:
        return SHOPPING  # general retail (book/jewelry/hobby/sporting/cosmetic/cigar...)

    # Financial / cash
    if 6010 <= code <= 6012 or code == 6051:
        return CASH
    if code == 6300 or 6380 <= code <= 6399:
        return BILLS  # insurance
    if code == 6513:
        return BILLS  # real estate / rent

    # Services 7000-7299: hotels(7011 handled), recreation services, personal svc
    if code == 7011:
        return TRAVEL
    if 7012 <= code <= 7299:
        # personal/health-adjacent services (laundry, beauty, spa) lean HEALTH/SHOPPING;
        # default SHOPPING, specific beauty/health below
        if code in (7230, 7297, 7298):  # beauty, massage, health & beauty spa
            return HEALTH
        return SHOPPING
    # Business / advertising / computer services
    if 7300 <= code <= 7399:
        return DIGITAL  # (7372 override) software/IT/business-services lean digital
    # Car rental / auto services
    if 7500 <= code <= 7699:
        return TRANSPORT  # (7523 parking -> TRANSPORT override; rentals -> transport/travel)
    # Amusement & recreation & entertainment
    if 7800 <= code <= 7999:
        return ENTERTAINMENT

    # Health / medical / professional medical
    if 8000 <= code <= 8099:
        return HEALTH
    # Schools / education
    if 8200 <= code <= 8299:
        return BILLS
    # Child care / professional services / membership / legal / govt
    if 8300 <= code <= 8999:
        return BILLS
    # Government services / fines / taxes / postal
    if 9000 <= code <= 9999:
        return BILLS

    return SHOPPING


def build_mcc_pfc_table(mccs) -> dict:
    """Convenience: materialize {mcc: pfc} for a set of MCC strings."""
    return {m: mcc_to_pfc(m) for m in mccs}

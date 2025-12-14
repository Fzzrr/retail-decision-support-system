"""
Preprocess products.csv to reduce unique commodities.
- Removes unwanted commodity categories (non-retail items)
- Groups similar commodities into broader categories
"""

import pandas as pd

# Commodities to delete (non-retail/internal use items)
LIST_TO_DELETE = [
    '(CORP USE ONLY)', 'BOOKSTORE', 'BOTTLE DEPOSITS', 'COFFEE SHOP',
    'CONTINUITIES', 'COUPON', 'COUPON/MISC ITEMS', 'COUPONS/STORE & MFG',
    'DELI SUPPLIES', 'GARDEN CENTER', 'IN-STORE PHOTOFINISHING', 'J-HOOKS',
    'LAWN AND GARDEN SHOP', 'MEAT SUPPLIES', 'MISCELLANEOUS(CORP USE ONLY)',
    'NO COMMODITY DESCRIPTION', 'NON EDIBLE PRODUCTS', 'OVERNIGHT PHOTOFINISHING',
    'PHARMACY', 'PROD SUPPLIES', 'QUICK SERVICE', 'SALAD BAR',
    'SERVICE BEVERAGE', 'TICKETS', 'UNKNOWN', 'WATCHES/CALCULATORS/LOBBY'
]

# Commodity grouping mapping (original -> grouped)
COMMODITY_MAPPING = {
    # --- Fresh Food (Meat, Deli, Seafood) ---
    'LUNCHMEAT': 'DELI LUNCHMEAT',
    'DELI MEATS': 'DELI LUNCHMEAT',
    'DELI SPECIALTIES (RETAIL PK)': 'DELI LUNCHMEAT',
    'BREAKFAST SAUSAGE/SANDWICHES': 'SAUSAGE & HOT DOGS',
    'DINNER SAUSAGE': 'SAUSAGE & HOT DOGS',
    'HOT DOGS': 'SAUSAGE & HOT DOGS',
    'MEAT - MISC': 'MEAT (OTHER)',
    'RW FRESH PROCESSED MEAT': 'MEAT (OTHER)',
    'SEAFOOD - MISC': 'FRESH SEAFOOD',
    'PKG.SEAFOOD MISC': 'FRESH SEAFOOD',

    # --- Fresh Food (Fruits & Vegetables) ---
    'VALUE ADDED FRUIT': 'FRESH FRUIT (PREPARED)',
    'VALUE ADDED VEGETABLES': 'FRESH VEGETABLES (PREPARED)',
    'ORGANICS FRUIT & VEGETABLES': 'FRESH PRODUCE (ORGANIC)',

    # --- Frozen Foods ---
    'SEAFOOD - FROZEN': 'FROZEN SEAFOOD',
    'FROZEN SEAFOOD': 'FROZEN SEAFOOD',
    'FRZN POTATOES': 'FROZEN FRUITS & VEGETABLES',
    'FROZEN PACKAGE MEAT': 'FROZEN MEAT & DINNERS',
    'FRZN ICE': 'FROZEN DESSERTS',

    # --- Dry Foods (Snacks, Bread, Cereal) ---
    'CANDY - CHECKLANE': 'CANDY',
    'CANDY - PACKAGED': 'CANDY',
    'COOKIES/CONES': 'COOKIES & CRACKERS',
    'PACKAGED NATURAL SNACKS': 'SNACKS (CHIPS/PRETZELS)',

    # --- Dry Foods (Cooking Ingredients) ---
    'BAKING MIXES': 'BAKING SUPPLIES',
    'BAKING NEEDS': 'BAKING SUPPLIES',
    'FLOUR & MEALS': 'BAKING SUPPLIES',
    'SUGARS/SWEETNERS': 'BAKING SUPPLIES',
    'SHORTENING/OIL': 'BAKING SUPPLIES',
    'BAKING': 'BAKING SUPPLIES',
    'DINNER MXS:DRY': 'DRY MIXES (SAVORY/SWEET)',
    'DRY MIX DESSERTS': 'DRY MIXES (SAVORY/SWEET)',
    'DRY SAUCES/GRAVY': 'DRY MIXES (SAVORY/SWEET)',
    'PICKLE/RELISH/PKLD VEG': 'CONDIMENTS & SAUCES',
    'OLIVES': 'CONDIMENTS & SAUCES',

    # --- Refrigerated Foods (Dairy, Cheese, Ready-to-Eat) ---
    'HEAT/SERVE': 'REFRIGERATED MEALS',
    'PREPARED FOOD': 'REFRIGERATED MEALS',
    'SANDWICHES': 'REFRIGERATED MEALS',
    'PREPARED/PKGD FOODS': 'REFRIGERATED MEALS',
    'SUSHI': 'REFRIGERATED MEALS',
    'MARGARINES': 'BUTTER & MARGARINE',
    'BUTTER': 'BUTTER & MARGARINE',
    'MISC. DAIRY': 'DAIRY (OTHER)',
    'REFRIGERATED': 'DAIRY (OTHER)',
    'PARTY TRAYS': 'PARTY TRAYS',
    'BAKERY PARTY TRAYS': 'PARTY TRAYS',
    'GIFT & FRUIT BASKETS': 'PARTY TRAYS',

    # --- Beverages ---
    'BEVERAGE': 'SOFT DRINKS',
    'NDAIRY/TEAS/JUICE/SOD': 'SOFT DRINKS',
    'DOMESTIC WINE': 'WINE',
    'IMPORTED WINE': 'WINE',
    'MISC WINE': 'WINE',

    # --- Household (Non-Food) ---
    'COFFEE FILTERS': 'HOUSEHOLD PAPER PRODUCTS',
    'FD WRAPS/BAGS/TRSH BG': 'FOOD STORAGE & WRAPS',
    'DISPOSIBLE FOILWARE': 'FOOD STORAGE & WRAPS',
    'HOME FREEZING & CANNING SUPPLY': 'FOOD STORAGE & WRAPS',
    'IRONING AND CHEMICALS': 'HOUSEHOLD CLEANING',
    'KITCHEN GADGETS': 'KITCHENWARE',
    'COOKWARE & BAKEWARE': 'KITCHENWARE',
    'PLASTIC HOUSEWARES': 'KITCHENWARE',
    'GLASSWARE & DINNERWARE': 'KITCHENWARE',
    'FUEL': 'FUEL & AUTOMOTIVE',
    'PROPANE': 'FUEL & AUTOMOTIVE',
    'AUTOMOTIVE PRODUCTS': 'FUEL & AUTOMOTIVE',
    'DOMESTIC GOODS': 'HOME GOODS',
    'HOME FURNISHINGS': 'HOME GOODS',
    'SEWING': 'HOME GOODS',

    # --- Health & Personal Care (HBC) ---
    'INFANT FORMULA': 'BABY FOOD',
    'DIETARY AID PRODUCTS': 'DIET & NUTRITION',
    'FITNESS&DIET': 'DIET & NUTRITION',
    'RESTRICTED DIET': 'DIET & NUTRITION',
    'BATH': 'SOAP & BATH',
    'SOAP - LIQUID & BAR': 'SOAP & BATH',
    'COLD AND FLU': 'MEDICINE (OTC)',
    'ANALGESICS': 'MEDICINE (OTC)',
    'ANTACIDS': 'MEDICINE (OTC)',
    'SINUS AND ALLERGY': 'MEDICINE (OTC)',
    'LAXATIVES': 'MEDICINE (OTC)',
    'HAND/BODY/FACIAL PRODUCTS': 'PERSONAL CARE (SKIN/HAIR)',
    'ETHNIC PERSONAL CARE': 'PERSONAL CARE (SKIN/HAIR)',
    'SUNTAN': 'PERSONAL CARE (SKIN/HAIR)',
    'FRAGRANCES': 'PERSONAL CARE (SKIN/HAIR)',
    'MAKEUP AND TREATMENT': 'MAKEUP & ACCESSORIES',
    'COSMETIC ACCESSORIES': 'MAKEUP & ACCESSORIES',
    'EYE AND EAR CARE PRODUCTS': 'VISION & HEARING',
    'GLASSES/VISION AIDS': 'VISION & HEARING',
    'NATURAL HBC': 'HBC (OTHER)',
    'MISCELLANEOUS HBC': 'HBC (OTHER)',
    'HOME HEALTH CARE': 'HBC (OTHER)',
    'SMOKING CESSATIONS': 'HBC (OTHER)',

    # --- General, Seasonal, Others ---
    'CIGARETTES': 'TOBACCO',
    'TOBACCO OTHER': 'TOBACCO',
    'CIGARS': 'TOBACCO',
    'CHRISTMAS SEASONAL': 'SEASONAL',
    'FIREWORKS': 'SEASONAL',
    'ROSES': 'FLORAL',
    'BOUQUET (NON ROSE)': 'FLORAL',
    'EASTER LILY': 'FLORAL',
    'HOSIERY/SOCKS': 'APPAREL',
    'MAGAZINE': 'MEDIA & ELECTRONICS',
    'NEWSPAPER': 'MEDIA & ELECTRONICS',
    'AUDIO/VIDEO PRODUCTS': 'MEDIA & ELECTRONICS',
    'FILM AND CAMERA PRODUCTS': 'MEDIA & ELECTRONICS',
    'PREPAID WIRELESS&ACCESSORIES': 'MEDIA & ELECTRONICS',
    'LONG DISTANCE CALLING CARDS': 'MEDIA & ELECTRONICS',
    'TOYS AND GAMES': 'TOYS & HOBBIES',
    'SPORTS MEMORABILIA': 'TOYS & HOBBIES',
    'TOYS': 'TOYS & HOBBIES'
}


def preprocess_products(input_path, output_path=None):
    """
    Preprocess products CSV:
    1. Remove unwanted commodities
    2. Apply commodity grouping
    
    Returns preprocessed DataFrame and saves to output_path if provided.
    """
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    original_commodities = df['COMMODITY_DESC'].nunique()
    print(f"Original: {original_rows:,} products, {original_commodities} unique commodities")
    
    # Step 1: Remove unwanted commodities
    df = df[~df['COMMODITY_DESC'].isin(LIST_TO_DELETE)]
    after_delete = len(df)
    deleted = original_rows - after_delete
    print(f"Removed {deleted:,} products from {len(LIST_TO_DELETE)} unwanted categories")
    
    # Step 2: Apply commodity mapping
    df['COMMODITY_DESC'] = df['COMMODITY_DESC'].replace(COMMODITY_MAPPING)
    
    final_commodities = df['COMMODITY_DESC'].nunique()
    print(f"Final: {len(df):,} products, {final_commodities} unique commodities")
    print(f"Reduction: {original_commodities} -> {final_commodities} commodities ({original_commodities - final_commodities} merged)")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Process the products file
    input_file = "datasets/product.csv"
    output_file = "datasets/product_grouped.csv"
    
    df = preprocess_products(input_file, output_file)
    
    # Show sample of grouped commodities
    print("\n--- Sample Grouped Commodities ---")
    commodity_counts = df['COMMODITY_DESC'].value_counts()
    print(commodity_counts.head(20))

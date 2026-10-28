"""Unified label space across all datasets.

Every source label maps to one unified key in PlantVillage style ("Crop___Condition")
so existing knowledge-base entries keep working, or to None (dropped).

Merge rule: only merge labels that name the same disease (same pathogen). Labels
that are ambiguous are dropped rather than guessed. Merges that are reasonable
but not certain are listed in UNCERTAIN so they can be reviewed.
"""
from __future__ import annotations

BACKGROUND = "Background_without_leaves"

# ---------------------------------------------------------------- PlantVillage
# Folder names are already unified keys.
PLANTVILLAGE = {k: k for k in [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    BACKGROUND, "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Corn___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]}

# ---------------------------------------------------------------- PlantDoc (folder names)
PLANTDOC = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry___healthy",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

# ---------------------------------------------------------------- PlantWild v1 (classes.txt names)
PLANTWILD = {
    "apple black rot": "Apple___Black_rot",
    "apple leaf": "Apple___healthy",
    "apple mosaic virus": "Apple___Mosaic_virus",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "banana leaf": "Banana___healthy",
    "banana panama disease": "Banana___Panama_disease",
    "basil downy mildew": "Basil___Downy_mildew",
    "basil leaf": "Basil___healthy",
    "bean halo blight": "Bean___Halo_blight",
    "bean leaf": "Bean___healthy",
    "bean mosaic virus": "Bean___Mosaic_virus",
    "bean rust": "Bean___Rust",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "blueberry rust": "Blueberry___Rust",
    "broccoli downy mildew": "Broccoli___Downy_mildew",
    "broccoli leaf": "Broccoli___healthy",
    "cabbage alternaria leaf spot": "Cabbage___Alternaria_leaf_spot",
    "cabbage leaf": "Cabbage___healthy",
    "carrot cavity spot": None,           # a root disease, not a leaf disease
    "cauliflower alternaria leaf spot": "Cauliflower___Alternaria_leaf_spot",
    "cauliflower leaf": "Cauliflower___healthy",
    "celery anthracnose": "Celery___Anthracnose",
    "celery early blight": "Celery___Early_blight",
    "celery leaf": "Celery___healthy",
    "cherry leaf": "Cherry___healthy",
    "cherry leaf spot": "Cherry___Leaf_spot",
    "cherry powdery mildew": "Cherry___Powdery_mildew",
    "citrus canker": "Citrus___Canker",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "coffee leaf": "Coffee___healthy",
    "coffee leaf rust": "Coffee___Leaf_rust",
    "corn gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf": "Corn___healthy",
    "corn northern leaf blight": "Corn___Northern_Leaf_Blight",
    "corn rust": "Corn___Common_rust",
    "corn smut": None,                    # galls on ears/stalks, not a leaf diagnosis
    "cucumber angular leaf spot": "Cucumber___Angular_leaf_spot",
    "cucumber bacterial wilt": "Cucumber___Bacterial_wilt",
    "cucumber leaf": "Cucumber___healthy",
    "cucumber powdery mildew": "Cucumber___Powdery_mildew",
    "eggplant cercospora leaf spot": "Eggplant___Cercospora_leaf_spot",
    "eggplant leaf": "Eggplant___healthy",
    "garlic leaf": "Garlic___healthy",
    "garlic leaf blight": "Garlic___Leaf_blight",
    "garlic rust": "Garlic___Rust",
    "ginger leaf": "Ginger___healthy",
    "ginger leaf spot": "Ginger___Leaf_spot",
    "ginger sheath blight": "Ginger___Sheath_blight",
    "grape black rot": "Grape___Black_rot",
    "grape downy mildew": "Grape___Downy_mildew",
    "grape leaf": "Grape___healthy",
    "grape leaf spot": None,              # generic; could overlap Isariopsis leaf spot or black rot
    "grapevine leafroll disease": "Grape___Leafroll_disease",
    "lettuce downy mildew": "Lettuce___Downy_mildew",
    "lettuce leaf": "Lettuce___healthy",
    "lettuce mosaic virus": "Lettuce___Mosaic_virus",
    "maple leaf": "Maple___healthy",
    "maple tar spot": "Maple___Tar_spot",
    "peach leaf": "Peach___healthy",
    "peach leaf curl": "Peach___Leaf_curl",
    "plum leaf": "Plum___healthy",
    "plum pocket disease": "Plum___Pocket_disease",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "rice blast": "Rice___Blast",
    "rice leaf": "Rice___healthy",
    "rice sheath blight": "Rice___Sheath_blight",
    "soybean leaf": "Soybean___healthy",
    "squash leaf": "Squash___healthy",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry anthracnose": "Strawberry___Anthracnose",
    "strawberry leaf": "Strawberry___healthy",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tobacco leaf": "Tobacco___healthy",
    "tobacco mosaic virus": "Tobacco___Mosaic_virus",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "zucchini yellow mosaic virus": "Squash___Zucchini_yellow_mosaic_virus",
}

# ---------------------------------------------------------------- CCMT raw data ("Crop/class" folders)
CCMT = {
    "Cashew/anthracnose": "Cashew___Anthracnose",
    "Cashew/gumosis": "Cashew___Gummosis",
    "Cashew/gummosis": "Cashew___Gummosis",
    "Cashew/healthy": "Cashew___healthy",
    "Cashew/leaf miner": "Cashew___Leaf_miner",
    "Cashew/red rust": "Cashew___Red_rust",
    "Cassava/bacterial blight": "Cassava___Bacterial_blight",
    "Cassava/brown spot": "Cassava___Brown_spot",
    "Cassava/green mite": "Cassava___Green_mite",
    "Cassava/healthy": "Cassava___healthy",
    "Cassava/mosaic": "Cassava___Mosaic_disease",
    "Maize/fall armyworm": "Corn___Fall_armyworm",
    "Maize/grasshoper": "Corn___Grasshopper_damage",
    "Maize/grasshopper": "Corn___Grasshopper_damage",
    "Maize/healthy": "Corn___healthy",
    "Maize/leaf beetle": "Corn___Leaf_beetle_damage",
    "Maize/leaf blight": "Corn___Northern_Leaf_Blight",
    "Maize/leaf spot": None,              # unspecified leaf spot; could be several pathogens
    "Maize/streak virus": "Corn___Maize_streak_virus",
    "Tomato/healthy": "Tomato___healthy",
    "Tomato/leaf blight": None,           # could be early OR late blight
    "Tomato/leaf curl": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato/septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato/verticulium wilt": "Tomato___Verticillium_wilt",
    "Tomato/verticillium wilt": "Tomato___Verticillium_wilt",
}

# ---------------------------------------------------------------- Plant Pathology 2021 (single-label only)
PLANT_PATHOLOGY = {
    "healthy": "Apple___healthy",
    "scab": "Apple___Apple_scab",
    "rust": "Apple___Cedar_apple_rust",
    # Frogeye leaf spot is the leaf phase of black rot (Botryosphaeria obtusa),
    # which is what PlantVillage's "Apple Black rot" leaf images show.
    "frog_eye_leaf_spot": "Apple___Black_rot",
    "powdery_mildew": "Apple___Powdery_mildew",
    "complex": None,                      # "several diseases", not one label
}

SOURCES = {
    "plantvillage": PLANTVILLAGE,
    "plantdoc": PLANTDOC,
    "plantwild": PLANTWILD,
    "ccmt": CCMT,
    "plantpathology": PLANT_PATHOLOGY,
}

# Lab vs real-world, used for per-domain reporting.
FIELD_SOURCES = {"plantdoc", "plantwild", "ccmt", "plantpathology"}

UNCERTAIN = {
    ("ccmt", "Maize/leaf blight"): "Assumed northern corn leaf blight (most common maize leaf blight).",
    ("ccmt", "Tomato/leaf curl"): "Assumed TYLCV (the dominant tomato leaf-curl virus in West Africa).",
    ("plantwild", "apple rust"): "Web 'apple rust' is mostly cedar-apple rust but may include other Gymnosporangium rusts.",
    ("plantdoc", "Bell_pepper leaf spot"): "PlantDoc's pepper leaf spot is treated as bacterial spot, as in the PlantDoc paper.",
}


def map_label(source: str, label: str) -> str | None:
    table = SOURCES[source]
    if label not in table:
        raise KeyError(f"unmapped {source} label: {label!r}")
    return table[label]

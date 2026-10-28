"""Knowledge-base entries for classes added by the multi-source dataset (ml/scratch).

Merged into disease_info.json by:  python api/app/data/extra_diseases.py
"""
import json
from pathlib import Path

P = Path(__file__).with_name("disease_info.json")


def d(crop, name, typ, pathogen, sev, summary, symptoms, treatment, prevention):
    return {"crop": crop, "name": name, "type": typ, "pathogen": pathogen, "severity": sev,
            "summary": summary, "symptoms": symptoms, "treatment": treatment, "prevention": prevention}


def healthy(crop, tips):
    return d(crop, "Healthy", "Healthy", None, "none",
             "No disease symptoms detected. The leaf looks healthy.", [], [], tips)


NEW = {
    "Apple___Mosaic_virus": d(
        "Apple", "Apple Mosaic", "Viral", "Apple mosaic virus (ApMV, ilarvirus)", "low",
        "A graft-transmitted virus that causes bright yellow or cream patterns on leaves. It usually reduces vigour slightly rather than killing trees.",
        ["Bright yellow to cream blotches, rings or bands, often along veins", "Patterns appear on spring leaves and may fade in summer heat", "Affected leaves may scorch and drop in hot weather"],
        ["There is no cure; mildly affected trees can usually stay in production", "Remove severely affected trees in new plantings"],
        ["Plant certified virus-free trees and rootstocks", "Use only virus-tested budwood", "Do not graft from symptomatic trees"]),
    "Apple___Powdery_mildew": d(
        "Apple", "Powdery Mildew", "Fungal", "Podosphaera leucotricha", "moderate",
        "A fungal disease that overwinters in buds and attacks new growth, stunting shoots and russeting fruit.",
        ["White to grey powdery felt on young leaves and shoots", "Leaves narrow, crinkle and curl upward", "Silvery, stunted shoot tips that carry the fungus over winter"],
        ["Prune out mildewed shoot tips in winter and during the season", "Apply sulfur or another labelled fungicide from tight cluster into early summer"],
        ["Plant resistant cultivars", "Prune for an open canopy", "Avoid excess nitrogen, which drives soft, susceptible growth"]),
    "Banana___Panama_disease": d(
        "Banana", "Panama Disease (Fusarium Wilt)", "Fungal", "Fusarium oxysporum f. sp. cubense (incl. Tropical Race 4)", "high",
        "A soil-borne wilt that blocks the plant's water-conducting tissue. The fungus survives in soil for decades, making it one of the most destructive banana diseases.",
        ["Yellowing of older leaves, starting at the margins", "Leaves collapse and hang around the pseudostem like a skirt", "Splitting at the pseudostem base; red-brown streaks inside the stem"],
        ["There is no cure; destroy infected plants in place and do not move soil or plant material", "Report suspected TR4 to your agricultural authority where it is regulated"],
        ["Use disease-free tissue-culture plantlets and tolerant cultivars", "Disinfect tools, boots and vehicles between fields", "Avoid moving soil and water from infested areas"]),
    "Banana___healthy": healthy("Banana", ["Use clean planting material", "Keep good drainage", "Remove old and dead leaves"]),
    "Basil___Downy_mildew": d(
        "Basil", "Downy Mildew", "Oomycete", "Peronospora belbahrii", "high",
        "A fast-spreading water mould favoured by humid nights; it can ruin a basil crop within days.",
        ["Yellowing between veins, starting on lower leaves", "Grey-purple fuzzy growth on the leaf underside", "Leaves turn brown and drop"],
        ["Remove and bag infected plants promptly", "Harvest healthy leaves early before the disease spreads"],
        ["Grow downy-mildew-resistant varieties", "Space plants and water in the morning so leaves dry quickly", "Buy seed and transplants from reliable sources"]),
    "Basil___healthy": healthy("Basil", ["Water at the base in the morning", "Give plants space for airflow", "Pinch flowers to keep growth vigorous"]),
    "Bean___Halo_blight": d(
        "Bean", "Halo Blight", "Bacterial", "Pseudomonas savastanoi pv. phaseolicola", "moderate",
        "A seed-borne bacterial disease favoured by cool, wet weather and spread by rain splash.",
        ["Small water-soaked spots on leaves", "A wide yellow-green halo around each spot", "Greasy spots on pods; yellowing of whole plants in severe cases"],
        ["Remove infected plants", "Copper sprays can slow spread but will not cure infected plants", "Do not work in the crop while leaves are wet"],
        ["Plant certified disease-free seed", "Rotate away from beans for 2 to 3 years", "Bury or remove crop debris"]),
    "Bean___Mosaic_virus": d(
        "Bean", "Bean Common Mosaic", "Viral", "Bean common mosaic virus (BCMV, potyvirus)", "moderate",
        "A seed-borne virus also spread by aphids. Early infections can cause large yield losses.",
        ["Light and dark green mosaic pattern on leaves", "Leaves curl downward, pucker or become distorted", "Stunted plants with fewer pods"],
        ["There is no cure; remove infected plants early", "Control aphids around young plants"],
        ["Grow resistant varieties", "Use certified virus-free seed", "Control weeds that host aphids"]),
    "Bean___Rust": d(
        "Bean", "Bean Rust", "Fungal", "Uromyces appendiculatus", "moderate",
        "A rust disease spread by wind-blown spores in humid weather with long dew periods.",
        ["Small reddish-brown powdery pustules, mainly on leaf undersides", "Yellow halos around pustules", "Leaves yellow and drop in severe cases"],
        ["Apply a labelled fungicide (e.g. sulfur or chlorothalonil) at the first pustules", "Remove heavily infected leaves"],
        ["Plant rust-resistant varieties", "Rotate crops and destroy bean debris", "Space rows for airflow"]),
    "Bean___healthy": healthy("Bean", ["Use certified seed", "Rotate with non-legume crops", "Avoid working among wet plants"]),
    "Blueberry___Rust": d(
        "Blueberry", "Leaf Rust", "Fungal", "Thekopsora minima (syn. Pucciniastrum vaccinii)", "moderate",
        "A rust disease that causes early defoliation, weakening bushes and reducing next season's crop.",
        ["Yellow spots on the upper leaf surface that turn reddish-brown", "Orange-yellow powdery pustules on the underside", "Premature leaf drop late in the season"],
        ["Apply a labelled fungicide when rust first appears", "Remove and destroy fallen infected leaves"],
        ["Choose less susceptible cultivars", "Prune for good airflow", "Keep hemlock, the alternate host, away from plantings where possible"]),
    "Broccoli___Downy_mildew": d(
        "Broccoli", "Downy Mildew", "Oomycete", "Hyaloperonospora brassicae", "moderate",
        "A water-mould disease of brassicas favoured by cool, humid conditions.",
        ["Yellow, angular patches on the upper leaf surface", "White to grey downy growth on the underside", "Dark streaks in heads in severe cases"],
        ["Remove infected leaves", "Apply a labelled fungicide if conditions stay cool and wet"],
        ["Avoid overhead watering and space plants for airflow", "Rotate brassicas with other crops", "Use resistant varieties where available"]),
    "Broccoli___healthy": healthy("Broccoli", ["Rotate brassica crops", "Keep leaves dry", "Remove crop debris after harvest"]),
    "Cabbage___Alternaria_leaf_spot": d(
        "Cabbage", "Alternaria Leaf Spot", "Fungal", "Alternaria brassicicola / A. brassicae", "moderate",
        "A seed-borne fungal disease of brassicas, favoured by warm, wet weather.",
        ["Dark brown to black round spots, often with concentric rings", "Yellow halo around spots; centres may drop out", "Spots on heads reduce marketability"],
        ["Remove lower infected leaves", "Apply a labelled fungicide (e.g. chlorothalonil or azoxystrobin) when spots appear"],
        ["Use hot-water-treated or certified seed", "Rotate brassicas for 3 years", "Plough in or remove crop debris"]),
    "Cabbage___healthy": healthy("Cabbage", ["Rotate brassica crops", "Water at the base", "Remove crop debris"]),
    "Cashew___Anthracnose": d(
        "Cashew", "Anthracnose", "Fungal", "Colletotrichum gloeosporioides", "high",
        "A major cashew disease in humid weather, attacking young leaves, flowers and developing nuts.",
        ["Reddish-brown to dark irregular lesions on young leaves", "Lesions merge and leaves wither", "Blackened flowers and shrivelled young nuts"],
        ["Prune and burn infected twigs", "Apply a copper-based fungicide at new flushing and flowering"],
        ["Prune for an open canopy", "Remove infected debris from under trees", "Plant tolerant clones"]),
    "Cashew___Gummosis": d(
        "Cashew", "Gummosis", "Fungal", "Lasiodiplodia theobromae", "high",
        "A fungal disease of the trunk and branches that causes gum oozing and can kill branches or whole trees.",
        ["Amber gum oozing from cracks in the bark", "Dark, sunken bark lesions", "Yellowing leaves and dieback of affected branches"],
        ["Scrape out infected bark and apply a copper-based or Bordeaux paste", "Cut out and burn dead branches"],
        ["Avoid wounding trunks during farm work", "Keep trees vigorous with good nutrition", "Remove dead wood regularly"]),
    "Cashew___Leaf_miner": d(
        "Cashew", "Leaf Miner", "Pest", "Acrocercops syngramma (moth larvae)", "moderate",
        "Moth larvae tunnel inside tender young leaves, damaging new flushes.",
        ["Silvery or whitish blister-like mines on young leaves", "Mined leaves crumple, curl and dry", "Damage concentrated on new flushes"],
        ["Spray neem or a labelled insecticide on new flushes when mines first appear", "Remove and destroy heavily mined shoots"],
        ["Monitor new flushes regularly", "Encourage natural enemies such as parasitic wasps", "Avoid unnecessary broad-spectrum sprays"]),
    "Cashew___Red_rust": d(
        "Cashew", "Red Rust (Algal Leaf Spot)", "Algal", "Cephaleuros virescens (parasitic alga)", "low",
        "A parasitic alga that forms rusty, velvety spots on leaves, mostly in humid, shaded conditions.",
        ["Round, raised, orange to rust-coloured velvety spots on leaves", "Spots later turn grey-green", "Heavy infections weaken leaves"],
        ["Apply a copper-based fungicide", "Prune affected branches"],
        ["Improve airflow and reduce shading", "Keep trees well nourished"]),
    "Cashew___healthy": healthy("Cashew", ["Prune to keep an open canopy", "Scout new flushes for pests", "Remove dead wood"]),
    "Cassava___Bacterial_blight": d(
        "Cassava", "Bacterial Blight", "Bacterial", "Xanthomonas phaseoli pv. manihotis", "high",
        "A bacterial disease spread mainly through infected stem cuttings and rain splash. It can cause severe losses in wet seasons.",
        ["Angular, water-soaked leaf spots that turn brown", "Leaf blight and wilting", "Gummy exudate on stems and shoot dieback"],
        ["Uproot and burn badly infected plants", "Do not take cuttings from infected fields"],
        ["Plant clean cuttings from healthy fields", "Use resistant varieties", "Rotate crops and disinfect cutting tools"]),
    "Cassava___Brown_spot": d(
        "Cassava", "Brown Leaf Spot", "Fungal", "Passalora henningsii (syn. Cercosporidium henningsii)", "low",
        "A common fungal leaf spot that mostly affects older leaves and is usually of minor importance.",
        ["Brown, angular spots with dark borders on older leaves", "Spots may have a yellow halo", "Older leaves yellow and drop early"],
        ["Usually no treatment is needed", "Remove heavily affected leaves"],
        ["Grow tolerant varieties", "Maintain good soil fertility", "Clear crop debris"]),
    "Cassava___Green_mite": d(
        "Cassava", "Green Mite", "Pest", "Mononychellus tanajoa (mite)", "moderate",
        "A tiny mite that feeds on young leaves, especially in the dry season, and can cut yields sharply.",
        ["Pale yellow speckling on young leaves", "Leaves become small, mottled and deformed", "Shoot tips look bare, like a candle-stick"],
        ["Release or conserve predatory mites (e.g. Typhlodromalus aripo)", "Remove and destroy badly infested shoot tips"],
        ["Plant tolerant varieties", "Plant early so crops are established before the dry season", "Avoid broad-spectrum insecticides that kill predators"]),
    "Cassava___Mosaic_disease": d(
        "Cassava", "Cassava Mosaic Disease", "Viral", "Cassava mosaic begomoviruses (spread by whitefly and infected cuttings)", "high",
        "The most damaging cassava disease in Africa. It spreads through infected stem cuttings and the whitefly Bemisia tabaci.",
        ["Yellow-green mosaic mottling on leaves", "Leaves distorted, twisted and reduced in size", "Stunted plants with small roots"],
        ["There is no cure; uproot and destroy infected plants early", "Do not use cuttings from infected plants"],
        ["Plant virus-free cuttings of resistant varieties", "Regularly remove infected plants (roguing)", "Manage whitefly populations"]),
    "Cassava___healthy": healthy("Cassava", ["Use clean cuttings", "Remove sick plants early", "Scout for whiteflies and mites"]),
    "Cauliflower___Alternaria_leaf_spot": d(
        "Cauliflower", "Alternaria Leaf Spot", "Fungal", "Alternaria brassicicola / A. brassicae", "moderate",
        "A seed-borne fungal disease of brassicas, favoured by warm, wet weather.",
        ["Dark round spots with concentric rings on leaves", "Yellow halos; spot centres may fall out", "Brown spots on curds reduce quality"],
        ["Remove infected lower leaves", "Apply a labelled fungicide when spots appear"],
        ["Use treated or certified seed", "Rotate brassicas for 3 years", "Remove crop debris"]),
    "Cauliflower___healthy": healthy("Cauliflower", ["Rotate brassica crops", "Water at the base", "Remove crop debris"]),
    "Celery___Anthracnose": d(
        "Celery", "Anthracnose (Leaf Curl)", "Fungal", "Colletotrichum fioriniae (C. acutatum complex)", "moderate",
        "A fungal disease that twists and curls celery leaves and stalks, favoured by warm, wet weather.",
        ["Leaves cup and curl; leaf stalks twist", "Tan to brown oval lesions on stalks", "Distorted, stunted plants"],
        ["Remove affected plants", "Apply a labelled fungicide during wet periods"],
        ["Use clean transplants", "Avoid overhead irrigation", "Rotate crops and remove debris"]),
    "Celery___Early_blight": d(
        "Celery", "Early Blight", "Fungal", "Cercospora apii", "moderate",
        "A common fungal leaf disease of celery, spread by splashing water in warm, humid weather.",
        ["Small yellowish spots that enlarge into grey-tan lesions", "Grey fuzzy growth on lesions in humid weather", "Leaves wither in severe cases"],
        ["Remove infected leaves", "Apply a labelled fungicide on a regular schedule in wet weather"],
        ["Use treated seed", "Rotate crops", "Water at the base"]),
    "Celery___healthy": healthy("Celery", ["Water at the base", "Rotate crops", "Remove crop debris"]),
    "Cherry___Leaf_spot": d(
        "Cherry", "Cherry Leaf Spot", "Fungal", "Blumeriella jaapii", "moderate",
        "A common fungal disease, especially of sour cherry, that causes early defoliation and weakens trees.",
        ["Small purple to red-brown spots on the upper leaf surface", "Spot centres fall out, leaving shot holes", "Leaves yellow and drop by mid-summer"],
        ["Apply a labelled fungicide from petal fall through harvest in wet seasons", "Rake and destroy fallen leaves"],
        ["Clean up leaf litter in autumn", "Prune for airflow", "Choose less susceptible cultivars"]),
    "Citrus___Canker": d(
        "Citrus", "Citrus Canker", "Bacterial", "Xanthomonas citri subsp. citri", "high",
        "A highly contagious bacterial disease spread by wind-driven rain, people and equipment. It is regulated in many regions.",
        ["Raised, corky, tan-brown lesions on leaves, twigs and fruit", "Water-soaked margins and a yellow halo around lesions", "Lesions visible on both sides of the leaf"],
        ["Report suspected cases to your agricultural authority", "Remove and destroy infected shoots", "Apply copper-based bactericides to protect new growth"],
        ["Plant certified disease-free trees", "Use windbreaks", "Control citrus leaf miner, whose damage helps infection", "Disinfect tools and equipment"]),
    "Coffee___Leaf_rust": d(
        "Coffee", "Coffee Leaf Rust", "Fungal", "Hemileia vastatrix", "high",
        "The most important coffee disease worldwide. It causes heavy leaf loss and weakens plants for the next harvest.",
        ["Yellow-orange spots on the upper leaf surface", "Orange powdery spores on the underside", "Early leaf drop and branch dieback"],
        ["Apply copper-based or labelled systemic fungicides at the start of the rainy season", "Remove heavily infected leaves"],
        ["Plant rust-resistant varieties", "Manage shade and plant spacing", "Maintain good nutrition"]),
    "Coffee___healthy": healthy("Coffee", ["Balance shade and sun", "Maintain good nutrition", "Scout leaf undersides for rust"]),
    "Corn___Fall_armyworm": d(
        "Corn (maize)", "Fall Armyworm", "Pest", "Spodoptera frugiperda (moth larvae)", "high",
        "An invasive caterpillar that feeds in the maize whorl and can destroy young crops quickly.",
        ["Small 'windowpane' patches where young larvae scraped the leaf", "Ragged holes in leaves as the whorl unfolds", "Sawdust-like frass in the whorl"],
        ["Scout weekly; hand-pick or crush larvae in small plots", "Apply biological products (Bt, neem) or a labelled insecticide into the whorl when thresholds are reached"],
        ["Plant early and at the same time as neighbours", "Intercropping and push-pull systems reduce infestation", "Conserve natural enemies"]),
    "Corn___Grasshopper_damage": d(
        "Corn (maize)", "Grasshopper Damage", "Pest", "Grasshoppers (Orthoptera)", "low",
        "Grasshoppers chew leaves from the margins; damage is usually limited unless populations are high.",
        ["Irregular chewed areas along leaf edges", "Large holes in leaves", "Grasshoppers visible on plants"],
        ["Light damage can usually be tolerated", "Use baits or a labelled insecticide only if defoliation is severe"],
        ["Manage weedy field margins where grasshoppers breed", "Scout regularly"]),
    "Corn___Leaf_beetle_damage": d(
        "Corn (maize)", "Leaf Beetle Damage", "Pest", "Leaf beetles (Chrysomelidae)", "low",
        "Adult leaf beetles scrape or chew maize leaves; some species can also spread viruses.",
        ["Long scraped stripes or small holes in leaves", "Feeding scars that turn white or brown", "Beetles on leaves"],
        ["Usually no treatment is needed for light damage", "Apply a labelled insecticide if heavy feeding threatens young plants"],
        ["Rotate crops", "Control grassy weeds", "Scout young crops"]),
    "Corn___Maize_streak_virus": d(
        "Corn (maize)", "Maize Streak Virus", "Viral", "Maize streak virus (MSV, mastrevirus; spread by Cicadulina leafhoppers)", "high",
        "Africa's most important maize virus. Plants infected young become stunted and may produce no grain.",
        ["Narrow, broken yellow streaks running along leaf veins", "Streaks merge into yellow stripes", "Stunting when infected young"],
        ["There is no cure; remove infected seedlings", "Manage leafhoppers early in the season"],
        ["Plant MSV-resistant varieties", "Avoid late plantings next to older maize", "Control grass weeds that host the virus"]),
    "Cucumber___Angular_leaf_spot": d(
        "Cucumber", "Angular Leaf Spot", "Bacterial", "Pseudomonas syringae pv. lachrymans", "moderate",
        "A seed-borne bacterial disease spread by splashing water in warm, wet weather.",
        ["Small water-soaked spots bounded by leaf veins (angular)", "Spots turn tan and dry; centres tear out", "White crusty ooze on spots in the morning"],
        ["Remove infected leaves", "Copper sprays can slow spread"],
        ["Use disease-free seed", "Rotate cucurbits for 2 years", "Use drip irrigation"]),
    "Cucumber___Bacterial_wilt": d(
        "Cucumber", "Bacterial Wilt", "Bacterial", "Erwinia tracheiphila (spread by cucumber beetles)", "high",
        "A bacterial wilt spread by striped and spotted cucumber beetles. Infected plants usually die.",
        ["Individual leaves or runners suddenly wilt", "The whole plant wilts and dies", "Sticky strands when a cut stem is slowly pulled apart"],
        ["There is no cure; remove wilted plants", "Control cucumber beetles immediately"],
        ["Use row covers until flowering", "Control beetles early", "Grow less susceptible varieties"]),
    "Cucumber___Powdery_mildew": d(
        "Cucumber", "Powdery Mildew", "Fungal", "Podosphaera xanthii", "moderate",
        "A very common cucurbit disease that does not need wet leaves to spread.",
        ["White powdery spots on leaves and stems", "Spots spread to cover whole leaves", "Leaves yellow and die early"],
        ["Apply sulfur, potassium bicarbonate or another labelled fungicide early", "Alternate fungicide groups to avoid resistance"],
        ["Grow resistant varieties", "Give plants full sun and airflow", "Remove crop debris"]),
    "Cucumber___healthy": healthy("Cucumber", ["Use drip irrigation", "Scout for cucumber beetles", "Rotate cucurbits"]),
    "Eggplant___Cercospora_leaf_spot": d(
        "Eggplant", "Cercospora Leaf Spot", "Fungal", "Cercospora melongenae", "moderate",
        "A fungal leaf spot favoured by warm, humid weather, causing early leaf loss.",
        ["Round to angular yellowish spots", "Spots turn brown with a grey centre", "Leaves yellow and drop"],
        ["Remove infected leaves", "Apply a labelled fungicide when spots appear"],
        ["Rotate crops", "Space plants for airflow", "Remove crop debris"]),
    "Eggplant___healthy": healthy("Eggplant", ["Rotate solanaceous crops", "Water at the base", "Scout for mites and beetles"]),
    "Garlic___Leaf_blight": d(
        "Garlic", "Leaf Blight", "Fungal", "Commonly Stemphylium vesicarium", "moderate",
        "A fungal leaf blight of garlic and onions that thrives in warm, wet weather and on stressed plants.",
        ["Small water-soaked or yellow spots on leaves", "Spots become elongated tan to brown lesions", "Leaf tips die back"],
        ["Remove badly affected leaves", "Apply a labelled fungicide in wet weather"],
        ["Rotate alliums for 3 years", "Avoid overhead irrigation", "Remove crop residue"]),
    "Garlic___Rust": d(
        "Garlic", "Garlic Rust", "Fungal", "Puccinia allii", "moderate",
        "A rust disease of garlic and leeks that reduces bulb size when infection is heavy.",
        ["Small yellow flecks on leaves", "Bright orange powdery pustules", "Leaves yellow and die early"],
        ["Apply a labelled fungicide at the first pustules", "Remove heavily infected leaves"],
        ["Rotate alliums", "Avoid excess nitrogen", "Space plants for airflow"]),
    "Garlic___healthy": healthy("Garlic", ["Plant clean cloves", "Rotate alliums", "Avoid waterlogged soil"]),
    "Ginger___Leaf_spot": d(
        "Ginger", "Leaf Spot", "Fungal", "Phyllosticta zingiberis", "moderate",
        "A fungal leaf spot of ginger spread by rain splash in humid weather.",
        ["Small oval spots with a white papery centre", "Dark brown margins with a yellow halo", "Spots merge and leaves dry"],
        ["Remove infected leaves", "Apply a labelled copper or other fungicide"],
        ["Use healthy seed rhizomes", "Provide good drainage", "Remove crop debris"]),
    "Ginger___Sheath_blight": d(
        "Ginger", "Sheath Blight", "Fungal", "Rhizoctonia solani", "moderate",
        "A soil-borne fungal disease favoured by warm, wet conditions and dense planting.",
        ["Water-soaked lesions on leaf sheaths and leaves", "Lesions with pale centres and irregular brown margins", "Leaves dry from the base upward"],
        ["Remove affected plants", "Apply a labelled fungicide where available"],
        ["Improve drainage", "Avoid dense planting", "Rotate crops"]),
    "Ginger___healthy": healthy("Ginger", ["Use healthy seed rhizomes", "Keep soil well drained", "Mulch to conserve moisture"]),
    "Grape___Downy_mildew": d(
        "Grape", "Downy Mildew", "Oomycete", "Plasmopara viticola", "high",
        "A destructive water-mould disease in warm, wet seasons that attacks leaves, shoots and clusters.",
        ["Yellow, oily-looking spots on the upper leaf surface", "White downy growth on the underside", "Clusters brown and shrivel"],
        ["Apply a labelled fungicide (e.g. copper or mancozeb) before and during wet periods", "Remove infected shoots and clusters"],
        ["Train vines for airflow", "Remove leaves around clusters", "Clean up fallen leaves"]),
    "Grape___Leafroll_disease": d(
        "Grape", "Leafroll Disease", "Viral", "Grapevine leafroll-associated viruses (e.g. GLRaV-3)", "high",
        "A virus disease spread by infected planting material and mealybugs. It lowers yield and delays ripening.",
        ["Leaves roll downward in late summer", "Red or purple leaves with green veins (red varieties)", "Yellowing between veins (white varieties)"],
        ["There is no cure; remove infected vines", "Control mealybugs to limit spread"],
        ["Plant certified virus-tested vines", "Monitor for mealybugs", "Remove infected vines promptly"]),
    "Lettuce___Downy_mildew": d(
        "Lettuce", "Downy Mildew", "Oomycete", "Bremia lactucae", "moderate",
        "A water-mould disease favoured by cool, moist conditions.",
        ["Pale yellow, angular patches on older leaves", "White downy growth on the underside", "Patches turn brown and dry"],
        ["Remove infected leaves", "Apply a labelled fungicide in cool, wet weather"],
        ["Grow resistant varieties", "Improve airflow", "Avoid evening watering"]),
    "Lettuce___Mosaic_virus": d(
        "Lettuce", "Lettuce Mosaic", "Viral", "Lettuce mosaic virus (LMV, potyvirus)", "moderate",
        "A seed-borne virus spread further by aphids.",
        ["Mottling and vein clearing on leaves", "Leaves ruffled or distorted", "Stunting; heads fail to form"],
        ["There is no cure; remove infected plants", "Control aphids"],
        ["Use virus-tested seed", "Control weeds that host the virus", "Avoid planting next to old lettuce crops"]),
    "Lettuce___healthy": healthy("Lettuce", ["Water in the morning", "Rotate crops", "Control aphids early"]),
    "Maple___Tar_spot": d(
        "Maple", "Tar Spot", "Fungal", "Rhytisma acerinum (and related species)", "low",
        "A mostly cosmetic leaf disease of maples that rarely harms tree health.",
        ["Yellow-green spots in early summer", "Spots develop raised, shiny black 'tar' patches", "Possible early leaf drop"],
        ["Usually no treatment is needed"],
        ["Rake and destroy fallen leaves", "Prune for airflow"]),
    "Maple___healthy": healthy("Maple", ["Rake fallen leaves", "Water during droughts", "Mulch around the root zone"]),
    "Peach___Leaf_curl": d(
        "Peach", "Peach Leaf Curl", "Fungal", "Taphrina deformans", "moderate",
        "A fungal disease that infects buds in late winter and distorts spring leaves.",
        ["Thickened, puckered, curled leaves", "Leaves turn red, purple or yellow", "Infected leaves drop early"],
        ["Once symptoms appear, spraying will not help this season", "Remove distorted leaves and support the tree with water and nutrition"],
        ["Apply one fungicide spray (e.g. copper or chlorothalonil) after leaf fall or before bud swell", "Plant resistant cultivars"]),
    "Plum___Pocket_disease": d(
        "Plum", "Plum Pocket", "Fungal", "Taphrina pruni", "moderate",
        "A fungal disease that turns young plums into swollen, hollow, deformed 'bladders'.",
        ["Swollen, hollow, distorted fruit", "Fruit surface turns grey and powdery", "Thickened, curled leaves and shoots"],
        ["Remove and destroy infected fruit and shoots"],
        ["Apply a dormant fungicide spray before bud swell", "Prune for airflow"]),
    "Plum___healthy": healthy("Plum", ["Prune for airflow", "Remove mummified fruit", "Apply dormant sprays where needed"]),
    "Rice___Blast": d(
        "Rice", "Rice Blast", "Fungal", "Magnaporthe oryzae (Pyricularia oryzae)", "high",
        "One of the most destructive rice diseases worldwide. It attacks leaves, nodes and panicles.",
        ["Diamond or spindle-shaped lesions with grey centres and brown margins", "Lesions merge and leaves die", "Neck rot that makes panicles break or stay empty"],
        ["Apply a labelled fungicide (e.g. tricyclazole or azoxystrobin) at early symptoms or booting", "Avoid excess nitrogen"],
        ["Plant resistant varieties", "Use balanced fertilisation", "Avoid drought stress"]),
    "Rice___Sheath_blight": d(
        "Rice", "Sheath Blight", "Fungal", "Rhizoctonia solani", "moderate",
        "A soil-borne disease favoured by dense crops, high nitrogen and warm, humid conditions.",
        ["Oval, greenish-grey lesions on sheaths near the water line", "Lesions turn pale with brown borders", "The disease climbs to upper leaves"],
        ["Apply a labelled fungicide (e.g. validamycin or azoxystrobin) when lesions are found"],
        ["Avoid overly dense planting", "Use balanced nitrogen", "Remove weeds and stubble"]),
    "Rice___healthy": healthy("Rice", ["Use balanced fertiliser", "Manage water evenly", "Scout for blast lesions"]),
    "Squash___Zucchini_yellow_mosaic_virus": d(
        "Squash", "Zucchini Yellow Mosaic", "Viral", "Zucchini yellow mosaic virus (ZYMV, potyvirus)", "high",
        "An aphid-transmitted virus that can make a whole crop unmarketable.",
        ["Yellow mosaic and vein clearing on leaves", "Leaves narrow, blistered and distorted", "Knobby, deformed fruit"],
        ["There is no cure; remove infected plants early", "Control aphids"],
        ["Grow tolerant varieties", "Use reflective mulch or row covers early", "Control weeds near the field"]),
    "Squash___healthy": healthy("Squash", ["Use drip irrigation", "Scout for powdery mildew", "Rotate cucurbits"]),
    "Strawberry___Anthracnose": d(
        "Strawberry", "Anthracnose", "Fungal", "Colletotrichum acutatum / C. fragariae", "high",
        "A fungal disease spread by infected plants and splashing water, attacking fruit, crowns and leaves.",
        ["Dark sunken spots on fruit with orange spore masses", "Black spots on leaves and runners", "Crown rot that makes plants wilt"],
        ["Remove infected plants and fruit", "Apply a labelled fungicide during wet weather"],
        ["Plant certified disease-free plants", "Use drip irrigation and straw mulch", "Avoid working in wet beds"]),
    "Tobacco___Mosaic_virus": d(
        "Tobacco", "Tobacco Mosaic", "Viral", "Tobacco mosaic virus (TMV, tobamovirus)", "moderate",
        "A very stable virus spread by contact: hands, tools and infected plant debris.",
        ["Light and dark green mosaic pattern", "Leaf puckering and distortion", "Stunted growth"],
        ["There is no cure; remove infected plants", "Wash hands and disinfect tools"],
        ["Grow resistant varieties", "Sanitise tools and equipment", "Avoid handling plants after using tobacco products"]),
    "Tobacco___healthy": healthy("Tobacco", ["Sanitise tools", "Rotate crops", "Remove volunteer plants"]),
    "Tomato___Verticillium_wilt": d(
        "Tomato", "Verticillium Wilt", "Fungal", "Verticillium dahliae", "moderate",
        "A soil-borne wilt fungus that survives in soil for years and blocks the plant's water-conducting tissue.",
        ["V-shaped yellow patches on lower leaves", "Wilting during the hottest part of the day", "Brown streaks inside the lower stem"],
        ["There is no chemical cure; remove infected plants", "Keep plants watered to reduce stress"],
        ["Plant resistant varieties (marked 'V')", "Rotate away from solanaceous crops for several years", "Solarise soil in hot climates"]),
}

if __name__ == "__main__":
    kb = json.loads(P.read_text(encoding="utf-8"))
    added = [k for k in NEW if k not in kb]
    for k in added:
        kb[k] = NEW[k]
    P.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"added {len(added)} entries; knowledge base now has {len(kb)}")

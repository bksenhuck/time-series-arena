import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/raw/timeseries.db")
XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", "models/xgboost_model.pkl")
ARIMA_MODELS_PATH = os.getenv("ARIMA_MODELS_PATH", "models/arima_models.pkl")

# ── Server ────────────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", 8050))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_START_DATE = os.getenv("DATA_START_DATE", "2022-01-01")
DATA_END_DATE = os.getenv("DATA_END_DATE", "2024-12-31")

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_HORIZON_DEFAULT = int(os.getenv("FORECAST_HORIZON_DEFAULT", 14))
MIN_HORIZON = 7
MAX_HORIZON = 30

# ── Wikipedia pages ───────────────────────────────────────────────────────────
WIKIPEDIA_PAGES = [

# Technology
"Python","JavaScript","Machine_learning","Deep_learning","Artificial_intelligence",
"Neural_network","TensorFlow","GitHub","Linux","Android","IPhone","Google",
"Microsoft","Amazon","Apple","Tesla","Netflix","Spotify","Facebook","Bitcoin",
"Blockchain","Internet","World_Wide_Web","Cybersecurity","Cloud_computing",
"Quantum_computing","Docker","Kubernetes","React","TypeScript","OpenAI",
"Nvidia","Intel","Advanced_Micro_Devices","Samsung","Oracle_Corporation",
"Adobe","Salesforce","PayPal","Stripe","Block,_Inc.","Shopify","EBay","Uber",
"Airbnb","Lyft","DoorDash","Zoom","Slack","Notion","Figma","Canva","MercadoLibre",

# Programming Languages
"Java","C","C++","C_Sharp","Rust","Go","Swift","Kotlin","PHP","Ruby","Scala",
"R","Julia","MATLAB","Dart","Groovy",

# AI / ML
"ChatGPT","Generative_artificial_intelligence","Large_language_model",
"Transformer","Stable_Diffusion","DALL-E","Midjourney","Computer_vision",
"Natural_language_processing","Reinforcement_learning",

# Programming Tools
"Stack_Overflow","Git","Visual_Studio_Code","PyTorch","Jupyter","Anaconda",
"NumPy","Pandas","Scikit-learn","Matplotlib",

# Science
"Physics","Chemistry","Biology","Mathematics","Astronomy","Quantum_mechanics",
"Theory_of_relativity","Evolution","DNA","Human_genome","Climate_change",
"COVID-19_pandemic","Vaccine","Black_hole","Mars","Moon","Sun","Earth",
"Universe","Periodic_table","Photosynthesis","Gravity","Atom","Molecule",
"Cell","Genetics","Neuroscience","Nanotechnology","CRISPR","Exoplanet",

# Finance / Economy
"Stock_market","Inflation","Interest_rate","Federal_Reserve","Nasdaq",
"Dow_Jones_Industrial_Average","S&P_500","Cryptocurrency","Ethereum","Dogecoin",

# Pop Culture / Movies / TV
"The_Beatles","Michael_Jackson","Taylor_Swift","Eminem","Beyonce",
"Leonardo_DiCaprio","Avengers:_Endgame","Star_Wars","Harry_Potter",
"Game_of_Thrones","Breaking_Bad","The_Office","Friends",
"SpongeBob_SquarePants","Minecraft","Fortnite","The_Simpsons","Batman",
"Family_Guy","The_Walking_Dead","Stranger_Things","The_Mandalorian",
"Superman","Spider-Man","The_Dark_Knight","Titanic","Jurassic_Park",
"The_Lord_of_the_Rings","Marvel_Cinematic_Universe","Disney","Pixar",
"Nintendo","PlayStation","Xbox","Grand_Theft_Auto_V",

# Games
"League_of_Legends","Counter-Strike","Dota_2","Valorant","World_of_Warcraft",
"Call_of_Duty","Elden_Ring","The_Legend_of_Zelda","Pokemon",

# Social Platforms
"WhatsApp","Instagram","TikTok","YouTube","Telegram","Signal","Reddit",
"Discord","Twitch","Pinterest","Snapchat","LinkedIn","Quora",

# Sports (General)
"Football","Association_football","Basketball","Tennis","Golf","Baseball",
"Cricket","Rugby","Ice_hockey","Motorsport",

# Major Sports Competitions
"FIFA_World_Cup","UEFA_Champions_League","UEFA_European_Championship",
"Copa_America","Olympic_Games","Summer_Olympic_Games","Winter_Olympic_Games",
"Paralympic_Games","Super_Bowl","Tour_de_France",

# Motorsport
"Formula_One","IndyCar","NASCAR","24_Hours_of_Le_Mans","Monaco_Grand_Prix",
"Daytona_500","World_Rally_Championship","MotoGP",

# Major Leagues
"NBA","NFL","Major_League_Baseball","Premier_League","La_Liga","Serie_A",
"Bundesliga",

# Major Tournaments
"Wimbledon","US_Open_(tennis)","French_Open","Australian_Open",

# Famous Athletes
"Lionel_Messi","Cristiano_Ronaldo","LeBron_James","Roger_Federer",
"Rafael_Nadal","Novak_Djokovic","Serena_Williams","Michael_Jordan",
"Usain_Bolt","Tiger_Woods","Lewis_Hamilton","Ayrton_Senna",
"Michael_Schumacher","Neymar","Kylian_Mbappe","Pele",

# History
"World_War_II","World_War_I","American_Civil_War","French_Revolution",
"Roman_Empire","Ancient_Egypt","Cold_War","Adolf_Hitler","Napoleon",
"Abraham_Lincoln","United_States","United_Kingdom","China","Russia",
"India","Ancient_Greece","Byzantine_Empire","Ottoman_Empire",
"British_Empire","Mongol_Empire","Renaissance","Industrial_Revolution",
"American_Revolution","Russian_Revolution","Alexander_the_Great",
"Julius_Caesar","Genghis_Khan","Cleopatra","Queen_Victoria",
"Winston_Churchill",

# Geography
"New_York_City","London","Paris","Tokyo","Los_Angeles","Chicago",
"Sydney","Toronto","Berlin","Rome","Beijing","Mumbai","Dubai",
"Singapore","Amsterdam","Barcelona","Seoul","Istanbul","Cairo",
"Mexico_City",

# Food
"Pizza","Chocolate","Coffee","Tea","Sushi","Bread","Wine","Beer",
"Hamburger","Ice_cream","Pasta","Cheese","Curry","Ramen","Avocado",

# Music
"Rock_music","Pop_music","Hip_hop_music","Jazz","Classical_music",
"Electronic_music","Country_music","Blues","Reggae","Punk_rock",
"Heavy_metal_music","Rhythm_and_blues","Opera","Beethoven","Mozart",
"Bob_Dylan","Elvis_Presley","David_Bowie","Freddie_Mercury",
"Tupac_Shakur",

# Notable People
"Albert_Einstein","Isaac_Newton","Charles_Darwin","Nikola_Tesla",
"Stephen_Hawking","Marie_Curie","Elon_Musk","Bill_Gates","Steve_Jobs",
"Jeff_Bezos","Barack_Obama","Donald_Trump","Mahatma_Gandhi",
"Nelson_Mandela","Martin_Luther_King_Jr.","Sigmund_Freud","Karl_Marx",
"Friedrich_Nietzsche","Aristotle","Socrates"
]

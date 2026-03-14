"""
Mock Twitter Dataset Generator for Geospatial Temporal Feature
Generates tweets across multiple locations with full Twitter API v2 fields.
Includes ALL 10 geo-aggregation edge cases for thorough testing.

CASE MAP:
  Case 1  → Berlin       : 1 tweet only             → insufficient_data
  Case 2  → Sydney       : 5 tweets, SAME author    → suspicious_single_source
  Case 3  → Mumbai       : 4 tweets, diff authors   → early_signal
  Case 4  → Houston/SF/Miami/London/Tokyo: 54-55 tweets each → confirmed_disaster
  Case 5  → Istanbul     : 30 tweets, mixed labels  → ambiguous
  Case 6  → Paris        : 25 tweets, mostly casual  → false_alarm (no_disaster)
  Case 7  → Jakarta      : 20 tweets, ambiguous text → low_confidence
  Case 8  → Manila       : 25 tweets, ALL within 2h  → active_event (temporal burst)
  Case 9  → Mexico City  : 30 tweets, spread 5 days  → recovery_phase
  Case 10 → Seoul        : 15 tweets, conflicting locs → location_uncertain
"""
import json
import random
import uuid
import requests
from datetime import datetime, timedelta

BACKEND_URL = "http://localhost:5000/api"

# ===========================================================================
# HELPER: author pool generator
# ===========================================================================
def _authors(prefix, n=15):
    """Generate n random author tuples for a location."""
    return [(f"{prefix}_{i}", f"{prefix.replace('_',' ').title()} {i}") for i in range(1, n + 1)]

# ===========================================================================
# CASE 4 — CONFIRMED DISASTER  (5 large locations, ~54 tweets each)
# ===========================================================================
CONFIRMED_LOCATIONS = {
    "Houston, Texas": {
        "place_tag": "Houston, TX", "country": "United States",
        "coords": (29.7604, -95.3698), "disaster": "flooding",
        "authors": _authors("htx", 15),
        "tweets": [
            # Affected Individuals (0)
            "Flooding in downtown Houston has left 15 people injured and 3 missing. Emergency crews are searching.",
            "My neighbor's family of 6 is stranded on their rooftop in Houston. Water rising fast, please help!",
            "At least 200 families displaced by Houston flooding. Many elderly residents unable to evacuate.",
            "8 people rescued from submerged vehicles on I-45 in Houston. 2 children among those saved.",
            "Houston flood waters have trapped dozens of residents in Meyerland area. People screaming for help.",
            "Confirmed 4 dead, 22 injured in Houston flooding disaster. Numbers expected to rise.",
            "Family of 5 found clinging to tree in Houston floodwaters. Children aged 2 and 5 among them.",
            "Houston nursing home flooded - 45 elderly residents awaiting evacuation. Staff doing their best.",
            "12 people still unaccounted for in southeast Houston after flash flooding hit overnight.",
            "Pregnant woman rescued from flooded apartment complex in Houston. Taken to hospital immediately.",
            "Over 500 residents evacuated from Houston's Third Ward. Many lost everything in the flooding.",
            # Infrastructure Damage (1)
            "Major bridge on Highway 59 in Houston has collapsed due to flooding. All lanes closed.",
            "Houston power grid failure - 50,000 homes without electricity as floodwaters damage substations.",
            "Water treatment plant in Houston damaged by flooding. Boil water advisory issued.",
            "Houston Metro rail system completely shut down. Tracks submerged under 4 feet of water.",
            "Multiple schools in Houston ISD have structural damage from flooding. Buildings deemed unsafe.",
            "Gas pipeline ruptured in Houston due to flood erosion. Fire department on scene, area evacuated.",
            "Houston Hobby Airport runway flooded - all flights cancelled for at least 48 hours.",
            "Sewage system overflow in Houston after heavy flooding. Contaminated water in streets.",
            "Buffalo Bayou has breached its banks in Houston destroying several commercial buildings.",
            "Cell towers damaged across Houston metro area. Communications severely disrupted.",
            "Houston freeway system completely impassable. I-10 and I-45 both underwater.",
            # Not Humanitarian (2)
            "Just saw a Houston Texans game highlight reel. What a season they're having!",
            "Best BBQ in Houston? Gotta be Killen's BBQ hands down. Amazing brisket!",
            "Houston weather is always so unpredictable lol. One day sunny, next day rain.",
            "Anyone know good restaurants near Houston Galleria? Looking for dinner recommendations.",
            "Traffic in Houston is always terrible even without flooding haha. Love this city though.",
            "Houston Astros looking strong this preseason. Can't wait for opening day!",
            "Moving to Houston next month for a new job. Any neighborhood recommendations?",
            "Houston has the best food trucks I've ever seen. Taco game is strong.",
            "My Houston vacation was amazing! Space Center was the highlight of the trip.",
            "Houston real estate prices are wild these days. Market is crazy competitive.",
            # Other Information (3)
            "Weather update: Heavy rain expected to continue in Houston for the next 72 hours.",
            "Houston emergency management has activated all disaster response protocols.",
            "National Guard has been deployed to Houston area. Military vehicles arriving.",
            "Houston Mayor press conference at 3 PM today regarding flood emergency declaration.",
            "FEMA has declared Houston a federal disaster area. Federal aid on its way.",
            "Houston flood waters contain dangerous bacteria. Health officials warn against contact.",
            "Evacuation routes from Houston: Take US-290 North or I-10 West. Avoid I-45.",
            "Houston shelters reaching capacity. New shelter at George R. Brown Convention Center.",
            "Insurance companies expect Houston flood claims to exceed $2 billion.",
            "Houston schools closed for remainder of the week due to flooding emergency.",
            "Curfew in effect for Houston from 10 PM to 6 AM until further notice.",
            # Rescue/Donation (4)
            "Volunteers needed for Houston flood rescue operations! Boats needed urgently!",
            "Donating medical supplies to Houston flood victims. Drop-off at Minute Maid Park.",
            "Houston food drive organized for displaced families. Non-perishable items needed.",
            "Red Cross Houston chapter needs blood donations urgently. All blood types needed.",
            "Free temporary housing available for Houston flood victims. Contact 832-555-0100.",
            "Search and rescue teams from Dallas heading to Houston to assist.",
            "Houston animal rescue - volunteers needed to save pets stranded by floodwaters.",
            "Medical volunteers needed at NRG Center in Houston. Doctors and nurses please contact Red Cross.",
            "Corporate donations pouring in for Houston relief. Amazon donating $5 million.",
            "Water purification tablets being distributed free at Houston community centers.",
            "Kayak and canoe owners in Houston area - your help is desperately needed for rescues!",
        ]
    },
    "San Francisco, California": {
        "place_tag": "San Francisco, CA", "country": "United States",
        "coords": (37.7749, -122.4194), "disaster": "earthquake",
        "authors": _authors("sf", 15),
        "tweets": [
            "10 people trapped in collapsed building in San Francisco after 6.5 magnitude earthquake.",
            "Elderly couple pulled from rubble in SF Mission District. Both critically injured.",
            "San Francisco earthquake has injured over 45 people so far. Hospitals overwhelmed.",
            "Family of 4 missing after apartment building collapsed in San Francisco earthquake.",
            "15 construction workers trapped in partially collapsed parking garage in downtown SF.",
            "San Francisco residents fleeing damaged buildings. Many have nowhere to go tonight.",
            "3 children orphaned after parents killed in San Francisco earthquake building collapse.",
            "At least 8 dead in San Francisco earthquake. Search and rescue ongoing.",
            "200+ people sleeping in Golden Gate Park after homes damaged in SF earthquake.",
            "Wheelchair-bound residents trapped on upper floors of damaged SF apartment building.",
            "Tourist group of 12 trapped in Fisherman's Wharf after road collapse.",
            "Bay Bridge severely damaged in San Francisco earthquake. Major cracks on upper deck.",
            "San Francisco's Millennium Tower tilting further after earthquake.",
            "BART system completely shut down after earthquake. Tunnels under inspection.",
            "Gas mains ruptured across San Francisco. Multiple fires in Marina District.",
            "San Francisco water supply disrupted. Hetch Hetchy pipeline damaged.",
            "Golden Gate Bridge closed for inspection after earthquake. Structural concerns.",
            "MUNI transit system offline. Overhead wires down throughout San Francisco.",
            "San Francisco hospital wing collapsed. Patients being relocated.",
            "Victorian homes in SF Haight-Ashbury severely damaged. Historic structures at risk.",
            "PG&E reports catastrophic damage to SF power grid. Weeks to restore.",
            "San Francisco airport control tower damaged. Flights diverted to Oakland.",
            "San Francisco has the best sourdough bread in the world. Life changing!",
            "Beautiful sunset over Golden Gate Bridge today. San Francisco never disappoints!",
            "Tech industry in San Francisco is booming again. So many startups hiring.",
            "Visited Alcatraz today on my San Francisco trip. Incredible historical site.",
            "San Francisco Giants spring training looks promising. Go Giants!",
            "Cable cars in San Francisco are so iconic. Every tourist must try them.",
            "San Francisco fog rolled in again. Classic Karl the Fog moment.",
            "Best coffee shops in San Francisco? I need recommendations.",
            "The food scene in San Francisco is unmatched. Dim sum in Chinatown was incredible.",
            "San Francisco real estate just hit new highs. Average home $1.5 million.",
            "Hiking in Marin Headlands near San Francisco. Views are stunning!",
            "USGS confirms 6.5 magnitude earthquake centered 12 miles SE of San Francisco.",
            "San Francisco declares state of emergency following devastating earthquake.",
            "Aftershocks expected in SF for 48 hours. 4.2 recorded 20 mins ago.",
            "San Francisco earthquake worst to hit Bay Area since 1989 Loma Prieta.",
            "Tsunami warning issued for San Francisco Bay following major earthquake.",
            "SF Mayor announces emergency press conference at City Hall at 5 PM.",
            "Seismologists warn SF could experience 5+ magnitude aftershocks tonight.",
            "Emergency shelters opening across SF. Check sf.gov for nearest location.",
            "Cell service intermittent across San Francisco. Use WiFi calling.",
            "San Francisco earthquake insurance claims expected to top $10 billion.",
            "National Guard mobilizing for SF earthquake response. Troops arriving tonight.",
            "Urban search and rescue teams deployed across San Francisco. K-9 units on scene.",
            "Blood drive at SF General Hospital for earthquake victims. Walk-ins welcome 24/7.",
            "Volunteers needed to sort donations at Moscone Center in San Francisco.",
            "Structural engineers volunteering to assess SF buildings for safety.",
            "Free meals at Dolores Park for SF earthquake displaced residents.",
            "Tech companies donating laptops and charging stations for SF shelters.",
            "Mental health counselors volunteering at SF shelters. Trauma support available.",
            "Heavy equipment operators needed for rubble clearing in downtown SF.",
            "GoFundMe for SF earthquake victims has raised $3 million in 24 hours.",
            "Emergency pet shelters open in SF. SPCA accepting displaced animals.",
            "Doctors Without Borders sending team to assist SF earthquake response.",
        ]
    },
    "Miami, Florida": {
        "place_tag": "Miami, FL", "country": "United States",
        "coords": (25.7617, -80.1918), "disaster": "hurricane",
        "authors": _authors("mia", 15),
        "tweets": [
            "Hurricane has devastated Miami - 25 injured, 2 confirmed dead in Miami Beach.",
            "Family trapped in collapsed home in Miami Gardens during hurricane. Roof caved in.",
            "Over 1000 Miami residents in emergency shelters tonight. Many lost everything.",
            "Elderly man found dead in flooded Miami apartment. Power out for 3 days.",
            "30 tourists stranded at Miami International Airport. Building partially damaged.",
            "Miami hospital reporting surge of hurricane injuries. 60+ patients in ER.",
            "Homeless population in Miami severely affected. 150 people unaccounted for.",
            "Mother and baby rescued from floodwaters in Little Havana, Miami. Stable condition.",
            "8 fishermen missing off Miami coast after hurricane. Coast Guard searching.",
            "Miami trailer park completely destroyed. 200 displaced, 15 injured.",
            "Children's school in Miami collapsed during hurricane. Evacuated hours before.",
            "Miami's Art Deco buildings on Ocean Drive severely damaged by hurricane winds.",
            "Miami causeway connecting Beach to mainland impassable. Debris covering road.",
            "PortMiami completely shut down. Cruise ships diverted. Massive damage to terminal.",
            "Miami power grid devastated - 300,000 homes without electricity.",
            "Venetian Causeway in Miami partially collapsed into Biscayne Bay.",
            "Miami Beach seawall breached. Ocean water flooding neighborhoods.",
            "Miami MetroMover completely offline. Elevated tracks damaged.",
            "Storm surge destroyed Miami marina. Over 100 boats sunk.",
            "Miami water treatment facility damaged. Boil water notice issued.",
            "Multiple cranes collapsed at Miami construction sites. Roads blocked.",
            "Miami Dolphins stadium roof torn off by winds. Debris scattered.",
            "Miami nightlife is absolutely incredible. Best clubs in the country.",
            "Just had the best Cuban coffee in Little Havana, Miami. Cafecito is life!",
            "Miami Heat looking good this season. Jimmy Butler playing amazing.",
            "Beach volleyball tournament in Miami next weekend. Can't wait!",
            "Miami sunset from South Beach tonight was breathtaking. No filter needed.",
            "Shopping at Aventura Mall in Miami today. Great stores and restaurants.",
            "Miami Art Basel was incredible this year. Mind-blowing installations.",
            "Best seafood in Miami? Joe's Stone Crab never disappoints.",
            "Driving through the Florida Keys from Miami. Most scenic drive in America.",
            "Miami Dolphins draft picks looking solid this year. Optimistic!",
            "Category 4 hurricane making landfall near Miami. 145+ mph winds.",
            "Miami-Dade County under mandatory evacuation order. Leave Zone A.",
            "Hurricane eye passing over downtown Miami. Brief calm before second half.",
            "Miami emergency says worst hurricane since Andrew in 1992.",
            "Storm surge warning: Miami could see 10-15 foot surge. Life-threatening.",
            "FEMA deploying to Miami. Pre-positioned supplies in staging areas.",
            "Miami schools and government offices closed until further notice.",
            "Gas stations across Miami running out of fuel. Long lines.",
            "Miami hurricane damage assessment begins at first light tomorrow.",
            "Insurance adjusters arriving in Miami. Damage estimates in billions.",
            "Florida Governor declares state of emergency for Miami-Dade.",
            "Coast Guard helicopter rescues 12 from flooded Miami rooftops.",
            "Donations needed for Miami victims. Water, canned food, baby supplies.",
            "Miami Dolphins opening Hard Rock Stadium as shelter. 5000 capacity.",
            "Volunteer boat rescuers needed in Miami. Report to Coconut Grove Marina.",
            "Red Cross setting up 15 emergency shelters across Miami-Dade.",
            "Free generator distribution for Miami residents at Tropical Park.",
            "Medical volunteers urgently needed at Jackson Memorial Hospital.",
            "Celebrity chefs cooking meals for Miami hurricane victims. Thousands daily.",
            "Uber and Lyft offering free rides to Miami shelters. Code MIAMIHURRICANE.",
            "Tree removal volunteers needed across Miami. Chainsaws welcome.",
            "Target donating $10 million in supplies to Miami hurricane relief.",
        ]
    },
    "London, United Kingdom": {
        "place_tag": "London, England", "country": "United Kingdom",
        "coords": (51.5074, -0.1278), "disaster": "flooding",
        "authors": _authors("ldn", 15),
        "tweets": [
            "Thames flooding displaced 500 families in east London. Shelters overwhelmed.",
            "Elderly woman found dead in flooded basement flat in Bermondsey, London.",
            "20 people rescued from London Underground station flooded by Thames overflow.",
            "London care home evacuated after flooding. 60 vulnerable residents moved.",
            "Family of 7 stranded on upper floor of flooded London home for 2 days.",
            "At least 12 people injured in London flooding. Several critical.",
            "Homeless community along London's South Bank severely hit. 30 missing.",
            "London school children trapped by floodwaters on school bus.",
            "Pregnant woman airlifted from flooded London neighbourhood.",
            "Over 2000 London residents sleeping in emergency accommodation tonight.",
            "Disabled residents unable to evacuate London tower block. Lifts flooded.",
            "Thames Barrier overwhelmed - London facing worst flooding in history.",
            "London Underground Northern Line completely flooded. Suspended.",
            "Tower Bridge mechanical failure after flooding damage. Stuck raised.",
            "London electricity grid failing. Canary Wharf without power.",
            "Westminster Bridge crumbling due to flood erosion. Closed.",
            "London sewer system overwhelmed. Sewage flooding streets.",
            "Heathrow Terminal 2 flooded. Hundreds of flights cancelled.",
            "London hospitals reporting flooding in lower levels.",
            "London Crossrail Elizabeth Line tunnels flooded. Suspended.",
            "Historic London buildings along Thames embankment suffering water damage.",
            "O2 Arena roof damaged by storm winds. Events cancelled.",
            "Just had amazing fish and chips in London. Borough Market never disappoints.",
            "London theatre scene incredible. Saw Hamilton in West End - brilliant!",
            "Arsenal match at Emirates Stadium was phenomenal tonight.",
            "Beautiful walk along the Thames today. Gorgeous in any weather.",
            "London has the best museums. British Museum and National Gallery are free!",
            "Finding a flat in London on a budget is basically impossible lol.",
            "London coffee scene has really improved. Amazing cafe in Shoreditch.",
            "Big Ben looking beautiful today. Restoration work was worth it.",
            "London's Chinatown has the best dim sum outside of Asia.",
            "Planning London trip. Any hotel recommendations near Covent Garden?",
            "Met Office issues red weather warning for London. Unprecedented rainfall.",
            "London Mayor declares major incident due to widespread flooding.",
            "Environment Agency warns Thames could reach record levels tomorrow.",
            "London 999 receiving 10x normal call volume due to flooding.",
            "Army Reserve called up for London flood response.",
            "London public transport south of Thames suspended.",
            "COBRA emergency committee meeting at Downing Street re London flooding.",
            "London flood insurance claims estimated at 5 billion pounds.",
            "Thames Barrier emergency closure tonight. Defences at max capacity.",
            "London schools across 12 boroughs closed tomorrow.",
            "BBC News: London flooding is worst UK natural disaster since 2007.",
            "RNLI deployment across London - 50 boats patrolling flooded areas.",
            "Donations needed for London victims. Blankets at any fire station.",
            "London sandbag filling stations in every borough. Help needed.",
            "British Red Cross opening 20 rest centres across London.",
            "Free hot meals at London community centres in all affected boroughs.",
            "London Fire Brigade needs volunteer boat operators. Lambeth HQ.",
            "Water and food airlifted to stranded London residents by RAF.",
            "Mental health helpline for London flood victims. 0800-555-0199.",
            "London businesses donating stock to relief. Collection at every Tesco.",
            "St John Ambulance deploying 200 first aiders across London.",
            "Shelter charity providing emergency housing for London displaced families.",
        ]
    },
    "Tokyo, Japan": {
        "place_tag": "Tokyo, Japan", "country": "Japan",
        "coords": (35.6762, 139.6503), "disaster": "earthquake and tsunami",
        "authors": _authors("tky", 15),
        "tweets": [
            "Massive earthquake in Tokyo - 30 people trapped in collapsed buildings in Shibuya.",
            "Tokyo hospital reporting 100+ injuries from earthquake. ERs at full capacity.",
            "Family of 5 missing after building collapse in Tokyo Shinjuku district.",
            "Elderly residents in Tokyo nursing home trapped. 40 people need rescue.",
            "Tsunami warning: Tokyo Bay coastal residents urgently evacuated. 10,000 at risk.",
            "15 workers trapped in underground mall in Tokyo Station after earthquake.",
            "Death toll in Tokyo earthquake rises to 25. Hundreds unaccounted for.",
            "Tokyo school evacuated before earthquake collapsed gymnasium roof.",
            "Foreign tourists in Tokyo unable to reach embassies. 500+ stranded at Narita.",
            "Children separated from parents during Tokyo evacuation. Reunion at Yoyogi Park.",
            "Over 5000 Tokyo residents in emergency shelters. Only clothes on their backs.",
            "Tokyo Tower observation deck damaged. Structure leaning, area evacuated.",
            "Tokyo Skytree closed indefinitely. Engineers assessing structural integrity.",
            "Shinkansen bullet train derailed near Tokyo Station. Tracks damaged.",
            "Tokyo expressway collapsed in multiple sections. Vehicles stranded.",
            "Massive power outage across Tokyo. 2 million homes without electricity.",
            "Tokyo water mains burst across Minato district. Streets impassable.",
            "Odaiba waterfront severely damaged by tsunami surge. Rainbow Bridge closed.",
            "Tokyo subway system completely shut down. All 13 lines suspended.",
            "Gas leaks across central Tokyo. Explosion risk in multiple areas.",
            "Tokyo Tsukiji area flooded by tsunami surge. Market buildings destroyed.",
            "NTT communications hub damaged. Internet and phone disrupted nationwide.",
            "Tokyo ramen is in a league of its own. Best bowl ever in Ikebukuro!",
            "Cherry blossom season starting early in Tokyo. Ueno Park will be gorgeous.",
            "Shibuya crossing is still the coolest intersection in the world.",
            "Tokyo Disneyland was absolutely magical. Planning return trip!",
            "Shopping in Harajuku, Tokyo today. Fashion here so creative.",
            "Sushi breakfast at Tsukiji outer market. Life changing honestly.",
            "Tokyo public transport is the most efficient in the world.",
            "Akihabara in Tokyo is a technology paradise. Amazing electronics deals.",
            "Beautiful temple visit in Asakusa. Senso-ji is absolutely stunning.",
            "Tokyo nightlife in Roppongi is incredible. Best bars in Asia!",
            "JMA reports 7.2 magnitude earthquake with epicenter 30km south of Tokyo Bay.",
            "Japanese PM declares national emergency following Tokyo earthquake.",
            "Tsunami warning: 3-5 meter waves expected along Tokyo Bay within 30 mins.",
            "Multiple aftershocks hitting Tokyo. 5.1 magnitude tremor 15 mins ago.",
            "Tokyo earthquake strongest to hit capital since Great Kanto earthquake.",
            "Japan Self-Defense Forces fully mobilized for Tokyo response.",
            "All international flights to Tokyo cancelled. Haneda and Narita closed.",
            "Tokyo government estimates damage could exceed 10 trillion yen.",
            "Nuclear plants near Tokyo reporting normal operation.",
            "Tokyo residents advised to prepare emergency kits. More aftershocks expected.",
            "Foreign governments issuing travel warnings for Tokyo.",
            "Japan Red Cross launching massive rescue across Tokyo. 1000 volunteers.",
            "International rescue teams from 15 countries heading to Tokyo.",
            "Tokyo residents: free food and water at all ward offices now.",
            "Japanese tech companies donating emergency communication equipment.",
            "Blood donations urgently needed for Tokyo victims. All hospitals accepting.",
            "Volunteer interpreters helping foreign nationals in Tokyo shelters.",
            "Navy ships with supplies arriving at Tokyo Bay for relief.",
            "Mobile medical clinics in Tokyo parks for earthquake victims.",
            "SoftBank and Docomo providing free WiFi and charging at all Tokyo shelters.",
            "Construction companies volunteering equipment for rubble clearing in Tokyo.",
            "Global donations for Tokyo relief exceed $100 million in 24 hours.",
        ]
    },
}

# ===========================================================================
# EDGE CASE LOCATIONS  (Cases 1, 2, 3, 5, 6, 7, 8, 9, 10)
# ===========================================================================
EDGE_CASE_LOCATIONS = {
    # -----------------------------------------------------------------------
    # CASE 1 — INSUFFICIENT DATA: Only 1 tweet from this location
    # -----------------------------------------------------------------------
    "Berlin, Germany": {
        "case": 1, "expected_status": "insufficient_data",
        "place_tag": "Berlin, Germany", "country": "Germany",
        "coords": (52.52, 13.405),
        "authors": [("berlin_watcher", "Berlin Watcher")],
        "tweets": [
            {"text": "Explosion reported near Berlin Hauptbahnhof. People running. Huge smoke cloud visible.",
             "force_single_author": True},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 2 — SUSPICIOUS SINGLE SOURCE: 5 tweets, ALL from the same author
    # -----------------------------------------------------------------------
    "Sydney, Australia": {
        "case": 2, "expected_status": "suspicious_single_source",
        "place_tag": "Sydney, NSW", "country": "Australia",
        "coords": (-33.8688, 151.2093),
        "authors": [("sydney_spammer", "Sydney Breaking News")],
        "tweets": [
            {"text": "BREAKING: Massive bushfire engulfing western Sydney suburbs. Hundreds of homes on fire!", "force_single_author": True},
            {"text": "Sydney bushfire update: Entire Penrith area evacuated. Fire out of control!", "force_single_author": True},
            {"text": "Sydney fire department overwhelmed. 500 firefighters deployed but not enough!", "force_single_author": True},
            {"text": "Sydney bushfire: 50 people confirmed dead. Worst disaster in Australian history!", "force_single_author": True},
            {"text": "Nuclear meltdown at Sydney research reactor due to bushfire. Radiation spreading!", "force_single_author": True},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 3 — EARLY SIGNAL: 4 tweets from different authors
    # -----------------------------------------------------------------------
    "Mumbai, India": {
        "case": 3, "expected_status": "early_signal",
        "place_tag": "Mumbai, India", "country": "India",
        "coords": (19.076, 72.8777),
        "authors": [("mumbai_1", "Mumbai Citizen 1"), ("mumbai_2", "Mumbai Citizen 2"),
                    ("mumbai_3", "Mumbai Citizen 3"), ("mumbai_4", "Mumbai Citizen 4")],
        "tweets": [
            {"text": "Heavy flooding in Dharavi, Mumbai. Water entering homes. Looks serious.", "author_idx": 0},
            {"text": "Mumbai local trains suspended due to waterlogging at Dadar station.", "author_idx": 1},
            {"text": "Roads completely flooded near Mumbai CST station. Cars stuck in water.", "author_idx": 2},
            {"text": "My area in Andheri, Mumbai is flooded. Never seen water this high before.", "author_idx": 3},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 5 — AMBIGUOUS: 30 tweets with heavily MIXED classifications
    # -----------------------------------------------------------------------
    "Istanbul, Turkey": {
        "case": 5, "expected_status": "ambiguous_needs_review",
        "place_tag": "Istanbul, Turkey", "country": "Turkey",
        "coords": (41.0082, 28.9784),
        "authors": _authors("ist", 12),
        "tweets": [
            # Mix of all categories — no clear majority
            {"text": "Small earthquake felt in Istanbul. Some items fell off shelves but nothing major."},
            {"text": "Istanbul earthquake: one old wall in Fatih area has cracked. Residents concerned."},
            {"text": "Was that an earthquake in Istanbul? My coffee cup shook for a second."},
            {"text": "Istanbul street food is the best in the world. Balik ekmek by the Bosphorus!"},
            {"text": "Beautiful evening in Istanbul. The mosques look stunning at sunset."},
            {"text": "Istanbul traffic is crazy today. Nothing new though haha."},
            {"text": "Grand Bazaar in Istanbul is a must-visit. Shopping paradise!"},
            {"text": "Minor tremor reported in Istanbul. Seismologists say no major concern."},
            {"text": "Istanbul residents sharing earthquake safety tips online. Good to be prepared."},
            {"text": "Some buildings in old Istanbul showing minor cracks after the tremor."},
            {"text": "Turkish authorities monitoring seismic activity near Istanbul."},
            {"text": "Volunteers in Istanbul organizing earthquake preparedness kits just in case."},
            {"text": "Istanbul Bosphorus cruise was amazing today. Perfect weather!"},
            {"text": "Hagia Sophia in Istanbul is breathtaking. Must see for every tourist."},
            {"text": "Cat cafes in Istanbul are the cutest thing ever. Cats everywhere!"},
            {"text": "Istanbul university cancelled classes for one day as a precaution."},
            {"text": "3 elderly people in Istanbul fell during minor tremor. Treated for minor injuries."},
            {"text": "Istanbul metro running normally after brief pause for earthquake inspection."},
            {"text": "Emergency services in Istanbul on standby but no major incidents reported."},
            {"text": "News says Istanbul earthquake was only 3.2 magnitude. No tsunami risk."},
            {"text": "Istanbul real estate prices are still climbing. Great investment city."},
            {"text": "Turkish tea and Turkish delight in Istanbul. What more do you need?"},
            {"text": "Galata Tower in Istanbul gives the best panoramic views of the city."},
            {"text": "Istanbul ferry service temporarily suspended for safety check. Resumed within an hour."},
            {"text": "Some anxiety among Istanbul residents but daily life continues normally."},
            {"text": "Istanbul hospitals report no surge in patients from earthquake."},
            {"text": "Neighbours in Istanbul checking on each other after minor tremor. Community spirit."},
            {"text": "Istanbul earthquake preparedness drill planned for next month."},
            {"text": "Donating old clothes and blankets in Istanbul just in case they're needed."},
            {"text": "Should I be worried about visiting Istanbul? The earthquake was tiny apparently."},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 6 — FALSE ALARM / NO DISASTER: 25 tweets, mostly casual/non-humanitarian
    # -----------------------------------------------------------------------
    "Paris, France": {
        "case": 6, "expected_status": "no_disaster",
        "place_tag": "Paris, France", "country": "France",
        "coords": (48.8566, 2.3522),
        "authors": _authors("par", 12),
        "tweets": [
            {"text": "Paris in springtime is absolutely magical. The Eiffel Tower never gets old."},
            {"text": "Best croissants in Paris? Paul bakery on Champs-Elysees is incredible."},
            {"text": "Paris Fashion Week was stunning this year. So many creative designs."},
            {"text": "Louvre Museum in Paris is worth spending an entire day at. Mona Lisa was smaller than expected though."},
            {"text": "Paris Metro is actually quite efficient. Got everywhere easily."},
            {"text": "Dinner at a rooftop restaurant in Paris overlooking the Seine. Unforgettable."},
            {"text": "Notre-Dame reconstruction in Paris is looking amazing. Almost complete."},
            {"text": "Wine and cheese in Montmartre, Paris. Living the dream."},
            {"text": "Paris street performers near Sacre-Coeur are so talented."},
            {"text": "Shakespeare and Company bookshop in Paris is a literary treasure."},
            {"text": "Rainy day in Paris. Still beautiful with umbrellas everywhere."},
            {"text": "Paris bridges at night are so romantic. Perfect city for couples."},
            {"text": "someone mentioned earthquake in Paris? it was just construction vibrations lol"},
            {"text": "Is Paris safe? Absolutely. Best trip of my life. Don't listen to the fear."},
            {"text": "Paris has such amazing public parks. Luxembourg Gardens are my favourite."},
            {"text": "Train strike in Paris today affecting some routes. Very typical honestly."},
            {"text": "Just arrived in Paris for holiday. Airport was smooth, city looks gorgeous."},
            {"text": "Paris marathon was incredible. Ran through the most beautiful streets in the world."},
            {"text": "Versailles is only 30min from Paris by train. Absolutely worth the day trip."},
            {"text": "Paris nightlife in Le Marais district. Great bars and people."},
            {"text": "Learning French in Paris is the best way to learn. Immersion is everything."},
            {"text": "Picnic by the Eiffel Tower in Paris. Baguette, cheese, wine. Perfection."},
            {"text": "Paris weather forecast: mostly sunny this week with mild temperatures."},
            {"text": "Small fire alarm at Paris hotel turned out to be burnt toast. All clear!"},
            {"text": "Visited Catacombs in Paris today. Creepy but fascinating history."},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 7 — LOW CONFIDENCE: 20 ambiguous tweets that models will struggle with
    # -----------------------------------------------------------------------
    "Jakarta, Indonesia": {
        "case": 7, "expected_status": "low_confidence_cluster",
        "place_tag": "Jakarta, Indonesia", "country": "Indonesia",
        "coords": (-6.2088, 106.8456),
        "authors": _authors("jkt", 10),
        "tweets": [
            # Deliberately ambiguous — could be disaster or metaphor or minor event
            {"text": "Jakarta is drowning again. When will they fix the drainage system?"},
            {"text": "So much water everywhere in Jakarta. Might need a boat to get to work lol."},
            {"text": "Jakarta streets are rivers today. Is this flooding or just bad drainage?"},
            {"text": "My Jakarta office basement has water. Not sure if it's a pipe burst or flooding."},
            {"text": "Think there was a tremor in Jakarta or maybe it was just a truck passing by."},
            {"text": "Jakarta ground shaking or am I imagining things? Third time this week."},
            {"text": "Power flickering in Jakarta. Storm related or grid issues? Can't tell."},
            {"text": "Jakarta roads are a disaster today. Rain or infrastructure problem?"},
            {"text": "Heard sirens in Jakarta but could be anything. Traffic accident maybe?"},
            {"text": "Water in Jakarta subway station. Leak or flooding? Nobody seems to know."},
            {"text": "Windows rattled in my Jakarta apartment. Earthquake or construction?"},
            {"text": "Jakarta air quality terrible today. Pollution or smoke from somewhere?"},
            {"text": "Something collapsed near Jakarta market. Could be construction related."},
            {"text": "Ambulance heading somewhere in Jakarta. Hope nobody is seriously hurt."},
            {"text": "Jakarta feeling warmer than usual. Climate change is real but is something happening?"},
            {"text": "SMS alert about Jakarta weather but I can't read Bahasa well. Anyone translate?"},
            {"text": "Jakarta canal overflowing a bit. Normal monsoon thing or something worse?"},
            {"text": "Kids in Jakarta wading through puddles. Cute but is this more serious?"},
            {"text": "Jakarta traffic completely stopped. Accident or road damage somewhere?"},
            {"text": "Helicopter circling over Jakarta. Emergency or just police routine?"},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 8 — ACTIVE EVENT (Temporal Burst): 25 tweets ALL within 2 hours
    # -----------------------------------------------------------------------
    "Manila, Philippines": {
        "case": 8, "expected_status": "active_event",
        "place_tag": "Manila, Philippines", "country": "Philippines",
        "coords": (14.5995, 120.9842),
        "authors": _authors("mnl", 15),
        "temporal_burst": True,  # Flag to generate all timestamps within 2 hours
        "tweets": [
            {"text": "JUST NOW: Massive typhoon hitting Manila! Wind is unbelievable!"},
            {"text": "Manila typhoon: roof just ripped off the building next door. Taking shelter!"},
            {"text": "9 people on our Manila street are injured. Flying debris everywhere. Need help NOW!"},
            {"text": "Manila underpass completely flooded in minutes. Cars submerging. Help!!"},
            {"text": "Tondo district in Manila underwater. Water rising to second floor in some areas."},
            {"text": "Manila hospital power gone. Emergency generators failing. Patients at risk!"},
            {"text": "Manila bridge cracking from storm surge. Crossing is suicide right now."},
            {"text": "Entire Manila Bay waterfront destroyed within last hour. Never seen anything like this."},
            {"text": "Communication towers down across Manila. Can barely send this tweet."},
            {"text": "Manila airport runway underwater. All flights cancelled effective immediately."},
            {"text": "Manila coastal areas evacuated but some people refused to leave. Now trapped!"},
            {"text": "Trees falling everywhere in Manila. Power lines down. Stay indoors!"},
            {"text": "Manila Red Cross activating all emergency teams RIGHT NOW. Volunteers report immediately!"},
            {"text": "Philippine Navy deploying rescue boats to Manila Bay area as we speak."},
            {"text": "Manila shelters filling up fast. Convention Center accepting evacuees now."},
            {"text": "Free rescue boats heading to Manila Tondo area. If you need help signal from windows!"},
            {"text": "Food and water being distributed at Manila City Hall. Come if you can reach it."},
            {"text": "Manila typhoon: PAGASA says eye wall passing directly over city center now."},
            {"text": "Manila mayor declaring state of calamity effective immediately."},
            {"text": "Philippine president addressing nation about Manila typhoon within the hour."},
            {"text": "All Manila schools closed until further notice. Government offices too."},
            {"text": "Manila death toll climbing: at least 15 confirmed in last 2 hours alone."},
            {"text": "Rescue helicopters unable to fly in Manila due to extreme winds. Ground rescue only."},
            {"text": "Manila children's hospital flooded. 30 pediatric patients being moved to upper floors."},
            {"text": "Praying for everyone in Manila right now. This typhoon is like nothing we've seen."},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 9 — RECOVERY PHASE: 30 tweets spread over 5 days showing transition
    # -----------------------------------------------------------------------
    "Mexico City, Mexico": {
        "case": 9, "expected_status": "recovery_phase",
        "place_tag": "Mexico City, Mexico", "country": "Mexico",
        "coords": (19.4326, -99.1332),
        "authors": _authors("mxc", 12),
        "temporal_spread_days": 5,  # Spread over 5 days
        "tweets": [
            # Day 1 — Active disaster (Affected Individuals + Infrastructure)
            {"text": "Powerful earthquake just hit Mexico City! Buildings swaying. People in panic!", "day": 0},
            {"text": "Mexico City earthquake: 20 people trapped in collapsed CDMX apartment building.", "day": 0},
            {"text": "Mexico City hospitals overwhelmed with earthquake injuries. 80+ patients.", "day": 0},
            {"text": "Mexico City Metro Line 3 tunnel partially collapsed. Passengers evacuated.", "day": 0},
            {"text": "Gas explosions reported in Mexico City Condesa neighbourhood after earthquake.", "day": 0},
            {"text": "Mexico City death toll from earthquake reaches 18. Search continues.", "day": 0},
            # Day 2 — Still active but rescue ramping up
            {"text": "Day 2 of Mexico City earthquake rescue. Still pulling survivors from collapsed buildings.", "day": 1},
            {"text": "Mexican army setting up field hospitals across Mexico City for earthquake victims.", "day": 1},
            {"text": "Volunteers flooding into Mexico City to help with earthquake rescue. Beautiful to see.", "day": 1},
            {"text": "Mexico City water supply partially restored in 4 of 16 delegaciones.", "day": 1},
            {"text": "130 aftershocks recorded in Mexico City since the main earthquake.", "day": 1},
            {"text": "International rescue teams arriving in Mexico City from USA, Japan, and Spain.", "day": 1},
            # Day 3 — Transition: less rescue, more logistics + recovery
            {"text": "Mexico City earthquake rescue operations transitioning to recovery phase.", "day": 2},
            {"text": "Temporary shelters in Mexico City expanding. 5000 people still displaced.", "day": 2},
            {"text": "Mexico City structural engineers declaring which buildings are safe to re-enter.", "day": 2},
            {"text": "Schools in Mexico City beginning damage assessment for eventual reopening.", "day": 2},
            {"text": "Mental health support teams deployed across Mexico City earthquake shelters.", "day": 2},
            {"text": "Mexico City government announces earthquake reconstruction fund. Accepting donations.", "day": 2},
            # Day 4 — Recovery; services resuming
            {"text": "Mexico City Metro Lines 1, 2, and 5 resuming service today after earthquake repairs.", "day": 3},
            {"text": "Power restored to 90% of Mexico City after earthquake. Full restoration by weekend.", "day": 3},
            {"text": "Mexico City businesses starting to reopen in unaffected areas. Economy slowly recovering.", "day": 3},
            {"text": "1200 families in Mexico City received emergency housing assistance so far.", "day": 3},
            {"text": "Mexico City earthquake insurance claims processing begins. Citizens urged to file.", "day": 3},
            {"text": "Mexico City cleanup operations underway. Debris removal from major roads complete.", "day": 3},
            # Day 5 — Mostly recovered; lessons learned
            {"text": "Mexico City returning to normal 5 days after devastating earthquake. Resilient city.", "day": 4},
            {"text": "Mexico City earthquake memorial planned for victims. City mourning but rebuilding.", "day": 4},
            {"text": "All Mexico City hospitals back to normal operations. Earthquake patients discharged.", "day": 4},
            {"text": "Mexico City government reviewing building codes after earthquake exposed weaknesses.", "day": 4},
            {"text": "Donations to Mexico City earthquake fund reach $50 million. Thank you to all.", "day": 4},
            {"text": "Mexico City earthquake: final death toll 34, over 500 injured. City rebuilding stronger.", "day": 4},
        ]
    },
    # -----------------------------------------------------------------------
    # CASE 10 — CONFLICTING LOCATIONS: Tweets where user profile != text location
    # -----------------------------------------------------------------------
    "Seoul, South Korea": {
        "case": 10, "expected_status": "location_uncertain",
        "place_tag": None,  # No place tag — simulates missing geo-data
        "country": "South Korea",
        "coords": (37.5665, 126.978),
        "authors": [
            # Authors whose profile says one location but tweet mentions Seoul
            ("ny_reporter", "New York Reporter", "New York, USA"),
            ("london_journo", "London Journalist", "London, UK"),
            ("tokyo_tourist", "Tokyo Tourist", "Tokyo, Japan"),
            ("delhi_news", "Delhi News Channel", "New Delhi, India"),
            ("sydney_expat", "Sydney Expat", "Sydney, Australia"),
            ("berlin_blogger", "Berlin Blogger", "Berlin, Germany"),
            ("paris_media", "Paris Media Outlet", "Paris, France"),
            ("toronto_press", "Toronto Press", "Toronto, Canada"),
            ("dubai_news", "Dubai News Network", "Dubai, UAE"),
            ("seoul_local1", "Seoul Local 1", "Seoul, South Korea"),
            ("seoul_local2", "Seoul Local 2", "Gangnam, Seoul"),
            ("seoul_local3", "Seoul Local 3", "Seoul, South Korea"),
            ("busan_user", "Busan User", "Busan, South Korea"),
            ("jeju_traveler", "Jeju Traveler", "Jeju Island"),
            ("incheon_news", "Incheon News", "Incheon, South Korea"),
        ],
        "conflicting_locations": True,
        "tweets": [
            # Mix of remote reporters and local witnesses
            {"text": "Reports of severe flooding in Seoul, South Korea. Major river overflowing.", "author_idx": 0, "profile_loc": "New York, USA"},
            {"text": "Breaking: Seoul flooding worsens. Han River at record levels.", "author_idx": 1, "profile_loc": "London, UK"},
            {"text": "My friend in Seoul says the flooding is really bad. Sending thoughts.", "author_idx": 2, "profile_loc": "Tokyo, Japan"},
            {"text": "Seoul flooding coverage: infrastructure severely damaged according to reports.", "author_idx": 3, "profile_loc": "New Delhi, India"},
            {"text": "Hoping my family in Seoul is safe. Can't reach them by phone.", "author_idx": 4, "profile_loc": "Sydney, Australia"},
            {"text": "Seoul disaster looks terrible from the videos I'm seeing online.", "author_idx": 5, "profile_loc": "Berlin, Germany"},
            {"text": "International media covering Seoul floods. Sending aid from Paris.", "author_idx": 6, "profile_loc": "Paris, France"},
            {"text": "Seoul flood update from overseas Korean community in Toronto.", "author_idx": 7, "profile_loc": "Toronto, Canada"},
            {"text": "Seoul flooding: watching news from Dubai. Looks devastating.", "author_idx": 8, "profile_loc": "Dubai, UAE"},
            # These are actual locals — more reliable
            {"text": "I'm IN Seoul right now. The flooding is real. Water up to my knees on Gangnam main road.", "author_idx": 9, "profile_loc": "Seoul, South Korea"},
            {"text": "Seoul resident here. My apartment building basement completely flooded. Evacuating.", "author_idx": 10, "profile_loc": "Gangnam, Seoul"},
            {"text": "Seoul flooding: I can confirm Han River has breached. Yeouido area underwater.", "author_idx": 11, "profile_loc": "Seoul, South Korea"},
            {"text": "Driving from Busan to Seoul to help family. Roads are getting worse.", "author_idx": 12, "profile_loc": "Busan, South Korea"},
            {"text": "Seoul area forecast says more rain coming tonight. Situation may get worse.", "author_idx": 13, "profile_loc": "Jeju Island"},
            {"text": "Incheon airport flights to Seoul area cancelled due to flooding.", "author_idx": 14, "profile_loc": "Incheon, South Korea"},
        ]
    },
}


# ===========================================================================
# TWEET GENERATION
# ===========================================================================
def _make_tweet(text, author_handle, author_name, place_tag, country, coords,
                created_at, user_profile_location=None, geo_chance=0.3,
                source="twitter_api_mock"):
    """Build a single tweet dict matching Twitter API v2 structure."""
    return {
        "tweetId": f"tw_{uuid.uuid4().hex[:16]}",
        "text": text,
        "author": author_handle,
        "authorName": author_name,
        "authorId": f"uid_{random.randint(100000000, 999999999)}",
        "createdAt": created_at.isoformat(),
        "retweetCount": random.randint(0, 5000),
        "favoriteCount": random.randint(0, 15000),
        "url": f"https://twitter.com/{author_handle}/status/{random.randint(10**17, 10**18)}",
        "placeTag": place_tag,
        "placeCountry": country,
        "userProfileLocation": user_profile_location or place_tag or "",
        "geoCoordinates": {
            "lat": coords[0] + random.uniform(-0.05, 0.05),
            "lng": coords[1] + random.uniform(-0.05, 0.05),
        } if coords and random.random() < geo_chance else None,
        "source": source,
        "status": "unverified",
    }


def generate_all_tweets():
    """Generate the complete dataset covering all 10 cases."""
    all_tweets = []
    base_time = datetime(2026, 3, 12, 8, 0, 0)

    # --- CASE 4: Confirmed disaster locations (large clusters) ---
    for loc_name, loc in CONFIRMED_LOCATIONS.items():
        for i, text in enumerate(loc["tweets"]):
            t_time = base_time + timedelta(hours=random.uniform(0, 36),
                                           minutes=random.randint(0, 59))
            auth = random.choice(loc["authors"])
            all_tweets.append(_make_tweet(
                text=text, author_handle=auth[0], author_name=auth[1],
                place_tag=loc["place_tag"], country=loc["country"],
                coords=loc["coords"], created_at=t_time,
                user_profile_location=loc_name,
            ))

    # --- EDGE CASES ---
    for loc_name, loc in EDGE_CASE_LOCATIONS.items():
        coords = loc["coords"]
        country = loc["country"]
        place_tag = loc.get("place_tag")
        is_burst = loc.get("temporal_burst", False)
        spread_days = loc.get("temporal_spread_days", 0)
        is_conflicting = loc.get("conflicting_locations", False)

        for i, t_data in enumerate(loc["tweets"]):
            text = t_data if isinstance(t_data, str) else t_data["text"]

            # --- Pick author ---
            if isinstance(t_data, dict) and t_data.get("force_single_author"):
                auth = loc["authors"][0]
                profile_loc = loc_name
            elif isinstance(t_data, dict) and "author_idx" in t_data:
                auth_entry = loc["authors"][t_data["author_idx"]]
                auth = (auth_entry[0], auth_entry[1])
                profile_loc = auth_entry[2] if len(auth_entry) > 2 else loc_name
            else:
                auth = random.choice(loc["authors"])
                auth = (auth[0], auth[1])
                profile_loc = loc_name

            # Override profile location for conflicting-location case
            if isinstance(t_data, dict) and "profile_loc" in t_data:
                profile_loc = t_data["profile_loc"]

            # --- Pick timestamp ---
            if is_burst:
                # All within 2 hours
                t_time = base_time + timedelta(minutes=random.randint(0, 120),
                                               seconds=random.randint(0, 59))
            elif spread_days:
                day = t_data.get("day", 0) if isinstance(t_data, dict) else 0
                t_time = base_time + timedelta(days=day,
                                               hours=random.randint(6, 22),
                                               minutes=random.randint(0, 59))
            else:
                t_time = base_time + timedelta(hours=random.uniform(0, 36),
                                               minutes=random.randint(0, 59))

            # No place tag for Seoul (case 10)
            geo_chance = 0.0 if is_conflicting else 0.3

            all_tweets.append(_make_tweet(
                text=text, author_handle=auth[0], author_name=auth[1],
                place_tag=place_tag, country=country, coords=coords,
                created_at=t_time, user_profile_location=profile_loc,
                geo_chance=geo_chance,
            ))

    random.shuffle(all_tweets)
    return all_tweets


# ===========================================================================
# OUTPUT
# ===========================================================================
def save_to_json(tweets, filepath="mock_geo_tweets.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tweets, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(tweets)} tweets to {filepath}")


def load_to_backend(tweets):
    print(f"\nLoading {len(tweets)} mock geo-tweets into backend...")
    print("=" * 60)
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("✗ Backend server is not running!"); return
        print("✓ Backend server is running\n")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend. Run: cd backend && npm run dev"); return

    ok, fail = 0, 0
    for i, tweet in enumerate(tweets, 1):
        try:
            disp = tweet["text"][:55] + "..." if len(tweet["text"]) > 55 else tweet["text"]
            print(f"[{i}/{len(tweets)}] {disp}")
            resp = requests.post(f"{BACKEND_URL}/tweets/create", json=tweet, timeout=10)
            if resp.status_code == 200 and resp.json().get("success"):
                ok += 1
            else:
                fail += 1; print(f"  ✗ {resp.json().get('error','Unknown')}")
        except Exception as e:
            fail += 1; print(f"  ✗ {e}")

    print(f"\n{'='*60}\n✓ Loaded: {ok}  ✗ Failed: {fail}  Total: {ok+fail}")


def print_summary(tweets):
    print("\n📊 Mock Dataset Summary")
    print("=" * 60)

    # Count by place tag
    loc_counts = {}
    for t in tweets:
        key = t.get("placeTag") or t.get("userProfileLocation") or "Unknown"
        loc_counts[key] = loc_counts.get(key, 0) + 1

    # Count unique authors per location
    loc_authors = {}
    for t in tweets:
        key = t.get("placeTag") or t.get("userProfileLocation") or "Unknown"
        loc_authors.setdefault(key, set()).add(t["author"])

    print(f"\n{'Location':<30} {'Tweets':>7} {'Authors':>8}  Case")
    print("-" * 70)

    case_map = {}
    for loc_name, loc in EDGE_CASE_LOCATIONS.items():
        tag = loc.get("place_tag") or loc_name
        case_map[tag] = (loc["case"], loc["expected_status"])

    for loc in sorted(loc_counts.keys()):
        cnt = loc_counts[loc]
        auth_cnt = len(loc_authors.get(loc, set()))
        case_info = case_map.get(loc, (4, "confirmed_disaster"))
        print(f"{loc:<30} {cnt:>7} {auth_cnt:>8}  Case {case_info[0]}: {case_info[1]}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {len(tweets):>7}")

    geo = sum(1 for t in tweets if t.get("geoCoordinates"))
    no_place = sum(1 for t in tweets if not t.get("placeTag"))
    print(f"\nWith GPS coords:  {geo}/{len(tweets)} ({geo*100//len(tweets)}%)")
    print(f"Missing placeTag: {no_place}/{len(tweets)}")


if __name__ == "__main__":
    print("🌍 Generating Mock Geo-Twitter Dataset (All 10 Cases)")
    print("=" * 60)
    tweets = generate_all_tweets()
    print_summary(tweets)
    save_to_json(tweets)
    print("\n" + "=" * 60)
    choice = input("Load tweets into backend? (y/n): ").strip().lower()
    if choice == "y":
        load_to_backend(tweets)
    else:
        print("Skipped. Run again to load.")

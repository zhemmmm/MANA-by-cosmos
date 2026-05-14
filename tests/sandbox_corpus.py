"""
Synthetic Taglish disaster posts for CorEx sandbox training.
~200 posts across 8 NDRRMC clusters. Heavily informal Taglish.
Each post has: text, ground_truth_topics (list), cluster_id.
Multi-label posts (~15%) appear in multiple topics.
NOT for production DB — in-memory training only.
"""

SANDBOX_POSTS = [
    # ── CLUSTER A: Food & NFIs (relief) ──────────────────────────────────
    {"text": "Grabe wala na kaming makain dito sa evac center!! 3 days na walang food pack dumating. Mga bata umiiyak na sa gutom. Pls padala na ng relief goods!!", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Saan na yung rice at canned goods na pinangako?? Ang tagal na namin naghihintay dito sa evacuation center. Walang pagkain ang pamilya namin", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "DSWD nagdistribute ng food pack kanina sa covered court. May bigas, sardinas, at noodles. Salamat sa tulong!! Sana makarating sa lahat", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Kulang na kulang ang relief goods dito sa amin. Isang food pack lang para sa 8 katao?? Hindi sapat yan. Kailangan pa ng blanket at hygiene kit", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Nangunguna ang Red Cross sa pamimigay ng water refill at canned goods sa mga evacuees. Salamat po sa donations ng mga kababayan!!", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "URGENT: walang tubig inumin dito sa evac site!! Mga bata dehydrated na. Pls send water sachet or drinking water ASAP!", "topics": ["relief", "health_medical"], "cluster": "cluster-a"},
    {"text": "Repacking ng relief goods ongoing sa barangay hall. Volunteers needed para sa distribution bukas. Tara tulungan natin mga kababayan natin", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Ayuda update: nagbigay na ng food pack ang LGU kanina. May rice, sardinas, noodles, at tubig. Maraming salamat sa lahat ng donors!", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Hindi pa rin dumadating ang relief goods sa sitio namin!! Nakalimutan na ba kami?? 5 days na kami walang sapat na pagkain dito", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Nagloading na ng food packs sa truck. Papunta na sa mga evacuation center sa Marikina. Hygiene kit kasama na rin. ETA 2 hours", "topics": ["relief", "logistics"], "cluster": "cluster-a"},
    {"text": "Pls help! Family of 7 kami dito sa evac walang nakain. Yung mga canned goods ubos na. Kahit bigas lang po pls", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Salamat sa mga nagdonate ng blanket at kumot! Malamig na malamig dito sa evacuation center lalo na pag gabi. Bless po kayo lahat", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Distribution ng NFI sa Brgy San Miguel bukas 8AM. Dalhin ang family registration card. May pagkain, tubig, at hygiene supplies", "topics": ["relief"], "cluster": "cluster-a"},
    {"text": "Grabe ang pila sa relief distribution!! Simula 5AM pumila kami hanggang ngayon wala pa rin. Sana dagdagan ang food pack para sa lahat", "topics": ["relief"], "cluster": "cluster-a"},

    # ── CLUSTER B: Health / Medical (health_medical) ─────────────────────
    {"text": "May nagkasakit na dito sa evac!! Lagnat at ubo ang mga bata. Walang available na doctor o gamot. Need ng medical team ASAP!!", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "URGENT: maraming evacuees na may diarrhea at vomiting. Baka contaminated ang tubig dito sa evacuation center. Kailangan ng DOH intervention!", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Leptospirosis warning sa mga lumakad sa baha!! Mag-ingat po. Pumunta agad sa health center kung may sugat sa paa at naglakad sa floodwater", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Medical team deployed na sa evac center. Nagbibigay ng gamot at first aid. Mga may fever at wound puntahan lang sa tent 3", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Tatay ko kailangan ng insulin!! Diabetic siya at ubos na ang gamot niya. Walang parmasya bukas dito. Hospital ang pinakamalapit pero hindi makarating!", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Grabe ang sitwasyon dito. Mga bata nahihirapan huminga. Walang ventilation sa evacuation center. Need ng nurse at doctor dito agad!", "topics": ["health_medical", "evacuation"], "cluster": "cluster-b"},
    {"text": "DOH advisory: mag-ingat sa dengue at cholera after ng baha. Linisin ang paligid at huwag uminom ng tubig galing sa gripo kung hindi pa cleared", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Ambulansya hindi makapasok sa area namin dahil sa baha!! May injured na senior citizen dito na kailangan ng hospital. Paano na to??", "topics": ["health_medical", "logistics"], "cluster": "cluster-b"},
    {"text": "Salamat sa medical mission ng Red Cross! Maraming natulungan. Nagbigay ng gamot, first aid, at mental health counseling sa mga evacuees", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "WASH advisory: Huwag uminom ng floodwater!! Marami ng nagkakasakit ng stomach flu dito sa evac. Clean water distribution bukas 7AM", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Nurse na volunteer dito sa evacuation center. Kulang kami sa medical supplies. Pls send paracetamol, oral rehydration salts, at bandage", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "Psychosocial support team dumating na sa evac center. Maraming mga bata at nanay na traumatized. Counseling sessions ongoing sa chapel", "topics": ["health_medical"], "cluster": "cluster-b"},
    {"text": "May outbreak ng skin disease sa evacuation site!! Scabies at fungal infection kumakalat. Kailangan ng dermatologist at gamot dito!!", "topics": ["health_medical"], "cluster": "cluster-b"},

    # ── CLUSTER C: Evacuation / CCCM (evacuation) ────────────────────────
    {"text": "PUNO NA ANG EVACUATION CENTER!! Hindi na kami makapasok sa covered court. 800+ families na dito grabe ang sikip. Saan kami pupunta??", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Registration sa evac center ongoing na. Dalhin ang valid ID. Mga bagong dating pumila sa gate 2. Overflow site sa gymnasium bukas na", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Toilet line sa evacuation center 45 minutes na!! Isang CR lang para sa 200+ tao. Kadiri na ang kalagayan dito. Dagdagan naman ang facilities!!", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Safe space for women at children available na sa tent 5. Wag mag-alinlangan lumapit sa social workers para sa protection concerns", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Evacuation center sa barangay hall overcrowded na!! Mga tao nasa labas natutulog kasi walang space sa loob. Sana magbukas ng overflow site", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Grabe ang displacement!! 3,000+ families lumikas na sa aming barangay. Lahat ng evacuation site puno na. Gymnasium at chapel ginawang shelter", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Camp management team deployed sa evacuation center. Headcount 1,200 individuals. Segregation ng area para sa elderly at PWD ongoing", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Mga evacuees sa covered court nagrereklamo ng privacy!! Walang divider o kurtina. Mga pamilya magkatabi-tabi. Sana ayusin ang setup", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "NDRRMC update: additional evacuation sites opened sa QC at Pasig. Buses available para sa transport ng evacuees. Registration required", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Curfew sa evacuation center 10PM. Bawal lumabas after. Security team ng barangay naka-duty 24/7. Para sa safety ng lahat", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Displaced families dito sa gym walang sleeping mat o banig. Cement lang ang higaan. Malamig at basang-basa pa ang ibang area", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Evacuation advisory: mga residente ng low-lying areas sa Marikina lumikas na!! Tumataas pa ang tubig. Pumunta sa nearest evacuation center", "topics": ["evacuation"], "cluster": "cluster-c"},
    {"text": "Nag-aaway na ang mga evacuees dahil sa overcrowding!! Kailangan ng peacekeeping team at social workers dito sa evac site ASAP", "topics": ["evacuation"], "cluster": "cluster-c"},

    # ── CLUSTER D: Logistics ─────────────────────────────────────────────
    {"text": "BLOCKED na ang Marcos Highway dahil sa landslide!! Mga truck ng relief goods hindi makalusot. Reroute via SLEX ang tanging option ngayon", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "DPWH road clearing ongoing sa Montalban. Debris at putik tinatanggal. Passable na ang isang lane pero restricted pa rin sa heavy vehicles", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Convoy ng relief truck stranded sa checkpoint!! Baha ang daan hindi makalusot. Alternate route check muna bago mag-dispatch ng bagong batch", "topics": ["logistics", "relief"], "cluster": "cluster-d"},
    {"text": "Bridge sa Brgy Tumana collapsed na!! Wala ng daanan ang mga tao at sasakyan. Kailangan ng temporary bridge o bangka para makatawid", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Road damage report: Ortigas Ave impassable due to sinkhole. C5 partially flooded. EDSA northbound open pero mabagal. Mag-ingat sa biyahe!", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Warehouse ng relief goods sa Pasig binaha!! Emergency transfer ng supplies sa alternate hub sa Cainta. Mga truck standby na", "topics": ["logistics", "relief"], "cluster": "cluster-d"},
    {"text": "DPWH advisory: alternate route para sa mga papuntang Rizal via Antipolo-Teresa road. Marcos Highway closed until further notice", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Delivery ng food packs delayed 6 hours dahil sa blocked road sa Montalban. Mga truck naka-standby pa sa NLEX exit waiting for clearance", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Guho sa kalsada sa Antipolo!! Landslide bumagsak sa national highway. Road clearing team deployed na. ETA 12 hours bago ma-clear", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Fuel shortage sa rescue operations!! Diesel para sa rescue boat at truck kulang na. Coordinate with depot para sa emergency fuel supply", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Aerial delivery ng relief goods via helicopter sa isolated barangay. Road access totally cut off dahil sa landslide at baha", "topics": ["logistics", "relief"], "cluster": "cluster-d"},
    {"text": "Transport ng evacuees delayed!! Bus breakdown sa daan papuntang evacuation center. Replacement vehicle requested sa LGU fleet", "topics": ["logistics"], "cluster": "cluster-d"},
    {"text": "Barge na lang ang pwedeng gamitin para mag-deliver ng supplies sa isla. Lahat ng kalsada blocked. Coordination with Coast Guard ongoing", "topics": ["logistics"], "cluster": "cluster-d"},

    # ── CLUSTER E: Telecom / Power (telecom_power) ───────────────────────
    {"text": "WALANG KURYENTE sa buong Brgy San Andres simula kagabi!! Brownout na brownout grabe. Kailan ba ibabalik?? Generator lang ang pag-asa namin!!", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "No signal Globe at Smart sa area namin!! Hindi kami makacontact ng rescue team. Cell site down pa rin. Pls restore na!!", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "PLDT restoration team deployed na sa Marikina. Power lines at cell site repair ongoing. ETA 24-48 hours para ma-restore ang signal", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Power bank distribution sa evacuation center!! Salamat sa mga nagdonate. Kahit papaano may pangcharge ng phone para makaconnect sa family", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Walang internet, walang signal, walang kuryente!! Paano kami makakahingi ng tulong?? Radio na lang ang tanging communication namin dito", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Cell site sa Brgy Tumana bumagsak dahil sa bagyo!! Dead zone ang buong area. DICT coordinating with telcos for emergency restoration", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Generator na donated by private company nagpapailaw sa evacuation center. Salamat!! Kahit papaano may kuryente na kami sa gabi", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Blackout sa 5 barangay sa Marikina!! MERALCO repair team hindi pa dumadating. Mga pagkain sa ref nasisira na. Kailan ba ang restoration??", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Emergency radio communication activated para sa rescue operations. HF radio lang ang gumagana ngayon. Satellite phone deployed sa command center", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Solar panel at inverter na-setup sa evacuation site para sa emergency power. Enough para sa lighting at phone charging. Salamat sa donors!", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Smart at Globe nagpadala na ng mobile cell site sa affected area!! Partial signal restored. Salamat! Sana ma-full restore na agad", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Grabe walang kuryente 4 days na!! Pagkain namin nasisira. Walang electric fan mainit na mainit. Mga bata at matanda nahihirapan na", "topics": ["telecom_power"], "cluster": "cluster-e"},
    {"text": "Network outage sa buong Rizal province!! Mga hotline ng MDRRMO hindi nare-reach. Paano mag-report ng emergency?? Fix na pls!!", "topics": ["telecom_power"], "cluster": "cluster-e"},

    # ── CLUSTER F: Education ─────────────────────────────────────────────
    {"text": "WALANG PASOK bukas!! DepEd NCR official: class suspended sa lahat ng level sa Marikina, QC, at Pasig dahil sa baha #WalangPasok", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "School namin ginawang evacuation center!! Paano na ang klase namin?? Sana mag-announce na ng online class o modular para di kami masyadong late", "topics": ["education", "evacuation"], "cluster": "cluster-f"},
    {"text": "DepEd nagdistribute ng self-learning modules sa mga evacuee students. Printed materials para sa mga walang gadget o internet access", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Class suspension extended hanggang Friday!! Mga students stay safe. Learning materials available sa barangay hall for pickup", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Mga libro at school supplies ng anak ko nasira sa baha!! Lahat basang-basa na. Paano na to?? DepEd sana may replacement program", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Temporary classroom na-setup sa chapel ng barangay. Mga teacher volunteers nagtuturo sa displaced students. Salamat sa kanila!!", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Virtual learning announced para sa affected schools. Problema: maraming students walang gadget at internet. Paano sila makakapag-aral??", "topics": ["education", "telecom_power"], "cluster": "cluster-f"},
    {"text": "DepEd assessment: 45 schools damaged by flooding sa NCR. 12 currently used as evacuation centers. Timeline for class resumption TBD", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Walang klase bukas at sa makalawa!! Stay home mga estudyante. Iwasan ang mga baha-bahang area. Safety first bago ang klase!!", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Mga teachers nagbo-volunteer sa evacuation center habang walang pasok. Tumutulong sa registration at pag-aalaga ng mga bata. Saludo!!", "topics": ["education", "evacuation"], "cluster": "cluster-f"},
    {"text": "School building inspection ongoing. Structural damage assessment bago payagan bumalik ang mga students. Safety protocol ng DepEd", "topics": ["education"], "cluster": "cluster-f"},
    {"text": "Printed modules para sa Grade 1-6 ready na for distribution. Mga parents puwede na kunin sa division office. Bring ID po", "topics": ["education"], "cluster": "cluster-f"},

    # ── CLUSTER G: Search, Rescue & Retrieval (rescue) ───────────────────
    {"text": "SOS!! TULONG PO!! Nastranded kami sa bubong ng bahay namin sa Brgy Sta Elena!! 5 kami dito may 2 bata at lola!! Pataas pa tubig!! RESCUE PLS!!", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "URGENT: Family trapped sa 2nd floor sa Purok 3 Marikina!! Hindi makalusot ang rescue boat dahil sa malakas na agos. SOS!!", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Rescue boat deployed na sa Brgy Tumana!! 15 families na-rescue na. Ongoing pa ang operations. Coast Guard at NDRRMC nandito na", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "SAKLOLO!! May naipit sa bumagsak na pader!! Kailangan ng rescue team dito sa Brgy San Miguel ASAP!! Hindi namin maialis mag-isa!!", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Swift water rescue team deployed sa Marikina River area. 3 kayak at 2 rubber boat. Retrieval ng stranded residents ongoing", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Helicopter rescue requested para sa isolated barangay!! Road access cut off at tubig mataas pa rin. 50+ stranded families waiting", "topics": ["rescue", "logistics"], "cluster": "cluster-g"},
    {"text": "Nailigtas na lahat ng stranded sa Purok 5!! Salamat sa rescue team at Coast Guard!! Walang casualty sa area namin. Thank God!!", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "TABANG TABANG!! Nalulunod na yung kapitbahay namin!! Ang lakas ng agos ng baha!! Wala kaming rescue boat dito!! HELP!!", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Search and rescue operation update: 120 individuals rescued sa Marikina. 8 areas cleared. Operations continue sa remaining flood zones", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Naka-stranded kami sa rooftop!! 2 pamilya kami dito kasama mga aso at pusa. Tubig hanggang 1st floor na!! Kailan darating ang rescue??", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "BFP at NDRRMC joint rescue operation sa collapsed building sa Pasig. USAR team deployed. 3 trapped individuals located via thermal scanner", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Retrieval ng mga gamit at sasakyan ongoing na sa Marikina pagbaba ng tubig. Maraming nasira pero buhay lahat ng tao. Salamat sa rescuers!!", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "Coast Guard rescue boat patuloy ang operation sa coastal areas. Mga mangingisda na-stranded sa dagat ni-rescue na. All safe", "topics": ["rescue"], "cluster": "cluster-g"},
    {"text": "SOS SOS!! Please help!! Naipit ang paa ng anak ko sa debris!! Hindi namin maalis!! Kailangan ng rescue team NGAYON NA!!", "topics": ["rescue"], "cluster": "cluster-g"},

    # ── CLUSTER H: Dead & Missing (dead_missing) ─────────────────────────
    {"text": "MISSING: Si Lola Conching 72 years old mula Brgy Tumana. Last seen kagabi bago lumala ang baha. Kung nakita nyo po siya pls contact kami!!", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Family tracing coordination desk bukas na sa City Hall. Kung may nawawalang kamag-anak pumunta po doon. Dalhin ang photo at details", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "3 bodies retrieved sa Cagayan River pagkatapos ng flash flood. Identification ongoing. Pamilya ng mga nawawala pumunta sa coordination desk", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "NAWAWALA ang tatay ko simula nung baha!! 4 days na walang contact. Huli siyang nakita sa Brgy San Andres. Pls share pakihanap po siya!!", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Hospital list ng mga na-admit after ng baha available na sa MDRRMO desk. Cross-reference with missing persons registry ongoing", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Death toll update: 7 confirmed fatalities sa NCR dahil sa flooding. NDRRMC coordinating with LGUs for verification at identification", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Missing person report filed sa barangay para kay Juan Dela Cruz, 45 anyos. Lost contact since Sunday night. Pakishare po pls!!", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Body identified sa riverbank ng Marikina River. Coordination with PNP at NBI ongoing. Pamilya contacted na for verification", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "MDM protocol activated sa Marikina. Ante-mortem data collection ongoing. Mga pamilya ng missing pumunta sa coordination desk City Hall", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Unaccounted pa rin ang 12 residente ng Brgy Nangka. Search teams deployed. Family tracing hotline: 161. Tawag lang po kung may info", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "Grabe ang lungkot. Natagpuan na ang lola naming nawawala. Patay na po. Huli siyang nakita sa ilog. Rest in peace po Lola Carmen", "topics": ["dead_missing"], "cluster": "cluster-h"},
    {"text": "DSWD psychosocial support available para sa families ng mga namatay at nawawala. Counseling sessions sa City Hall 2nd floor", "topics": ["dead_missing", "health_medical"], "cluster": "cluster-h"},
    {"text": "Missing child alert!! 5 year old boy nawala sa evacuation center kagabi. Wearing blue shirt at shorts. Pls help us find him!!", "topics": ["dead_missing"], "cluster": "cluster-h"},
]

# Quick stats
if __name__ == "__main__":
    from collections import Counter
    cluster_counts = Counter(p["cluster"] for p in SANDBOX_POSTS)
    topic_counts = Counter(t for p in SANDBOX_POSTS for t in p["topics"])
    multi = sum(1 for p in SANDBOX_POSTS if len(p["topics"]) > 1)
    print(f"Total posts: {len(SANDBOX_POSTS)}")
    print(f"Multi-label: {multi} ({100*multi/len(SANDBOX_POSTS):.0f}%)")
    print(f"\nPer cluster: {dict(cluster_counts)}")
    print(f"Per topic:   {dict(topic_counts)}")

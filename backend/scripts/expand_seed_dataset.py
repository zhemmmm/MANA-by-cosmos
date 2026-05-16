"""
Expand seed_dataset.json with 15 Cluster B (WASH — water utility emergency)
and 15 Cluster D (Logistics — road/infrastructure disruption) posts.
Run from backend/ directory: python scripts/expand_seed_dataset.py
"""
import json
import random
import hashlib
from pathlib import Path

SEED_FILE = Path(__file__).parent.parent / "seed_dataset.json"


def make_post(cluster, idx, text, page, likes=None, shares=None, comments=None):
    uid = hashlib.md5(text.encode()).hexdigest()[:8]
    likes = likes or random.randint(50, 300)
    shares = shares or random.randint(10, 80)
    comments = comments or random.randint(15, 60)
    return {
        "postId": f"seed_{cluster}_{idx:04d}_{uid}",
        "text": text,
        "pageName": page,
        "facebookUrl": f"https://www.facebook.com/page/{random.randint(1000, 9999)}",
        "url": f"https://www.facebook.com/post/{random.randint(100000, 999999)}",
        "time": "2025-11-14T08:00:00+00:00",
        "likes": likes,
        "topReactionsCount": likes + random.randint(5, 20),
        "reactionLikeCount": likes,
        "reactionLoveCount": random.randint(0, 10),
        "reactionCareCount": random.randint(0, 10),
        "reactionHahaCount": 0,
        "reactionWowCount": random.randint(0, 5),
        "reactionSadCount": random.randint(0, 10),
        "reactionAngryCount": random.randint(0, 15),
        "shares": shares,
        "comments": comments,
        "viewsCount": likes * random.randint(3, 8),
        "_seed_cluster_id": cluster,
    }


NEW_B = [
    make_post("cluster-b", 148,
        "SERVICE ADVISORY: Emergency repair and maintenance activities scheduled May 14-19 affecting "
        "Pasig, Taguig, Quezon City, Manila, Marikina, Rizal. Expect water interruption in affected areas. "
        "Manila Water regulated and monitored by MWSS Regulatory Office.",
        "Manila Water"),
    make_post("cluster-b", 149,
        "WATER INTERRUPTION NOTICE: Maynilad scheduled emergency pipeline repair on May 15 will affect "
        "Caloocan, Malabon, Navotas, and Valenzuela. Water supply cut from 10PM to 6AM. "
        "Residents advised to store water in advance.",
        "Maynilad Water Services"),
    make_post("cluster-b", 150,
        "ADVISORY: Emergency repair of burst main pipeline in Quezon City. Water supply unavailable in "
        "Batasan Hills, Commonwealth, and nearby barangays today. Estimated restoration by 8PM. "
        "Water rationing trucks deployed.",
        "Manila Water"),
    make_post("cluster-b", 151,
        "WATER SERVICE DISRUPTION: Typhoon damage to water treatment facility in Angat reduced supply by 40%. "
        "Affected: Metro Manila and parts of Bulacan. Manila Water implementing zonal rationing. "
        "Check website for schedule.",
        "Manila Water"),
    make_post("cluster-b", 152,
        "MAYNILAD ADVISORY: Emergency shutdown of Putatan Water Treatment Plant due to turbidity surge "
        "caused by heavy rains. Service interruption 24-48 hours for Muntinlupa, Las Pinas, Paranaque, "
        "and parts of Cavite.",
        "Maynilad Water Services"),
    make_post("cluster-b", 153,
        "URGENT: Contaminated water supply reported in Brgy. Tondo Manila after floodwaters infiltrated "
        "distribution line. Residents warned NOT to use tap water for drinking or cooking. "
        "Boil water advisory in effect. Tanker trucks dispatched.",
        "Manila City DRRMO"),
    make_post("cluster-b", 154,
        "Water supply alert: MWSS reports reduction in raw water supply from Angat Dam due to El Nino. "
        "Conservation measures in effect. Metro Manila residents urged to limit water use. "
        "Supply rotation schedule posted on MWSS website.",
        "MWSS Regulatory Office"),
    make_post("cluster-b", 155,
        "NOTICE: Pipe rehabilitation work along EDSA Cubao will interrupt water service for Quezon City "
        "zones 1-4 on May 17-18. Affected households advised to store water. Water trucks at Araneta "
        "Coliseum parking lot.",
        "Manila Water"),
    make_post("cluster-b", 156,
        "Flash flood damage to water mains in Marikina City. Brgy. Sto. Nino, Brgy. Concepcion, "
        "Brgy. Tumana affected. Estimated repair: 72 hours. MWSS deploying emergency water tankers "
        "to affected communities.",
        "Marikina DRRMO"),
    make_post("cluster-b", 157,
        "BOIL WATER ADVISORY: Heavy rainfall caused turbidity in Laguna Lake affecting water treatment "
        "in Taguig and Muntinlupa. Water from taps may appear cloudy. Boil all water before consumption "
        "until further notice.",
        "DOH Philippines"),
    make_post("cluster-b", 158,
        "Maynilad emergency repair sa main pipe sa Valenzuela City. Affected: Parada, Dalandanan, "
        "Malinta, Gen. T. De Leon barangays. Walang tubig mula 8PM hanggang 6AM. "
        "Mag-imbak ng tubig. Tanker trucks sa barangay halls.",
        "Maynilad Water Services"),
    make_post("cluster-b", 159,
        "WATER INTERRUPTION: Manila Water emergency pipeline replacement in San Juan City affects "
        "Brgy. Addition Hills, Brgy. Batis, Brgy. Corazon de Jesus. Restoration expected 10PM. "
        "Store water in advance.",
        "Manila Water"),
    make_post("cluster-b", 160,
        "Critical water shortage in evacuation centers in Cagayan Valley after Typhoon Pepito. "
        "DSWD and LWUA coordinating emergency water supply. 15,000 evacuees without safe drinking water. "
        "Tanker trucks en route.",
        "DSWD Philippines"),
    make_post("cluster-b", 161,
        "ADVISORY: Leakage detected in transmission main along C-5 Road affecting water pressure in "
        "Pasig, Taguig, and Pateros. Manila Water deploying repair crew. Reduced water pressure for "
        "24 hours while repairs ongoing.",
        "Manila Water"),
    make_post("cluster-b", 162,
        "Water supply cut sa Paranaque, Las Pinas, Muntinlupa dahil sa emergency maintenance ng "
        "Putatan Treatment Plant. Maynilad nagpapadala ng water trucks sa mga barangay. "
        "Asahan ang service restoration bukas ng umaga.",
        "Maynilad Water Services"),
]

NEW_D = [
    make_post("cluster-d", 61,
        "ROAD CLOSURE ALERT: Marcos Highway in Marikina impassable due to landslide debris. "
        "All vehicles diverted to Sumulong Highway. DPWH crews mobilized for clearing. "
        "Estimated clearance: 6-8 hours.",
        "DPWH NCR"),
    make_post("cluster-d", 62,
        "TRAFFIC ADVISORY: MacArthur Bridge connecting Manila and Makati closed due to structural damage "
        "from flooding. Heavy vehicles rerouted via Nagtahan Bridge. DPWH assessment team on site.",
        "MMDA"),
    make_post("cluster-d", 63,
        "Relief convoy from Quezon City to Cagayan de Oro delayed due to blocked highway in Nueva Vizcaya. "
        "Typhoon Pepito downed trees covering 3km stretch of Maharlika Highway. "
        "DPWH clearing operations underway.",
        "DSWD Philippines"),
    make_post("cluster-d", 64,
        "LOGISTICS UPDATE: Calamity goods warehouse in Pasig City damaged by fire. 2,000 sacks of rice "
        "and emergency supplies lost. DSWD activating contingency stocks from Laguna and Batangas depots.",
        "DSWD Philippines"),
    make_post("cluster-d", 65,
        "ROAD ALERT: Kennon Road and Marcos Highway both blocked due to landslides from heavy rainfall. "
        "Baguio City accessible only via Naguilian Road. Long queues expected. "
        "DPWH deploying heavy equipment for clearing.",
        "DPWH CAR"),
    make_post("cluster-d", 66,
        "Flood closed NLEX northbound in Bulacan from km 37 to km 43. Vehicles rerouted via SCTEX. "
        "Stranded motorists being assisted by NLEX crew. DPWH monitoring flood levels.",
        "NLEX Corporation"),
    make_post("cluster-d", 67,
        "TYPHOON LOGISTICS BRIEF: Port of Manila suspended all cargo and passenger operations due to "
        "Typhoon Signal No. 3. Stranded vessels sheltering in Manila Bay. "
        "MARINA coordinating with PCG for vessel monitoring.",
        "Philippine Ports Authority"),
    make_post("cluster-d", 68,
        "Relief operations in Catanduanes hampered by runway damage at Virac Airport after typhoon. "
        "Military C-130 unable to land. Sea transport only option. "
        "PCG coordinating vessels for relief goods delivery.",
        "PCG Philippines"),
    make_post("cluster-d", 69,
        "ADVISORY: Quirino Highway in Rizal impassable due to rockfall from Sierra Madre slopes. "
        "No estimated clearing time due to continuous rain. Alternative via Marikina-Infanta Road "
        "recommended for relief convoys.",
        "DPWH Region IV-A"),
    make_post("cluster-d", 70,
        "Landslide blocks access road to 3 barangays in Benguet. Around 500 residents isolated. "
        "DSWD pre-positioned relief goods but delivery impossible. "
        "Helicopter extraction being considered for medical cases.",
        "Benguet PDRRMO"),
    make_post("cluster-d", 71,
        "Tulay sa Ilocos Norte bumagsak dahil sa malakas na baha. Mga residente ng dalawang barangay "
        "walang access sa bayan. DPWH nagtatayo ng bailey bridge. "
        "Asahang matatapus sa loob ng 3 araw.",
        "DPWH Region I"),
    make_post("cluster-d", 72,
        "LOGISTICS ALERT: Fuel depot in Leyte flooded after Typhoon Uring. Gasoline supply critically low "
        "across Eastern Visayas. DOE coordinating emergency fuel delivery via sea. "
        "Generators at hospitals and evacuation centers prioritized.",
        "DOE Philippines"),
    make_post("cluster-d", 73,
        "ROAD CONDITION UPDATE: Cagayan Valley Road in Isabela has 15 sections with floodwater depth "
        "0.5m to 1.2m. Heavy vehicles advised to avoid route. "
        "DPWH deploying water pumps and warning signs.",
        "DPWH Region II"),
    make_post("cluster-d", 74,
        "Typhoon damaged 3 warehouses at NFA complex in Leyte. Emergency procurement of replacement "
        "relief goods underway. NFA and DSWD coordinating resupply from Cebu and Mindanao depots.",
        "NFA Philippines"),
    make_post("cluster-d", 75,
        "Emergency road clearing on Nagcarlan-Liliw Road in Laguna after rockslide blocked both lanes. "
        "Relief vehicles carrying goods to flooded barangays forced to wait. "
        "DPWH crew working 24 hours to restore access.",
        "DPWH Region IV-A"),
]


def main():
    with SEED_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    from collections import Counter
    before = Counter(p["_seed_cluster_id"] for p in data)
    print("Before:")
    for k in sorted(before):
        print(f"  {k}: {before[k]}")
    print(f"  Total: {len(data)}")

    data.extend(NEW_B)
    data.extend(NEW_D)

    with SEED_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    after = Counter(p["_seed_cluster_id"] for p in data)
    print("\nAfter:")
    for k in sorted(after):
        print(f"  {k}: {after[k]}")
    print(f"  Total: {len(data)}")
    print(f"\nAdded: {len(data) - sum(before.values())} posts")


if __name__ == "__main__":
    main()

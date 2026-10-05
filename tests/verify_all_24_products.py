import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_cache import get_semantic_cache

cache = get_semantic_cache()
stats = cache.stats()
print(f"Total entries loaded in semantic cache: {stats['total_entries']}")

products = [
    ("crm", "Nimbus CRM"),
    ("leads", "Nimbus Leads"),
    ("quote", "Nimbus Quote"),
    ("campaigns", "Nimbus Campaigns"),
    ("social", "Nimbus Social"),
    ("sites", "Nimbus Sites"),
    ("books", "Nimbus Books"),
    ("invoice", "Nimbus Invoice"),
    ("expense", "Nimbus Expense"),
    ("people", "Nimbus People"),
    ("recruit", "Nimbus Recruit"),
    ("payroll", "Nimbus Payroll"),
    ("desk", "Nimbus Desk"),
    ("chat", "Nimbus Chat"),
    ("knowledge", "Nimbus Knowledge"),
    ("projects", "Nimbus Projects"),
    ("docs", "Nimbus Docs"),
    ("boards", "Nimbus Boards"),
    ("analytics", "Nimbus Analytics"),
    ("dashboards", "Nimbus Dashboards"),
    ("datapipe", "Nimbus DataPipe"),
    ("vault", "Nimbus Vault"),
    ("sso", "Nimbus SSO"),
    ("endpoint", "Nimbus Endpoint"),
]

print("\n--- Testing All 24 Product Overviews ---")
all_passed = True
for key, name in products:
    queries = [
        f"Can you tell me about {name}?",
        f"What is {name}?",
    ]
    for q in queries:
        res = cache.get(q)
        if not res:
            print(f"[FAIL] Query: '{q}' -> None")
            all_passed = False
        else:
            text, sim, can = res
            if sim < 0.72:
                print(f"[FAIL] Low sim ({sim}): '{q}' -> '{can}'")
                all_passed = False
            else:
                print(f"[OK] '{q}' -> sim={sim:.3f} | {text[:60]}...")

print("\n--- Testing All 24 Product Pricing Queries ---")
for key, name in products:
    queries = [
        f"How much does {name} cost?",
        f"What is the price of {name}?",
    ]
    for q in queries:
        res = cache.get(q)
        if not res:
            print(f"[FAIL] Query: '{q}' -> None")
            all_passed = False
        else:
            text, sim, can = res
            if sim < 0.72 or "dollars" not in text:
                print(f"[FAIL] Invalid pricing response: '{q}' -> '{can}' ({text[:60]})")
                all_passed = False
            else:
                print(f"[OK] '{q}' -> sim={sim:.3f} | {text[:60]}...")

print("\n--- Testing Specific Call Query ---")
specific_q = "Okay. Can you tell me about Nimbus docs?"
res = cache.get(specific_q)
assert res is not None, f"Expected cache HIT for '{specific_q}'"
print(f"[PASS] Specific query '{specific_q}' -> sim={res[1]:.3f} | response: '{res[0]}'")

print(f"\nSTATUS: {'ALL 24 PRODUCTS (OVERVIEW & PRICING) PASSED!' if all_passed else 'SOME TESTS FAILED'}")

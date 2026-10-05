import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_cache import get_semantic_cache

cache = get_semantic_cache()
print(f"Total entries loaded: {len(cache.entries)}")
assert len(cache.entries) >= 25, f"Expected at least 25 entries, got {len(cache.entries)}"

queries = [
    # All policies from context.md
    ("What is your refund policy?", "policy_refund"),
    ("What are your refund policies?", "policy_refund"),
    ("Do you offer a free trial?", "policy_free_trial"),
    ("Do you have free trials?", "policy_free_trial"),
    ("What is your data residency policy?", "policy_data_residency"),
    ("Where is customer data stored?", "policy_data_residency"),
    ("What are your billing policies?", "policy_billing"),
    ("What payment methods do you accept?", "policy_billing"),
    ("What is your cancellation policy?", "policy_cancellation"),
    ("How do I cancel my subscription?", "policy_cancellation"),
    ("What is your SLA policy?", "policy_sla"),
    ("What is your uptime guarantee?", "policy_sla"),
    ("Is Nimbus secure and compliant?", "policy_security"),
    ("Are you SOC 2 compliant?", "policy_security"),
    ("What is your customer support policy?", "policy_support"),
    ("How can I contact customer support?", "policy_support"),
    ("Do you offer an annual discount?", "policy_annual_discount"),
    ("Where is Nimbus located?", "company_overview"),

    # Singular & Plural comparisons
    ("What is your cheapest product?", "comp_cheapest_single"),
    ("Which product is the cheapest?", "comp_cheapest_single"),
    ("What are your cheapest products?", "comp_cheapest_plural"),
    ("What are the top three cheapest products?", "comp_cheapest_plural"),
    ("What is your most expensive product?", "comp_expensive_single"),
    ("What are your most expensive products?", "comp_expensive_plural"),
    ("Do you offer a free plan?", "comp_free_tier"),
    ("Do you have free products?", "comp_free_tier"),
    ("What products does Nimbus offer?", "comp_catalog_overview"),

    # Individual products info / overview
    ("What is Nimbus CRM?", "info_crm"),
    ("Can you tell me about Nimbus Docs?", "info_docs"),
    ("What does Nimbus Invoice do?", "info_invoice"),
    ("What is Nimbus Payroll?", "info_payroll"),
    ("Tell me about Nimbus Desk", "info_desk"),

    # Individual products pricing
    ("How much does Nimbus CRM cost?", "pricing_crm"),
    ("Nimbus Docs pricing", "pricing_docs"),
    ("What is the price of Nimbus Invoice?", "pricing_invoice"),
    ("How much is Nimbus Payroll?", "pricing_payroll"),
    ("How much is Nimbus Desk?", "pricing_desk"),
]

passed = 0
for q, expected_id in queries:
    res = cache.get(q)
    if res is None:
        print(f"[FAIL]: Query '{q}' got None (expected {expected_id})")
    else:
        text, sim, canonical = res
        print(f"[OK] ({sim:.3f}): '{q}' -> {canonical}")
        passed += 1

print(f"\nResult: {passed}/{len(queries)} queries matched!")
assert passed == len(queries), "Not all queries passed!"
print("All policy & product queries passed successfully!")

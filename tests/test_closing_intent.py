import re

PRODUCT_AND_POLICY_KEYWORDS = {
    "crm", "sso", "vault", "leads", "recruit", "payroll", "books", "invoice",
    "expense", "people", "desk", "chat", "projects", "docs", "analytics",
    "datapipe", "endpoint", "dashboards", "boards", "knowledge", "platform",
    "sites", "campaigns", "quote", "refund", "trial", "pricing", "price",
    "cost", "sla", "security", "support", "billing", "payment", "residency"
}

QUESTION_PHRASES = [
    r"\b(what|how|why|when|where|who|which)\b",
    r"\b(can you|could you|tell me|explain|help me with)\b",
    r"\b(how much|how many|is there|are there|do you have)\b",
]

CLOSING_PATTERNS = [
    r"\b(that'?s\s+(?:it|all|everything))\b",
    r"\b(that\s+is\s+(?:it|all|everything))\b",
    r"\b(that\'?ll\s+be\s+all)\b",
    r"\b(nothing\s+else)\b",
    r"\b(no\s+more\s+questions)\b",
    r"\b(i\'?m\s+(?:all\s+set|good|done))\b",
    r"\b(all\s+good|all\s+set)\b",
    r"\b(thank\s+you|thanks|thx)\b",
    r"\b(bye|goodbye|have\s+a\s+good\s+(?:day|one))\b",
]

def is_closing_intent(query: str) -> bool:
    clean = query.strip().lower()
    if not clean:
        return False

    # 1. Any product/policy keywords -> definitely NOT closing
    words = set(re.findall(r"\b[a-z0-9\']+\b", clean))
    if words & PRODUCT_AND_POLICY_KEYWORDS:
        return False

    # 2. Any explicit question phrasing -> definitely NOT closing
    for q_pat in QUESTION_PHRASES:
        if re.search(q_pat, clean):
            return False

    # 3. Check for closing intent pattern
    for pat in CLOSING_PATTERNS:
        if re.search(pat, clean):
            return True

    return False

test_cases = [
    ("Okay. That's it. Thank you.", True),
    ("That's it", True),
    ("Thank you", True),
    ("Thanks", True),
    ("That is all, thank you", True),
    ("No, that's everything. Thanks!", True),
    ("No more questions, thank you.", True),
    ("All good, thanks bye", True),
    ("Goodbye", True),
    ("I'm good, thanks", True),
    ("Nothing else, thank you", True),
    ("That will be all, thank you so much", True),
    ("Thank you, can you tell me about Nimbus CRM?", False),
    ("Thanks for that, what does Nimbus SSO cost?", False),
    ("What is your refund policy? Thanks", False),
    ("SSO?", False),
    ("How much does Nimbus Vault cost?", False),
    ("Can you explain the pricing?", False),
]

passed = 0
for text, expected in test_cases:
    match = is_closing_intent(text)
    status = "PASS" if match == expected else "FAIL"
    if match == expected:
        passed += 1
    print(f"[{status}] '{text}' -> {match} (expected {expected})")

print(f"\n{passed}/{len(test_cases)} tests passed.")

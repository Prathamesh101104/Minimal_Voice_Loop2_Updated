"""rag.py

Simple in-memory Retrieval-Augmented Generation (RAG) engine for Nimbus website knowledge.
Loads catalog.json and context.md into discrete, searchable chunks and provides sub-millisecond
retrieval of relevant product details, pricing, policies, and comparisons.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger


@dataclass
class KnowledgeChunk:
    id: str
    title: str
    category: str  # "product", "policy", "company", "comparison"
    content: str
    keywords: Set[str] = field(default_factory=set)
    tokens: List[str] = field(default_factory=list)


STOP_WORDS: Set[str] = {
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "get", "got", "had", "has", "have", "he", "hello", "help", "her", "here", "hers", "hey",
    "hi", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like",
    "me", "my", "no", "not", "now", "of", "off", "ok", "okay", "on", "once", "one", "only",
    "or", "other", "our", "ours", "out", "over", "please", "shall", "she", "should", "so",
    "some", "such", "sure", "tell", "than", "thank", "thanks", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "us", "very", "was", "we", "well", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "yes", "you", "your", "yours",
}


class SimpleRAG:
    """Fast in-memory BM25-based retriever with domain-specific keyword boosting."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "nimbus-voice-agent-starter" / "data"
        self.data_dir = data_dir
        self.chunks: List[KnowledgeChunk] = []
        self._doc_lens: List[int] = []
        self._avg_doc_len: float = 0.0
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._product_names: Dict[str, str] = {}  # normalized alias -> chunk id

        self._load_and_index()

    def _tokenize(self, text: str, filter_stops: bool = False) -> List[str]:
        """Convert text to normalized lowercase word tokens."""
        clean = re.sub(r"[^\w\s\$\.]", " ", text.lower())
        tokens = [w for w in clean.split() if len(w) >= 2]
        if filter_stops:
            tokens = [w for w in tokens if w not in STOP_WORDS]
        return tokens

    def _load_and_index(self):
        consolidated_path = self.data_dir / "nimbus_consolidated_knowledge.json"
        catalog_path = self.data_dir / "catalog.json"

        if consolidated_path.exists():
            data = json.loads(consolidated_path.read_text(encoding="utf-8"))
        elif catalog_path.exists():
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
        else:
            data = {}

        # ── 1. Company Profile Chunk ──
        company = data.get("company", {})
        co_text = (
            f"Nimbus Company Profile:\n"
            f"Nimbus is an all-in-one cloud business software suite headquartered in {company.get('hq', 'Austin, TX')}, founded in {company.get('founded', '2014')}.\n"
            f"Stats: {company.get('stats', {}).get('Customers', '50,000+ businesses')}, 24 integrated cloud products across 8 categories.\n"
            f"Contact: Phone {company.get('contact', {}).get('phone', '+1 (800) 555-0142')}. Support email: {company.get('contact', {}).get('support', 'support@nimbus.example')}.\n"
            f"Mission: {company.get('about', 'Empower businesses with accessible, integrated cloud software.')}"
        )
        self.chunks.append(
            KnowledgeChunk(
                id="company_profile",
                title="Nimbus Company Overview & Headquarters",
                category="company",
                content=co_text,
                keywords={"nimbus", "company", "headquarters", "austin", "texas", "about", "office", "contact", "founded"},
            )
        )

        # ── 2. Precise Policy Chunks (Dynamically loaded from consolidated knowledge) ──
        policies = data.get("policies", {})
        policy_keyword_map = {
            "refund": {"refund", "money", "back", "guarantee", "30", "days", "return", "satisfaction", "reimbursement", "refund policy"},
            "free_trial": {"free", "trial", "14", "days", "card", "test", "demo", "try", "policy"},
            "freeTrial": {"free", "trial", "14", "days", "card", "test", "demo", "try", "policy"},
            "billing": {"billing", "payment", "annual", "monthly", "discount", "invoice", "credit", "card", "pay", "charge"},
            "cancellation": {"cancel", "cancellation", "terminate", "cancel subscription", "stop subscription", "cancel plan", "end subscription"},
            "sla": {"sla", "uptime", "availability", "reliability", "downtime", "maintenance", "99.9"},
            "security": {"security", "soc2", "gdpr", "encryption", "privacy", "safe", "compliance", "iso"},
            "support": {"support", "help", "contact", "phone", "hours", "email", "chat", "representative", "agent"},
        }

        for pol_key, pol_val in policies.items():
            if isinstance(pol_val, dict):
                title = pol_val.get("title", f"Nimbus {pol_key.replace('_', ' ').title()} Policy")
                summary = pol_val.get("summary", "")
                details = [f"{title}:", summary]
                for k, v in pol_val.items():
                    if k not in ["title", "summary"]:
                        k_label = k.replace("_", " ").title()
                        v_str = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                        details.append(f"- {k_label}: {v_str}")
                p_content = "\n".join(details)
            else:
                title = f"Nimbus {pol_key.replace('_', ' ').title()} Policy"
                p_content = f"{title}:\n{pol_val}"

            p_kws = policy_keyword_map.get(pol_key, {pol_key.lower(), "policy"})
            self.chunks.append(
                KnowledgeChunk(
                    id=f"policy_{pol_key.lower()}",
                    title=title,
                    category="policy",
                    content=p_content,
                    keywords=p_kws,
                )
            )

        # 2b. Standalone Data Residency Policy
        self.chunks.append(
            KnowledgeChunk(
                id="policy_data_residency",
                title="Nimbus Data Residency & Regional Hosting Policy",
                category="policy",
                content=(
                    "Nimbus Data Residency & Hosting Policy:\n"
                    "Customers can choose to host their data in our secure regional data centers located in the "
                    "United States (US), European Union (Frankfurt, EU), India (IN), or Australia (AU). "
                    "Data residency selections are made during account creation and ensure compliance with local "
                    "data localization regulations (including GDPR, Indian DPDP, and Australian Privacy Principles)."
                ),
                keywords={
                    "residency",
                    "data residency",
                    "hosting",
                    "region",
                    "regions",
                    "datacenter",
                    "data center",
                    "location",
                    "india",
                    "australia",
                    "frankfurt",
                    "sovereignty",
                    "policy",
                },
            )
        )

        # ── 3. Product Chunks (All 24 Products) ──
        products = data.get("products", [])
        price_index_data = []

        for prod in products:
            pid = prod.get("id", "")
            pname = prod.get("name", "")
            pcat = prod.get("category", "")
            ptag = prod.get("tagline", "")
            psummary = prod.get("summary", "")

            # Tier pricing extraction
            tiers_text_parts = []
            starter_monthly = None
            starter_annual = None
            pro_monthly = None
            pro_annual = None

            tiers_raw = prod.get("tiers", [])
            if isinstance(tiers_raw, dict):
                for tname, tinfo in tiers_raw.items():
                    m_price = tinfo.get("monthly")
                    a_price = tinfo.get("annual_monthly")
                    if tname.lower() == "free":
                        tiers_text_parts.append("Free Tier: $0/month (forever free basic features)")
                    elif tname.lower() == "starter":
                        starter_monthly = m_price
                        starter_annual = a_price
                        tiers_text_parts.append(
                            f"Starter Tier: ${m_price}/user/month (or ${a_price}/user/month billed annually)"
                        )
                    elif tname.lower() in ["pro", "professional"]:
                        pro_monthly = m_price
                        pro_annual = a_price
                        tiers_text_parts.append(
                            f"Professional Tier: ${m_price}/user/month (or ${a_price}/user/month billed annually)"
                        )
                    elif tname.lower() == "enterprise":
                        tiers_text_parts.append("Enterprise Tier: Custom pricing (contact sales team)")
            elif isinstance(tiers_raw, list):
                for t in tiers_raw:
                    tname = t.get("name", "")
                    m_price = t.get("priceMonthly")
                    a_price = t.get("priceAnnualMonthly")
                    if tname.lower() == "free":
                        tiers_text_parts.append("Free Tier: $0/month (forever free basic features)")
                    elif tname.lower() == "starter":
                        starter_monthly = m_price
                        starter_annual = a_price
                        tiers_text_parts.append(
                            f"Starter Tier: ${m_price}/user/month (or ${a_price}/user/month billed annually)"
                        )
                    elif tname.lower() in ["pro", "professional"]:
                        pro_monthly = m_price
                        pro_annual = a_price
                        tiers_text_parts.append(
                            f"Professional Tier: ${m_price}/user/month (or ${a_price}/user/month billed annually)"
                        )
                    elif tname.lower() == "enterprise":
                        tiers_text_parts.append("Enterprise Tier: Custom pricing (contact sales team)")

            tiers_str = "; ".join(tiers_text_parts)

            # Features & Add-ons
            feats = ", ".join(prod.get("keyFeatures", [])[:6])
            addons = []
            for ao in prod.get("addOns", []):
                ao_name = ao.get("name", "")
                ao_price = ao.get("price") or (f"${ao.get('priceMonthly')}/mo" if ao.get("priceMonthly") else "")
                ao_desc = ao.get("desc") or ao.get("description", "")
                addons.append(f"{ao_name}: {ao_price} ({ao_desc})")
            addons_str = "; ".join(addons) if addons else "None"

            # FAQs (Specific, non-redundant)
            faqs_list = []
            faqs_raw = prod.get("specificFaqs") or prod.get("faqs", [])
            for f in faqs_raw[:3]:
                q = f.get("question") or f.get("q", "")
                a = f.get("answer") or f.get("a", "")
                if q and a:
                    faqs_list.append(f"Q: {q} A: {a}")
            faqs_str = " | ".join(faqs_list) if faqs_list else "None"

            p_content = (
                f"Product: {pname} (Category: {pcat})\n"
                f"Tagline: {ptag}\n"
                f"Summary: {psummary}\n"
                f"Pricing Tiers: {tiers_str}\n"
                f"Key Features: {feats}\n"
                f"Add-ons: {addons_str}\n"
                f"Common FAQs: {faqs_str}"
            )

            # Build product keywords
            short_id = pid.replace("nimbus-", "").strip().lower()
            short_name = pname.replace("Nimbus", "").strip().lower()
            p_kws = {
                pid.lower(),
                pname.lower(),
                short_id,
                short_name,
                pcat.lower(),
                "product",
                "pricing",
                "tier",
                "starter",
                "professional",
            }
            for token in short_name.split():
                if len(token) >= 2:
                    p_kws.add(token)

            chunk_id = f"product_{pid}"
            self.chunks.append(
                KnowledgeChunk(
                    id=chunk_id,
                    title=f"{pname} - Pricing, Features & Details",
                    category="product",
                    content=p_content,
                    keywords=p_kws,
                )
            )

            # Product aliases and singular/plural variants
            aliases = {short_name, short_id, pname.lower(), pid.lower()}

            # Add singular & plural forms
            for a in list(aliases):
                if a.endswith("s") and len(a) > 3:
                    aliases.add(a[:-1])  # e.g., docs -> doc, campaigns -> campaign
                elif not a.endswith("s"):
                    aliases.add(a + "s")  # e.g., invoice -> invoices, quote -> quotes

            # Domain-specific conversational synonyms & natural speech terms
            domain_synonyms = {
                "docs": ["doc", "document", "documents"],
                "chat": ["chats", "messaging", "livechat"],
                "desk": ["desks", "helpdesk", "ticketing", "support ticket"],
                "recruit": ["recruiting", "recruitment", "ats", "hiring"],
                "people": ["hr", "human resources", "employee", "personnel"],
                "payroll": ["payrolls", "salary", "paycheck"],
                "knowledge": ["kb", "wiki", "docs hub"],
                "datapipe": [
                    "data pipe", "etl", "pipeline", "data type", "data five",
                    "data file", "data byte", "datatype", "datafive", "data-pipe"
                ],
                "vault": ["secrets", "passwords", "password manager"],
                "sso": ["single sign on", "identity", "saml"],
                "endpoint": ["endpoints", "mdm", "device management"],
                "boards": ["board", "kanban"],
                "projects": ["project", "task tracking"],
                "sites": ["site", "website builder"],
                "leads": ["lead", "lead generation"],
                "campaigns": ["campaign", "email marketing"],
                "invoice": ["invoices", "invoicing", "billing tool"],
                "expense": ["expenses", "expense management"],
            }
            if short_name in domain_synonyms:
                aliases.update(domain_synonyms[short_name])
            if short_id in domain_synonyms:
                aliases.update(domain_synonyms[short_id])

            for alias in aliases:
                self._product_names[alias] = chunk_id
                p_kws.add(alias)

            # Save for comparative chunk
            price_index_data.append({
                "name": pname,
                "id": pid,
                "category": pcat,
                "starter_monthly": starter_monthly,
                "starter_annual": starter_annual,
                "pro_monthly": pro_monthly,
                "pro_annual": pro_annual,
            })

        # ── 4. Comparative Pricing & Ranking Chunks (Consolidated Knowledge) ──
        rankings = data.get("price_rankings")
        if rankings:
            cheapest_starter = rankings.get("cheapest_starter", [])
            expensive_starter = rankings.get("most_expensive_starter", [])
            cheapest_pro = rankings.get("cheapest_pro", [])
            expensive_pro = rankings.get("most_expensive_pro", [])

            comp_lines = [
                "Nimbus Products Pricing Comparison & Rankings (from Consolidated Knowledge):",
                "",
                "1. Cheapest / Most Affordable Products (Starter Tier monthly):",
            ]
            for p in cheapest_starter:
                comp_lines.append(f"- {p['name']} ({p.get('category', '')}): ${p.get('price_monthly')}/user/month")

            comp_lines.append("")
            comp_lines.append("2. Most Expensive Products (Starter Tier monthly):")
            for p in expensive_starter:
                comp_lines.append(f"- {p['name']} ({p.get('category', '')}): ${p.get('price_monthly')}/user/month")

            comp_lines.append("")
            comp_lines.append("3. Professional Tier Price Range (from lowest to highest):")
            for p in cheapest_pro[:3]:
                comp_lines.append(f"- Lowest Pro: {p['name']} at ${p.get('price_monthly')}/user/month")
            for p in expensive_pro[:3]:
                comp_lines.append(f"- Highest Pro: {p['name']} at ${p.get('price_monthly')}/user/month")
        else:
            valid_starter = [p for p in price_index_data if p["starter_monthly"] is not None]
            valid_starter.sort(key=lambda x: x["starter_monthly"])
            cheapest_starter = valid_starter[:5]
            expensive_starter = sorted(valid_starter, key=lambda x: x["starter_monthly"], reverse=True)[:5]
            valid_pro = [p for p in price_index_data if p["pro_monthly"] is not None]
            valid_pro.sort(key=lambda x: x["pro_monthly"])
            cheapest_pro = valid_pro[:5]
            expensive_pro = sorted(valid_pro, key=lambda x: x["pro_monthly"], reverse=True)[:5]

            comp_lines = [
                "Nimbus Products Pricing Comparison & Rankings:",
                "",
                "1. Cheapest / Most Affordable Products (Starter Tier monthly):",
            ]
            for p in cheapest_starter:
                comp_lines.append(f"- {p['name']} ({p['category']}): ${p['starter_monthly']}/user/month")

            comp_lines.append("")
            comp_lines.append("2. Most Expensive Products (Starter Tier monthly):")
            for p in expensive_starter:
                comp_lines.append(f"- {p['name']} ({p['category']}): ${p['starter_monthly']}/user/month")

            comp_lines.append("")
            comp_lines.append("3. Professional Tier Price Range (from lowest to highest):")
            for p in cheapest_pro[:3]:
                comp_lines.append(f"- Lowest Pro: {p['name']} at ${p['pro_monthly']}/user/month")
            for p in expensive_pro[:3]:
                comp_lines.append(f"- Highest Pro: {p['name']} at ${p['pro_monthly']}/user/month")

        comp_lines.append("")
        comp_lines.append("General Rules for All 24 Products:")
        comp_lines.append("- Every product offers a Free tier ($0).")
        comp_lines.append("- Every paid product offers a 14-day free trial with no credit card required.")
        comp_lines.append("- Annual billing saves 20% compared to monthly billing.")
        comp_lines.append("- Enterprise tier is available for all products with custom volume pricing.")

        comp_content = "\n".join(comp_lines)
        self.chunks.append(
            KnowledgeChunk(
                id="pricing_comparison",
                title="Product Pricing Comparison & Rankings (Cheapest to Most Expensive)",
                category="comparison",
                content=comp_content,
                keywords={
                    "cheapest",
                    "expensive",
                    "most expensive",
                    "affordable",
                    "costly",
                    "budget",
                    "compare",
                    "comparison",
                    "lowest",
                    "highest",
                    "price range",
                    "pricing comparison",
                    "rank",
                    "ranking",
                    "catalog",
                    "all products",
                },
            )
        )

        # ── 5. Build BM25 / TF-IDF Vocabulary ──
        total_docs = len(self.chunks)
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.title + " " + chunk.content + " " + " ".join(chunk.keywords))
            chunk.tokens = tokens
            self._doc_lens.append(len(tokens))

            # Record document frequencies
            seen_in_doc = set(tokens)
            for t in seen_in_doc:
                self._df[t] = self._df.get(t, 0) + 1

        self._avg_doc_len = sum(self._doc_lens) / max(total_docs, 1)

        # Calculate IDFs
        for t, df in self._df.items():
            # Standard smoothed BM25 IDF
            self._idf[t] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

        source_name = consolidated_path.name if consolidated_path.exists() else catalog_path.name
        logger.info(f"📚 [SimpleRAG] Successfully linked and indexed {len(self.chunks)} knowledge chunks from '{source_name}'")

    def retrieve(self, query: str, top_k: int = 2) -> List[KnowledgeChunk]:
        """Retrieve top_k most relevant chunks for a user query.
        Combines BM25 scoring with exact product/policy keyword matching boosts.
        """
        if not query or not query.strip():
            return []

        clean_query = query.lower().strip()

        # Conversational commands / interruptions: do not trigger knowledge retrieval
        clean_no_punct = re.sub(r"[^\w\s]", "", clean_query).strip()
        CONVERSATIONAL_CMDS = {
            "stop", "hey stop", "stop it", "wait", "hold on", "pause", "one second",
            "hello", "hi", "hey", "ok", "okay", "yes", "no", "bye", "goodbye"
        }
        if clean_no_punct in CONVERSATIONAL_CMDS:
            return []

        q_tokens = self._tokenize(clean_query, filter_stops=True)

        # Check for comparative queries
        is_comparative = any(
            w in clean_query
            for w in [
                "cheapest",
                "expensive",
                "affordable",
                "compare",
                "comparison",
                "least",
                "lowest",
                "highest",
                "budget",
                "most costly",
                "price range",
            ]
        )

        # Check for specific product mentions
        matched_product_chunk_ids: Set[str] = set()
        for alias, cid in self._product_names.items():
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, clean_query):
                matched_product_chunk_ids.add(cid)

        # Fuzzy / phonetic product matching (especially over telephony audio)
        if not matched_product_chunk_ids:
            import difflib
            m = re.search(r"nimbus\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)", clean_no_punct)
            if m:
                after_nimbus = m.group(1).strip()
                valid_aliases = [a for a in self._product_names.keys() if len(a) >= 4]
                matches = difflib.get_close_matches(after_nimbus, valid_aliases, n=1, cutoff=0.6)
                if matches:
                    matched_product_chunk_ids.add(self._product_names[matches[0]])
                else:
                    first_word = after_nimbus.split()[0]
                    matches = difflib.get_close_matches(first_word, valid_aliases, n=1, cutoff=0.68)
                    if matches:
                        matched_product_chunk_ids.add(self._product_names[matches[0]])

        if not q_tokens and not is_comparative and not matched_product_chunk_ids:
            return []

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        scores: List[Tuple[float, KnowledgeChunk]] = []

        for i, chunk in enumerate(self.chunks):
            doc_len = self._doc_lens[i]
            score = 0.0

            # Term frequency counting
            tf_map: Dict[str, int] = {}
            for t in chunk.tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in q_tokens:
                if qt in tf_map:
                    tf = tf_map[qt]
                    idf = self._idf.get(qt, 1.0)
                    # BM25 formula
                    num = tf * (k1 + 1.0)
                    denom = tf + k1 * (1.0 - b + b * (doc_len / self._avg_doc_len))
                    score += idf * (num / denom)

            # ── Keyword & Domain Boosts ──
            # 1. Direct keyword match
            for qt in q_tokens:
                if qt in chunk.keywords:
                    score += 4.0

            # 2. Matched product boost
            if chunk.id in matched_product_chunk_ids:
                score += 15.0

            # 3. Comparative query boost
            if is_comparative and chunk.id == "pricing_comparison":
                score += 20.0

            # 4. Direct policy name match boost
            if chunk.category == "policy":
                for kw in chunk.keywords:
                    if kw in clean_query:
                        score += 5.0

            # 4b. Direct product name & keyword match boost (same parameter as policy)
            if chunk.category == "product":
                for kw in chunk.keywords:
                    if kw in clean_query:
                        score += 5.0

            if score >= 3.0:
                scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [chunk for _, chunk in scores[:top_k]]
        return results

    def format_context_for_prompt(self, query: str, top_k: int = 2) -> str:
        """Convenience method returning formatted knowledge block for LLM prompt injection."""
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return ""

        parts = ["[Retrieved Website Knowledge]:"]
        for c in chunks:
            parts.append(f"--- Information: {c.title} ---")
            parts.append(c.content.strip())
        parts.append("--- End of Retrieved Knowledge ---")
        return "\n\n".join(parts)


# Singleton instance
_GLOBAL_RAG: Optional[SimpleRAG] = None


def get_rag() -> SimpleRAG:
    """Get or create singleton SimpleRAG instance."""
    global _GLOBAL_RAG
    if _GLOBAL_RAG is None:
        _GLOBAL_RAG = SimpleRAG()
    return _GLOBAL_RAG

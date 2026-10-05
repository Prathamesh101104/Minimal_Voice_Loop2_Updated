"""semantic_cache.py

Semantic Response & Vector Embedding Cache for Nimbus voice agent.
Stores and reuses voice-ready responses using normalized vector embeddings and
cosine similarity, skipping database (RAG) lookups and LLM generation entirely.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

try:
    from pipecat.frames.frames import Frame
except ImportError:
    class Frame:
        pass

try:
    from nltk.stem import PorterStemmer
    _STEMMER: Optional[PorterStemmer] = PorterStemmer()
except Exception:
    _STEMMER = None


class CacheHitFrame(Frame):
    """Frame emitted when an incoming user query is resolved by SemanticResponseCache."""

    def __init__(
        self,
        query: str,
        response: str,
        similarity: float,
        canonical_query: str,
        category: str = "policy",
    ):
        super().__init__()
        self.query = query
        self.response = response
        self.similarity = similarity
        self.canonical_query = canonical_query
        self.category = category


class CacheHitResult(tuple):
    """3-tuple (response, similarity, canonical_query) with .category attribute."""

    def __new__(
        cls,
        response: str,
        similarity: float,
        canonical_query: str,
        category: str = "policy",
    ):
        return super().__new__(cls, (response, similarity, canonical_query))

    def __init__(
        self,
        response: str,
        similarity: float,
        canonical_query: str,
        category: str = "policy",
    ):
        self.response = response
        self.similarity = similarity
        self.canonical_query = canonical_query
        self.category = category


# Stop words to downweight during vectorization
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

# Key product entity keywords to prevent cross-product mismatch
PRODUCT_ENTITIES: Set[str] = {
    "crm", "leads", "quote", "campaigns", "social", "sites",
    "books", "invoice", "expense", "people", "recruit", "payroll",
    "desk", "chat", "projects", "docs", "analytics", "vault",
    "sso", "datapipe", "endpoint", "dashboards", "boards", "knowledge", "platform",
}

POLICY_ENTITIES: Set[str] = {
    "refund", "trial", "try", "cancel", "cancellation", "sla", "security",
    "support", "annual", "cheapest", "expensive", "headquarters", "headquarter",
    "location", "located", "based", "residency", "billing", "payment", "uptime",
    "compliance", "gdpr", "freetier", "catalog", "overview", "policy", "policies",
    "price", "secure", "compliant",
}

SYNONYM_REPLACEMENTS = [
    # Overview intent
    (r"\b(tell me about|what is|what are|explain|overview of|info about|information about|how does .+ work|what does .+ do)\b", "overview"),
    # Pricing intent
    (r"\b(how much does|how much is|what is the price of|what does .+ cost|what is the cost of|pricing for)\b", "price"),
    (r"\b(pricing|cost|costs|rate|rates|fee|fees|charge|charges)\b", "price"),
    # Refund intent
    (r"\b(money back|money-back|return policy|returns policy|refund policy|refunds policy|reimbursement|satisfaction guarantee)\b", "refund"),
    # Free trial intent
    (r"\b(free trial|free trials|try\s+\w+\s+for free|try for free|try it out|try it|trial period|trial periods|test drive)\b", "trial"),
    # Headquarters / Location
    (r"\b(where\s+(?:is|are)\s+(?:nimbus|you|your\s+(?:company|office|headquarters))(?:\s+located|\s+based)?|where\s+are\s+your\s+headquarters|where\s+is\s+your\s+office|where\s+is\s+nimbus\s+based)\b", "headquarters location"),
    (r"\b(located|location|headquarters|headquarter)\b", "location"),
    # Cheapest / Affordable
    (r"\b(cheapest|most affordable|lowest price|lowest prices|lowest cost|lowest costs|least expensive|budget friendly)\b", "cheapest affordable"),
    # Most Expensive
    (r"\b(most expensive|highest price|highest prices|highest cost|highest costs|costliest|most costly)\b", "most expensive"),
    # Cancellation
    (r"\b(cancel my subscription|cancel subscription|cancellation policy|how to cancel|can i cancel|cancel my plan)\b", "cancel cancellation"),
    # Annual discount
    (r"\b(annual discount|yearly discount|annual plan discount|yearly billing discount)\b", "annual discount"),
    # Data residency
    (r"\b(data residency|regional hosting|data hosting|data centers?|where is (?:customer )?data (?:hosted|stored)|data localization|where do you store data|where is data stored|regional data)\b", "data residency"),
    # SLA & Uptime
    (r"\b(sla|uptime|service level agreement|uptime guarantee|uptime policy|availability guarantee|downtime)\b", "sla uptime"),
    # Billing policy
    (r"\b(billing policy|billing policies|billing cycle|billing cycles|payment methods?|how do you bill|payment options?|accepted payments?|how to pay)\b", "billing payment"),
    # Support policy
    (r"\b(support policy|support policies|customer support|support hours|technical support|help desk|reach support|contact support)\b", "support"),
    # Security policy
    (r"\b(security policy|security policies|soc 2|iso 27001|gdpr|ccpa|encryption|compliance policy|secure and compliant|security and compliance)\b", "security compliance"),
    # Free tier
    (r"\b(free tier|free plan|free plans|free products?|forever free|free version)\b", "freetier free plan"),
    # Catalog
    (r"\b(product catalog|all products|product suite|what products|what tools|what software)\b", "catalog products"),
    # Product domain synonyms
    (r"\b(helpdesk|ticketing|support tickets?)\b", "desk"),
    (r"\b(livechat|messaging|chat tool)\b", "chat"),
    (r"\b(human resources|employee management|personnel)\b", "people"),
    (r"\b(applicant tracking|recruiting|hiring|ats)\b", "recruit"),
    (r"\b(paycheck|salaries|salary)\b", "payroll"),
    (r"\b(documentation|documents|document tool)\b", "docs"),
    (r"\b(wiki|help center|knowledge base|docs hub)\b", "knowledge"),
    (r"\b(data pipe|data pipeline|etl)\b", "datapipe"),
    (r"\b(secrets?|password manager|passwords)\b", "vault"),
    (r"\b(single sign on|identity|saml)\b", "sso"),
    (r"\b(device management|mdm|endpoints)\b", "endpoint"),
    (r"\b(kanban|whiteboards?)\b", "boards"),
    (r"\b(task tracking|gantt)\b", "projects"),
    (r"\b(landing pages?|website builder)\b", "sites"),
    (r"\b(lead generation|dialer|outreach)\b", "leads"),
    (r"\b(email marketing|marketing automation)\b", "campaigns"),
    (r"\b(invoicing|billing tool)\b", "invoice"),
    (r"\b(receipts?|expense management)\b", "expense"),
]


class VectorEmbeddingEngine:
    """Fast, deterministic sub-millisecond local vector embedding engine.
    Normalizes semantic intents, computes subword character 3-grams, and projects
    into a normalized, dense float32 vector space (dim=384).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def normalize_intent(self, text: str) -> str:
        """Map synonyms and conversational phrasing to standardized semantic intent tokens."""
        s = text.lower()
        for pattern, replacement in SYNONYM_REPLACEMENTS:
            s = re.sub(pattern, replacement, s)
        return s

    def _tokenize(self, text: str) -> List[str]:
        clean = re.sub(r"[^\w\s\$\.]", " ", text.lower())
        tokens = [w for w in clean.split() if len(w) >= 2]
        if _STEMMER:
            tokens = [_STEMMER.stem(t) for t in tokens]
        return tokens

    def _extract_features(self, text: str) -> Dict[int, float]:
        """Extract word tokens, character 3-grams, and domain boosts into hashed feature buckets."""
        features: Dict[int, float] = {}
        normalized = self.normalize_intent(text)
        tokens = self._tokenize(normalized)
        if not tokens:
            return features

        # 1. Word token unigrams (with stop-word downweighting)
        for t in tokens:
            weight = 0.3 if t in STOP_WORDS else 2.0
            h = int(hashlib.md5(f"w_{t}".encode("utf-8")).hexdigest(), 16) % self.dim
            features[h] = features.get(h, 0.0) + weight

        # 2. Subword character 3-grams (for morphological & spelling invariance)
        padded = f" {normalized} "
        for i in range(len(padded) - 2):
            trigram = padded[i : i + 3]
            h = int(hashlib.md5(f"tri_{trigram}".encode("utf-8")).hexdigest(), 16) % self.dim
            features[h] = features.get(h, 0.0) + 0.5

        # 3. Domain Entity Anchors (heavy weight to guarantee topic fidelity)
        for t in tokens:
            if t in PRODUCT_ENTITIES or t in POLICY_ENTITIES:
                h = int(hashlib.md5(f"entity_{t}".encode("utf-8")).hexdigest(), 16) % self.dim
                features[h] = features.get(h, 0.0) + 4.0

        return features

    def embed(self, text: str) -> np.ndarray:
        """Generate an L2-normalized float32 vector embedding."""
        vec = np.zeros(self.dim, dtype=np.float32)
        features = self._extract_features(text)
        if not features:
            return vec

        for idx, val in features.items():
            vec[idx] = val

        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec /= norm
        return vec

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two unit-normalized vectors."""
        dot = float(np.dot(vec_a, vec_b))
        return max(0.0, min(1.0, dot))


@dataclass
class CacheEntry:
    id: str
    canonical_query: str
    response: str
    category: str  # "policy", "product", "company", "comparison", "dynamic"
    examples: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    example_embeddings: List[List[float]] = field(default_factory=list)
    required_entities: List[str] = field(default_factory=list)
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)


class SemanticResponseCache:
    """In-memory semantic cache with persistent JSON backing.
    Stores and reuses voice-ready responses to skip RAG retrieval and LLM generation.
    """

    def __init__(self, cache_file: Optional[Path] = None, threshold: float = 0.72):
        if cache_file is None:
            cache_file = (
                Path(__file__).resolve().parent
                / "nimbus-voice-agent-starter"
                / "data"
                / "semantic_cache.json"
            )
        self.cache_file = cache_file
        self.threshold = threshold
        self.engine = VectorEmbeddingEngine(dim=384)
        self.entries: Dict[str, CacheEntry] = {}
        self._cached_vectors: Dict[str, np.ndarray] = {}
        self._example_vectors: Dict[str, List[np.ndarray]] = {}

        self._load_or_initialize()

    def _extract_entities(self, text: str) -> Set[str]:
        normalized = self.engine.normalize_intent(text)
        tokens = set(re.findall(r"\b[a-z]{3,}\b", normalized.lower())) | set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        entities = (tokens & PRODUCT_ENTITIES) | (tokens & POLICY_ENTITIES)
        return entities

    def _load_or_initialize(self):
        """Load pre-warmed canonical responses and merge with persisted dynamic entries."""
        # 1. Initialize all pre-warmed entries from code first
        self._initialize_prewarmed_entries()

        # 2. If persistent cache file exists, load and merge any dynamic entries
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text(encoding="utf-8"))
                raw_entries = data.get("entries", [])
                for item in raw_entries:
                    entry_id = item.get("id")
                    # If this is a dynamic entry not in pre-warmed, preserve it
                    if entry_id and entry_id not in self.entries and "examples" in item:
                        entry = CacheEntry(**item)
                        self.entries[entry.id] = entry
                        self._cached_vectors[entry.id] = np.array(entry.embedding, dtype=np.float32)
                        self._example_vectors[entry.id] = [
                            np.array(ex_v, dtype=np.float32) for ex_v in entry.example_embeddings
                        ]
            except Exception as e:
                logger.warning(f"Failed to load semantic cache from {self.cache_file}: {e}")

        # 3. Persist the complete set
        self.persist()
        logger.info(f"⚡ [SemanticCache] Loaded {len(self.entries)} pre-warmed & cached entries")

    def _initialize_prewarmed_entries(self):
        """Pre-populate high-frequency Nimbus queries with verified, voice-optimized answers and exemplars."""
        seed_data: List[Dict[str, Any]] = [
            # ══════════════════════════════════════════════════════════════════
            # 1. ALL COMPANY POLICIES (FROM CONTEXT.MD)
            # ══════════════════════════════════════════════════════════════════
            {
                "id": "policy_refund",
                "canonical_query": "What is your refund policy?",
                "category": "policy",
                "required_entities": ["refund"],
                "examples": [
                    "What is your refund policy?",
                    "What are your refund policies?",
                    "Can I get a refund?",
                    "Do you offer refunds?",
                    "Can I get my money back?",
                    "What is your money back guarantee?",
                    "How does your refund policy work?",
                    "Do you have a 30-day money-back guarantee?",
                    "Tell me about your refund policy",
                    "What is the policy on refunds?",
                ],
                "response": (
                    "Nimbus offers a 30-day money-back guarantee for all new subscriptions on both monthly and annual plans. "
                    "If you are not satisfied within your first thirty days, you can submit a ticket in your admin console for a full refund."
                ),
            },
            {
                "id": "policy_free_trial",
                "canonical_query": "Do you offer a free trial?",
                "category": "policy",
                "required_entities": ["trial", "try"],
                "examples": [
                    "Do you offer a free trial?",
                    "Do you have free trials?",
                    "Can I try Nimbus for free?",
                    "Is there a free trial available?",
                    "Do you have a trial period?",
                    "Do I need a credit card to try it?",
                    "How long is the free trial?",
                    "What is your free trial policy?",
                    "Tell me about your trial periods",
                ],
                "response": (
                    "All Nimbus products include a 14-day free trial with full access to premium features and no credit card required. "
                    "At the end of the trial, you can seamlessly upgrade to a paid plan without losing any setup data."
                ),
            },
            {
                "id": "policy_data_residency",
                "canonical_query": "What is your data residency policy?",
                "category": "policy",
                "required_entities": ["residency"],
                "examples": [
                    "What is your data residency policy?",
                    "What are your data residency policies?",
                    "Where is customer data stored?",
                    "Where are your data centers located?",
                    "Can I choose my data center location?",
                    "Do you have data centers in India, US, or Europe?",
                    "What are your regional hosting options?",
                    "How does data residency work?",
                    "Where do you host data?",
                    "Tell me about regional data hosting",
                    "Is data hosted locally in Australia or EU?",
                ],
                "response": (
                    "Customers can choose to host their data in our secure regional data centers located in the United States, "
                    "European Union (Frankfurt), India, or Australia. You make this selection during account creation to comply with local data regulations."
                ),
            },
            {
                "id": "policy_billing",
                "canonical_query": "What is your billing and payment policy?",
                "category": "policy",
                "required_entities": ["billing"],
                "examples": [
                    "What is your billing policy?",
                    "What are your billing policies?",
                    "What payment methods do you accept?",
                    "What payment options are available?",
                    "How does billing work?",
                    "Do you accept credit cards and PayPal?",
                    "Can I pay by wire transfer?",
                    "What are your billing cycles?",
                    "How do you charge for subscriptions?",
                    "Can we pay annually or monthly?",
                ],
                "response": (
                    "Nimbus offers both monthly and annual billing cycles, with annual plans receiving a twenty percent discount. "
                    "We accept all major credit cards, PayPal, and wire transfers for enterprise accounts exceeding 5,000 dollars."
                ),
            },
            {
                "id": "policy_cancellation",
                "canonical_query": "What is your cancellation policy?",
                "category": "policy",
                "required_entities": ["cancel"],
                "examples": [
                    "What is your cancellation policy?",
                    "What are your cancellation policies?",
                    "How do I cancel my subscription?",
                    "Can I cancel anytime?",
                    "How do I close my account?",
                    "Are there fees to cancel?",
                    "What happens to my data if I cancel?",
                    "Can I downgrade my plan?",
                    "How long is data kept after cancellation?",
                ],
                "response": (
                    "Administrators can cancel subscriptions or downgrade plans anytime directly in the billing portal without penalty. "
                    "Your account remains active until the end of your billing cycle, and your data is retained for sixty days before permanent deletion."
                ),
            },
            {
                "id": "policy_sla",
                "canonical_query": "What is your SLA and uptime policy?",
                "category": "policy",
                "required_entities": ["sla"],
                "examples": [
                    "What is your SLA policy?",
                    "What are your SLA policies?",
                    "What is your uptime guarantee?",
                    "What is your service level agreement?",
                    "What happens if there is downtime?",
                    "Do you offer service credits for outages?",
                    "What is your uptime SLA?",
                    "What is the availability guarantee?",
                ],
                "response": (
                    "Nimbus guarantees a 99.9 percent monthly uptime for standard plans and 99.99 percent for enterprise tiers. "
                    "If we fail to meet this SLA, affected customers receive service credits ranging from ten to fifty percent of their monthly fee."
                ),
            },
            {
                "id": "policy_security",
                "canonical_query": "What is your security and compliance policy?",
                "category": "policy",
                "required_entities": ["security"],
                "examples": [
                    "What is your security policy?",
                    "What are your security policies?",
                    "Is Nimbus secure and compliant?",
                    "Are you SOC 2 compliant?",
                    "Are you GDPR and CCPA compliant?",
                    "What encryption standards do you use?",
                    "Is customer data encrypted?",
                    "Do you conduct penetration testing?",
                ],
                "response": (
                    "Nimbus is SOC 2 Type 2 certified, ISO 27001 compliant, and adheres to GDPR and CCPA. "
                    "Customer data is encrypted with TLS 1.3 in transit and AES-256 at rest, backed by regular third-party penetration testing."
                ),
            },
            {
                "id": "policy_support",
                "canonical_query": "What is your customer support policy?",
                "category": "policy",
                "required_entities": ["support"],
                "examples": [
                    "What is your support policy?",
                    "What are your support policies?",
                    "How can I contact customer support?",
                    "What are your support hours?",
                    "Do you offer 24/7 support?",
                    "How do I reach your help desk?",
                    "Do you have phone support?",
                    "What support options do you provide?",
                    "Do enterprise accounts get dedicated support?",
                ],
                "response": (
                    "Standard plans include email and live chat support during regular business hours. "
                    "Enterprise customers receive priority phone support and a dedicated technical account manager, with 24/7 access to our knowledge base and forums."
                ),
            },
            {
                "id": "policy_annual_discount",
                "canonical_query": "Do you offer an annual discount?",
                "category": "policy",
                "required_entities": ["annual"],
                "examples": [
                    "Do you offer an annual discount?",
                    "Do you offer annual discounts?",
                    "Is there a discount for yearly billing?",
                    "How much do I save if I pay annually?",
                    "Do you have yearly plan discounts?",
                    "What is the annual savings rate?",
                ],
                "response": (
                    "Yes, choosing annual billing saves you twenty percent compared to monthly billing across all Nimbus products."
                ),
            },
            {
                "id": "company_overview",
                "canonical_query": "Where is Nimbus located and headquartered?",
                "category": "company",
                "required_entities": ["headquarters", "location"],
                "examples": [
                    "Where is Nimbus located?",
                    "Where are your company offices located?",
                    "Where are your headquarters?",
                    "Where is your office based?",
                    "Where is Nimbus based?",
                    "Tell me about Nimbus company headquarters",
                ],
                "response": (
                    "Nimbus is headquartered in Austin, Texas, and was founded in 2014. "
                    "We currently power over 50,000 businesses across 120 countries with our unified cloud software suite."
                ),
            },

            # ══════════════════════════════════════════════════════════════════
            # 2. PRODUCT COMPARISONS & RANKINGS (SINGULAR & PLURAL FORMS)
            # ══════════════════════════════════════════════════════════════════
            {
                "id": "comp_cheapest_single",
                "canonical_query": "What is your cheapest product?",
                "category": "comparison",
                "required_entities": ["cheapest"],
                "examples": [
                    "What is your cheapest product?",
                    "What is the single cheapest product you offer?",
                    "Which product is the cheapest?",
                    "What is your most affordable product?",
                    "What is your lowest price product?",
                    "Which single tool has the lowest price?",
                    "What is the cheapest software tool you have?",
                    "What is your lowest cost item?",
                    "Which software is cheapest?",
                ],
                "response": (
                    "Our cheapest product is Nimbus Docs, starting at eight dollars per user per month on the Starter tier, "
                    "or six dollars and forty cents billed annually."
                ),
            },
            {
                "id": "comp_cheapest_plural",
                "canonical_query": "What are your cheapest products?",
                "category": "comparison",
                "required_entities": ["cheapest"],
                "examples": [
                    "What are your cheapest products?",
                    "What are the top three cheapest products?",
                    "Which products are the most affordable?",
                    "What are your lowest cost tools?",
                    "What are the most budget friendly products in your catalog?",
                    "Which software tools have the lowest pricing?",
                    "Tell me your top affordable products",
                    "What are your cheapest plans?",
                ],
                "response": (
                    "Our three most affordable products on the Starter tier are Nimbus Docs at eight dollars, "
                    "Nimbus People at ten dollars, and Nimbus Invoice at ten dollars per user per month."
                ),
            },
            {
                "id": "comp_expensive_single",
                "canonical_query": "What is your most expensive product?",
                "category": "comparison",
                "required_entities": ["expensive"],
                "examples": [
                    "What is your most expensive product?",
                    "Which product costs the most?",
                    "What is the single most expensive product in your suite?",
                    "What is your costliest product?",
                    "Which tool has the highest price?",
                    "What is the most expensive software you sell?",
                ],
                "response": (
                    "Our most expensive products on the Starter tier are Nimbus Leads and Nimbus DataPipe at twenty-five dollars per user per month, "
                    "and sixty-five dollars on the Professional tier."
                ),
            },
            {
                "id": "comp_expensive_plural",
                "canonical_query": "What are your most expensive products?",
                "category": "comparison",
                "required_entities": ["expensive"],
                "examples": [
                    "What are your most expensive products?",
                    "Which products are the most expensive?",
                    "What are your highest priced tools?",
                    "Top most expensive products in your suite?",
                    "What are the costliest software products you offer?",
                    "Which products have the highest monthly fees?",
                ],
                "response": (
                    "Our highest priced products on the Starter tier are Nimbus Leads, Nimbus DataPipe, Nimbus Sites, and Nimbus Recruit, "
                    "starting at twenty-five dollars per user per month, and up to sixty-five dollars on Professional."
                ),
            },
            {
                "id": "comp_free_tier",
                "canonical_query": "Do you offer free products or a free tier?",
                "category": "comparison",
                "required_entities": ["freetier"],
                "examples": [
                    "Do you have a free product?",
                    "Do you have free products?",
                    "Do you offer a free plan?",
                    "Do you offer free plans?",
                    "Is there a free version of your software?",
                    "What products have a free tier?",
                    "Can I use Nimbus for free?",
                    "Tell me about your free tier options",
                ],
                "response": (
                    "Every single Nimbus product includes a forever free tier at zero dollars per month for basic features, "
                    "plus a 14-day free trial on paid plans with no credit card required."
                ),
            },
            {
                "id": "comp_catalog_overview",
                "canonical_query": "What products does Nimbus offer?",
                "category": "comparison",
                "required_entities": ["catalog"],
                "examples": [
                    "What products do you offer?",
                    "What product does Nimbus sell?",
                    "What software products do you have?",
                    "Tell me about your product suite?",
                    "What is your product catalog?",
                    "Give me an overview of your products",
                    "What kind of applications are in the Nimbus suite?",
                ],
                "response": (
                    "Nimbus offers twenty-four integrated cloud applications across Sales, Operations, Finance, HR, and IT, "
                    "including Nimbus CRM, Books, Payroll, Desk, Docs, People, and Leads."
                ),
            },

            # ══════════════════════════════════════════════════════════════════
            # 3. ALL 24 PRODUCT OVERVIEWS / FEATURES / WHAT IS IT
            # ══════════════════════════════════════════════════════════════════
            {
                "id": "info_crm",
                "canonical_query": "What is Nimbus CRM?",
                "category": "product",
                "required_entities": ["crm"],
                "examples": [
                    "What is Nimbus CRM?",
                    "Can you tell me about Nimbus CRM?",
                    "Tell me about Nimbus CRM",
                    "What does Nimbus CRM do?",
                    "Tell me about CRM",
                    "Give me an overview of Nimbus CRM",
                ],
                "response": (
                    "Nimbus CRM is our sales and contact management hub. It centralizes your customer data, tracks deals across multiple visual pipelines, and automates daily sales tasks to help your team close deals faster."
                ),
            },
            {
                "id": "info_leads",
                "canonical_query": "What is Nimbus Leads?",
                "category": "product",
                "required_entities": ["leads"],
                "examples": [
                    "What is Nimbus Leads?",
                    "Can you tell me about Nimbus Leads?",
                    "Tell me about Nimbus Leads",
                    "What does Nimbus Leads do?",
                    "How does Nimbus Leads work?",
                ],
                "response": (
                    "Nimbus Leads automates outbound sales with multi-channel outreach sequences, a built-in power dialer, and predictive lead scoring to connect reps with high-value prospects."
                ),
            },
            {
                "id": "info_quote",
                "canonical_query": "What is Nimbus Quote?",
                "category": "product",
                "required_entities": ["quote"],
                "examples": [
                    "What is Nimbus Quote?",
                    "Can you tell me about Nimbus Quote?",
                    "Tell me about Nimbus Quote",
                    "What does Nimbus Quote do?",
                ],
                "response": (
                    "Nimbus Quote is a CPQ and proposal solution that accelerates sales cycles with automated quote generation, interactive deal rooms, and secure e-signatures."
                ),
            },
            {
                "id": "info_campaigns",
                "canonical_query": "What is Nimbus Campaigns?",
                "category": "product",
                "required_entities": ["campaigns"],
                "examples": [
                    "What is Nimbus Campaigns?",
                    "Can you tell me about Nimbus Campaigns?",
                    "Tell me about Nimbus Campaigns",
                    "What does Nimbus Campaigns do?",
                ],
                "response": (
                    "Nimbus Campaigns is an email marketing automation platform to design responsive email campaigns, segment audiences based on behavior, and build multi-step customer journeys."
                ),
            },
            {
                "id": "info_social",
                "canonical_query": "What is Nimbus Social?",
                "category": "product",
                "required_entities": ["social"],
                "examples": [
                    "What is Nimbus Social?",
                    "Can you tell me about Nimbus Social?",
                    "Tell me about Nimbus Social",
                    "What does Nimbus Social do?",
                ],
                "response": (
                    "Nimbus Social is a unified social media management tool to draft, schedule, and publish posts across platforms while monitoring brand mentions and engagement analytics."
                ),
            },
            {
                "id": "info_sites",
                "canonical_query": "What is Nimbus Sites?",
                "category": "product",
                "required_entities": ["sites"],
                "examples": [
                    "What is Nimbus Sites?",
                    "Can you tell me about Nimbus Sites?",
                    "Tell me about Nimbus Sites",
                    "What does Nimbus Sites do?",
                ],
                "response": (
                    "Nimbus Sites is a visual landing page and website builder with mobile-optimized templates, integrated lead capture forms, and native A/B testing."
                ),
            },
            {
                "id": "info_books",
                "canonical_query": "What is Nimbus Books?",
                "category": "product",
                "required_entities": ["books"],
                "examples": [
                    "What is Nimbus Books?",
                    "Can you tell me about Nimbus Books?",
                    "Tell me about Nimbus Books",
                    "What does Nimbus Books do?",
                ],
                "response": (
                    "Nimbus Books simplifies business finances with automated double-entry accounting, bank reconciliation, tax-ready financial statements, and multi-currency support."
                ),
            },
            {
                "id": "info_invoice",
                "canonical_query": "What is Nimbus Invoice?",
                "category": "product",
                "required_entities": ["invoice"],
                "examples": [
                    "What is Nimbus Invoice?",
                    "Can you tell me about Nimbus Invoice?",
                    "Tell me about Nimbus Invoice",
                    "What does Nimbus Invoice do?",
                ],
                "response": (
                    "Nimbus Invoice generates professional invoices, automates recurring billing, and accepts fast online payments with automated payment reminders."
                ),
            },
            {
                "id": "info_expense",
                "canonical_query": "What is Nimbus Expense?",
                "category": "product",
                "required_entities": ["expense"],
                "examples": [
                    "What is Nimbus Expense?",
                    "Can you tell me about Nimbus Expense?",
                    "Tell me about Nimbus Expense",
                    "What does Nimbus Expense do?",
                ],
                "response": (
                    "Nimbus Expense streamlines corporate spend management with smart receipt scanning, multi-level approval workflows, and automated employee reimbursements."
                ),
            },
            {
                "id": "info_people",
                "canonical_query": "What is Nimbus People?",
                "category": "product",
                "required_entities": ["people"],
                "examples": [
                    "What is Nimbus People?",
                    "Can you tell me about Nimbus People?",
                    "Tell me about Nimbus People",
                    "What does Nimbus People do?",
                ],
                "response": (
                    "Nimbus People is a centralized HRIS managing the entire employee lifecycle with digital onboarding, time-off tracking, document storage, and interactive company org charts."
                ),
            },
            {
                "id": "info_recruit",
                "canonical_query": "What is Nimbus Recruit?",
                "category": "product",
                "required_entities": ["recruit"],
                "examples": [
                    "What is Nimbus Recruit?",
                    "Can you tell me about Nimbus Recruit?",
                    "Tell me about Nimbus Recruit",
                    "What does Nimbus Recruit do?",
                ],
                "response": (
                    "Nimbus Recruit is an applicant tracking system that streamlines hiring with customizable candidate pipelines, job board syndication, resume parsing, and interview scheduling."
                ),
            },
            {
                "id": "info_payroll",
                "canonical_query": "What is Nimbus Payroll?",
                "category": "product",
                "required_entities": ["payroll"],
                "examples": [
                    "What is Nimbus Payroll?",
                    "Can you tell me about Nimbus Payroll?",
                    "Tell me about Nimbus Payroll",
                    "What does Nimbus Payroll do?",
                ],
                "response": (
                    "Nimbus Payroll automates direct deposit payroll runs, payroll tax filings, and contractor payments with complete legal compliance."
                ),
            },
            {
                "id": "info_desk",
                "canonical_query": "What is Nimbus Desk?",
                "category": "product",
                "required_entities": ["desk"],
                "examples": [
                    "What is Nimbus Desk?",
                    "Can you tell me about Nimbus Desk?",
                    "Tell me about Nimbus Desk",
                    "What does Nimbus Desk do?",
                ],
                "response": (
                    "Nimbus Desk is an omnichannel customer support helpdesk with shared ticketing inboxes, SLA tracking, intelligent routing, and escalation rules."
                ),
            },
            {
                "id": "info_chat",
                "canonical_query": "What is Nimbus Chat?",
                "category": "product",
                "required_entities": ["chat"],
                "examples": [
                    "What is Nimbus Chat?",
                    "Can you tell me about Nimbus Chat?",
                    "Tell me about Nimbus Chat",
                    "What does Nimbus Chat do?",
                ],
                "response": (
                    "Nimbus Chat provides live website chat and conversational AI bots to engage visitors, qualify inbound leads, and answer customer queries around the clock."
                ),
            },
            {
                "id": "info_knowledge",
                "canonical_query": "What is Nimbus Knowledge?",
                "category": "product",
                "required_entities": ["knowledge"],
                "examples": [
                    "What is Nimbus Knowledge?",
                    "Can you tell me about Nimbus Knowledge?",
                    "Tell me about Nimbus Knowledge",
                    "What does Nimbus Knowledge do?",
                ],
                "response": (
                    "Nimbus Knowledge enables companies to create, organize, and publish searchable self-service help centers, customer knowledge bases, and product documentation."
                ),
            },
            {
                "id": "info_projects",
                "canonical_query": "What is Nimbus Projects?",
                "category": "product",
                "required_entities": ["projects"],
                "examples": [
                    "What is Nimbus Projects?",
                    "Can you tell me about Nimbus Projects?",
                    "Tell me about Nimbus Projects",
                    "What does Nimbus Projects do?",
                ],
                "response": (
                    "Nimbus Projects is a comprehensive project management solution featuring task tracking, interactive Gantt charts, sprint planning, and team workload allocation."
                ),
            },
            {
                "id": "info_docs",
                "canonical_query": "What is Nimbus Docs?",
                "category": "product",
                "required_entities": ["docs"],
                "examples": [
                    "What is Nimbus Docs?",
                    "Can you tell me about Nimbus Docs?",
                    "Tell me about Nimbus Docs",
                    "What does Nimbus Docs do?",
                    "Can you tell me about Nimbus docs?",
                    "Okay. Can you tell me about Nimbus docs?",
                    "Tell me about docs",
                ],
                "response": (
                    "Nimbus Docs is a modern collaborative workspace for team documentation, meeting notes, and company wikis with real-time multiplayer editing and rich media embedding."
                ),
            },
            {
                "id": "info_boards",
                "canonical_query": "What is Nimbus Boards?",
                "category": "product",
                "required_entities": ["boards"],
                "examples": [
                    "What is Nimbus Boards?",
                    "Can you tell me about Nimbus Boards?",
                    "Tell me about Nimbus Boards",
                    "What does Nimbus Boards do?",
                ],
                "response": (
                    "Nimbus Boards offers agile visual workspaces combining structured Kanban sprint boards with freeform digital whiteboards for brainstorming and mapping workflows."
                ),
            },
            {
                "id": "info_analytics",
                "canonical_query": "What is Nimbus Analytics?",
                "category": "product",
                "required_entities": ["analytics"],
                "examples": [
                    "What is Nimbus Analytics?",
                    "Can you tell me about Nimbus Analytics?",
                    "Tell me about Nimbus Analytics",
                    "What does Nimbus Analytics do?",
                ],
                "response": (
                    "Nimbus Analytics is a self-serve business intelligence tool to build interactive visual reports, model complex metrics, and uncover actionable business insights without code."
                ),
            },
            {
                "id": "info_dashboards",
                "canonical_query": "What is Nimbus Dashboards?",
                "category": "product",
                "required_entities": ["dashboards"],
                "examples": [
                    "What is Nimbus Dashboards?",
                    "Can you tell me about Nimbus Dashboards?",
                    "Tell me about Nimbus Dashboards",
                    "What does Nimbus Dashboards do?",
                ],
                "response": (
                    "Nimbus Dashboards displays real-time operational metrics, team KPIs, and live status monitors with customizable widgets and instant threshold alerts."
                ),
            },
            {
                "id": "info_datapipe",
                "canonical_query": "What is Nimbus DataPipe?",
                "category": "product",
                "required_entities": ["datapipe"],
                "examples": [
                    "What is Nimbus DataPipe?",
                    "Can you tell me about Nimbus DataPipe?",
                    "Tell me about Nimbus DataPipe",
                    "What does Nimbus DataPipe do?",
                ],
                "response": (
                    "Nimbus DataPipe is a no-code ETL platform that connects cloud applications with data warehouses to extract, transform, and sync business data automatically."
                ),
            },
            {
                "id": "info_vault",
                "canonical_query": "What is Nimbus Vault?",
                "category": "product",
                "required_entities": ["vault"],
                "examples": [
                    "What is Nimbus Vault?",
                    "Can you tell me about Nimbus Vault?",
                    "Tell me about Nimbus Vault",
                    "What does Nimbus Vault do?",
                ],
                "response": (
                    "Nimbus Vault securely manages company passwords, API keys, and environment secrets with zero-knowledge AES-256 encryption and granular role-based access."
                ),
            },
            {
                "id": "info_sso",
                "canonical_query": "What is Nimbus SSO?",
                "category": "product",
                "required_entities": ["sso"],
                "examples": [
                    "What is Nimbus SSO?",
                    "Can you tell me about Nimbus SSO?",
                    "Tell me about Nimbus SSO",
                    "What does Nimbus SSO do?",
                ],
                "response": (
                    "Nimbus SSO provides enterprise single sign-on, identity governance, multi-factor authentication, and automated SCIM provisioning across all your company applications."
                ),
            },
            {
                "id": "info_endpoint",
                "canonical_query": "What is Nimbus Endpoint?",
                "category": "product",
                "required_entities": ["endpoint"],
                "examples": [
                    "What is Nimbus Endpoint?",
                    "Can you tell me about Nimbus Endpoint?",
                    "Tell me about Nimbus Endpoint",
                    "What does Nimbus Endpoint do?",
                ],
                "response": (
                    "Nimbus Endpoint secures and monitors all corporate laptops, desktops, and mobile devices with next-generation antivirus, remote wipe, USB device controls, and compliance reporting."
                ),
            },

            # ══════════════════════════════════════════════════════════════════
            # 4. ALL 24 PRODUCT PRICING (MONTHLY, ANNUAL, TIERS)
            # ══════════════════════════════════════════════════════════════════
            {
                "id": "pricing_crm",
                "canonical_query": "How much does Nimbus CRM cost?",
                "category": "product",
                "required_entities": ["crm"],
                "examples": [
                    "How much does Nimbus CRM cost?",
                    "What is the price of Nimbus CRM?",
                    "How much is Nimbus CRM?",
                    "Nimbus CRM pricing",
                    "What are the CRM plans?",
                    "Tell me about CRM pricing",
                ],
                "response": (
                    "Nimbus CRM starts at fifteen dollars per user per month on the Starter tier, or twelve dollars billed annually. Professional is forty-five dollars per user per month, and a forever free tier is also included."
                ),
            },
            {
                "id": "pricing_leads",
                "canonical_query": "How much does Nimbus Leads cost?",
                "category": "product",
                "required_entities": ["leads"],
                "examples": [
                    "How much does Nimbus Leads cost?",
                    "What is the price of Nimbus Leads?",
                    "How much is Nimbus Leads?",
                    "Nimbus Leads pricing",
                    "What does Leads cost?",
                ],
                "response": (
                    "Nimbus Leads starts at twenty-five dollars per user per month for Starter, or twenty dollars billed annually. Professional is sixty-five dollars per user per month with power dialer and lead scoring."
                ),
            },
            {
                "id": "pricing_quote",
                "canonical_query": "How much does Nimbus Quote cost?",
                "category": "product",
                "required_entities": ["quote"],
                "examples": [
                    "How much does Nimbus Quote cost?",
                    "What is the price of Nimbus Quote?",
                    "How much is Nimbus Quote?",
                    "Nimbus Quote pricing",
                ],
                "response": (
                    "Nimbus Quote starts at twenty dollars per user per month for Starter, or sixteen dollars billed annually. Professional is fifty dollars per user per month with CPQ and electronic signatures."
                ),
            },
            {
                "id": "pricing_campaigns",
                "canonical_query": "How much does Nimbus Campaigns cost?",
                "category": "product",
                "required_entities": ["campaigns"],
                "examples": [
                    "How much does Nimbus Campaigns cost?",
                    "What is the price of Nimbus Campaigns?",
                    "How much is Nimbus Campaigns?",
                    "Nimbus Campaigns pricing",
                ],
                "response": (
                    "Nimbus Campaigns starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is forty-five dollars per user per month with email marketing automation."
                ),
            },
            {
                "id": "pricing_social",
                "canonical_query": "How much does Nimbus Social cost?",
                "category": "product",
                "required_entities": ["social"],
                "examples": [
                    "How much does Nimbus Social cost?",
                    "What is the price of Nimbus Social?",
                    "How much is Nimbus Social?",
                    "Nimbus Social pricing",
                ],
                "response": (
                    "Nimbus Social starts at twenty dollars per user per month for Starter, or sixteen dollars billed annually. Professional is fifty dollars per user per month with cross-platform scheduling and analytics."
                ),
            },
            {
                "id": "pricing_sites",
                "canonical_query": "How much does Nimbus Sites cost?",
                "category": "product",
                "required_entities": ["sites"],
                "examples": [
                    "How much does Nimbus Sites cost?",
                    "What is the price of Nimbus Sites?",
                    "How much is Nimbus Sites?",
                    "Nimbus Sites pricing",
                ],
                "response": (
                    "Nimbus Sites starts at twenty-five dollars per user per month for Starter, or twenty dollars billed annually. Professional is sixty dollars per user per month with custom domains and A/B testing."
                ),
            },
            {
                "id": "pricing_books",
                "canonical_query": "How much does Nimbus Books cost?",
                "category": "product",
                "required_entities": ["books"],
                "examples": [
                    "How much does Nimbus Books cost?",
                    "What is the price of Nimbus Books?",
                    "How much is Nimbus Books?",
                    "Nimbus Books pricing",
                    "What are the Books plans?",
                ],
                "response": (
                    "Nimbus Books starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is forty dollars per month with multi-currency and double-entry accounting."
                ),
            },
            {
                "id": "pricing_invoice",
                "canonical_query": "How much does Nimbus Invoice cost?",
                "category": "product",
                "required_entities": ["invoice"],
                "examples": [
                    "How much does Nimbus Invoice cost?",
                    "What is the price of Nimbus Invoice?",
                    "How much is Nimbus Invoice?",
                    "Nimbus Invoice pricing",
                    "What are the Invoice plans?",
                ],
                "response": (
                    "Nimbus Invoice starts at ten dollars per user per month for Starter, or eight dollars billed annually. Professional is thirty dollars per user per month with automated recurring billing."
                ),
            },
            {
                "id": "pricing_expense",
                "canonical_query": "How much does Nimbus Expense cost?",
                "category": "product",
                "required_entities": ["expense"],
                "examples": [
                    "How much does Nimbus Expense cost?",
                    "What is the price of Nimbus Expense?",
                    "How much is Nimbus Expense?",
                    "Nimbus Expense pricing",
                ],
                "response": (
                    "Nimbus Expense starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is thirty-five dollars per user per month with smart receipt scanning and corporate cards."
                ),
            },
            {
                "id": "pricing_people",
                "canonical_query": "How much does Nimbus People cost?",
                "category": "product",
                "required_entities": ["people"],
                "examples": [
                    "How much does Nimbus People cost?",
                    "What is the price of Nimbus People?",
                    "How much is Nimbus People?",
                    "Nimbus People pricing",
                    "Tell me about Nimbus People HR tool",
                ],
                "response": (
                    "Nimbus People starts at ten dollars per user per month for Starter, or eight dollars billed annually. Professional is thirty dollars per user per month with performance reviews and time tracking."
                ),
            },
            {
                "id": "pricing_recruit",
                "canonical_query": "How much does Nimbus Recruit cost?",
                "category": "product",
                "required_entities": ["recruit"],
                "examples": [
                    "How much does Nimbus Recruit cost?",
                    "What is the price of Nimbus Recruit?",
                    "How much is Nimbus Recruit?",
                    "Nimbus Recruit pricing",
                ],
                "response": (
                    "Nimbus Recruit starts at twenty-five dollars per user per month for Starter, or twenty dollars billed annually. Professional is sixty dollars per user per month with applicant tracking and job board syndication."
                ),
            },
            {
                "id": "pricing_payroll",
                "canonical_query": "How much does Nimbus Payroll cost?",
                "category": "product",
                "required_entities": ["payroll"],
                "examples": [
                    "How much does Nimbus Payroll cost?",
                    "What is the price of Nimbus Payroll?",
                    "How much is Nimbus Payroll?",
                    "Nimbus Payroll pricing",
                    "Tell me about Payroll cost",
                    "How much is payroll?",
                ],
                "response": (
                    "Nimbus Payroll starts at twenty dollars per user per month for Starter, or sixteen dollars billed annually. Professional is fifty dollars per month with direct deposit and automated tax filing."
                ),
            },
            {
                "id": "pricing_desk",
                "canonical_query": "How much does Nimbus Desk cost?",
                "category": "product",
                "required_entities": ["desk"],
                "examples": [
                    "How much does Nimbus Desk cost?",
                    "What is the price of Nimbus Desk?",
                    "How much is Nimbus Desk?",
                    "Nimbus Desk pricing",
                    "Tell me about Nimbus Desk plans",
                ],
                "response": (
                    "Nimbus Desk starts at twenty dollars per user per month for Starter, or sixteen dollars billed annually. Professional is fifty dollars per user per month with multi-channel ticketing and SLA management."
                ),
            },
            {
                "id": "pricing_chat",
                "canonical_query": "How much does Nimbus Chat cost?",
                "category": "product",
                "required_entities": ["chat"],
                "examples": [
                    "How much does Nimbus Chat cost?",
                    "What is the price of Nimbus Chat?",
                    "How much is Nimbus Chat?",
                    "Nimbus Chat pricing",
                ],
                "response": (
                    "Nimbus Chat starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is forty-five dollars per month with proactive chat triggers and AI bots."
                ),
            },
            {
                "id": "pricing_knowledge",
                "canonical_query": "How much does Nimbus Knowledge cost?",
                "category": "product",
                "required_entities": ["knowledge"],
                "examples": [
                    "How much does Nimbus Knowledge cost?",
                    "What is the price of Nimbus Knowledge?",
                    "How much is Nimbus Knowledge?",
                    "Nimbus Knowledge pricing",
                ],
                "response": (
                    "Nimbus Knowledge starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is thirty-five dollars per month with custom branding and multilingual articles."
                ),
            },
            {
                "id": "pricing_projects",
                "canonical_query": "How much does Nimbus Projects cost?",
                "category": "product",
                "required_entities": ["projects"],
                "examples": [
                    "How much does Nimbus Projects cost?",
                    "What is the price of Nimbus Projects?",
                    "How much is Nimbus Projects?",
                    "Nimbus Projects pricing",
                    "What are the Projects plans?",
                ],
                "response": (
                    "Nimbus Projects starts at ten dollars per user per month for Starter, or eight dollars billed annually. Professional is thirty dollars per user per month with Gantt charts and workload management."
                ),
            },
            {
                "id": "pricing_docs",
                "canonical_query": "How much does Nimbus Docs cost?",
                "category": "product",
                "required_entities": ["docs"],
                "examples": [
                    "How much does Nimbus Docs cost?",
                    "What is the price of Nimbus Docs?",
                    "How much is Nimbus Docs?",
                    "Nimbus Docs pricing",
                    "What are the Docs plans?",
                    "Tell me about Nimbus Docs cost",
                ],
                "response": (
                    "Nimbus Docs starts at eight dollars per user per month for Starter, or six dollars and forty cents billed annually. Professional is fifteen dollars per user per month, and a forever free tier is also included."
                ),
            },
            {
                "id": "pricing_boards",
                "canonical_query": "How much does Nimbus Boards cost?",
                "category": "product",
                "required_entities": ["boards"],
                "examples": [
                    "How much does Nimbus Boards cost?",
                    "What is the price of Nimbus Boards?",
                    "How much is Nimbus Boards?",
                    "Nimbus Boards pricing",
                ],
                "response": (
                    "Nimbus Boards starts at twelve dollars per user per month for Starter, or nine dollars and sixty cents billed annually. Professional is twenty-five dollars per month with unlimited boards and custom workflows."
                ),
            },
            {
                "id": "pricing_analytics",
                "canonical_query": "How much does Nimbus Analytics cost?",
                "category": "product",
                "required_entities": ["analytics"],
                "examples": [
                    "How much does Nimbus Analytics cost?",
                    "What is the price of Nimbus Analytics?",
                    "How much is Nimbus Analytics?",
                    "Nimbus Analytics pricing",
                ],
                "response": (
                    "Nimbus Analytics starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is forty-five dollars per user per month with advanced data modeling and visual dashboards."
                ),
            },
            {
                "id": "pricing_dashboards",
                "canonical_query": "How much does Nimbus Dashboards cost?",
                "category": "product",
                "required_entities": ["dashboards"],
                "examples": [
                    "How much does Nimbus Dashboards cost?",
                    "What is the price of Nimbus Dashboards?",
                    "How much is Nimbus Dashboards?",
                    "Nimbus Dashboards pricing",
                ],
                "response": (
                    "Nimbus Dashboards starts at ten dollars per user per month for Starter, or eight dollars billed annually. Professional is thirty dollars per user per month with live TV displays and scheduled email snapshots."
                ),
            },
            {
                "id": "pricing_datapipe",
                "canonical_query": "How much does Nimbus DataPipe cost?",
                "category": "product",
                "required_entities": ["datapipe"],
                "examples": [
                    "How much does Nimbus DataPipe cost?",
                    "What is the price of Nimbus DataPipe?",
                    "How much is Nimbus DataPipe?",
                    "Nimbus DataPipe pricing",
                ],
                "response": (
                    "Nimbus DataPipe starts at twenty-five dollars per user per month for Starter, or twenty dollars billed annually. Professional is sixty-five dollars per user per month with real-time sync and custom connectors."
                ),
            },
            {
                "id": "pricing_vault",
                "canonical_query": "How much does Nimbus Vault cost?",
                "category": "product",
                "required_entities": ["vault"],
                "examples": [
                    "How much does Nimbus Vault cost?",
                    "What is the price of Nimbus Vault?",
                    "How much is Nimbus Vault?",
                    "Nimbus Vault pricing",
                ],
                "response": (
                    "Nimbus Vault starts at ten dollars per user per month for Starter, or eight dollars billed annually. Professional is twenty-five dollars per month with secret rotation and automated audit logging."
                ),
            },
            {
                "id": "pricing_sso",
                "canonical_query": "How much does Nimbus SSO cost?",
                "category": "product",
                "required_entities": ["sso"],
                "examples": [
                    "How much does Nimbus SSO cost?",
                    "What is the price of Nimbus SSO?",
                    "How much is Nimbus SSO?",
                    "Nimbus SSO pricing",
                ],
                "response": (
                    "Nimbus SSO starts at fifteen dollars per user per month for Starter, or twelve dollars billed annually. Professional is thirty-five dollars per month with SAML 2.0 and automated user provisioning."
                ),
            },
            {
                "id": "pricing_endpoint",
                "canonical_query": "How much does Nimbus Endpoint cost?",
                "category": "product",
                "required_entities": ["endpoint"],
                "examples": [
                    "How much does Nimbus Endpoint cost?",
                    "What is the price of Nimbus Endpoint?",
                    "How much is Nimbus Endpoint?",
                    "Nimbus Endpoint pricing",
                ],
                "response": (
                    "Nimbus Endpoint starts at twenty dollars per user per month for Starter, or sixteen dollars billed annually. Professional is forty-five dollars per user per month with remote device management and patch deployment."
                ),
            },
        ]

        self.entries.clear()
        self._cached_vectors.clear()
        self._example_vectors.clear()

        for seed in seed_data:
            canonical_vec = self.engine.embed(seed["canonical_query"])
            ex_vecs = [self.engine.embed(ex) for ex in seed["examples"]]
            entry = CacheEntry(
                id=seed["id"],
                canonical_query=seed["canonical_query"],
                response=seed["response"],
                category=seed["category"],
                examples=seed["examples"],
                embedding=canonical_vec.tolist(),
                example_embeddings=[ev.tolist() for ev in ex_vecs],
                required_entities=seed.get("required_entities", []),
            )
            self.entries[entry.id] = entry
            self._cached_vectors[entry.id] = canonical_vec
            self._example_vectors[entry.id] = ex_vecs

    def persist(self):
        """Save current cache entries to JSON file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": time.time(),
                "total_entries": len(self.entries),
                "entries": [asdict(e) for e in self.entries.values()],
            }
            self.cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist semantic cache: {e}")

    def get(
        self, query: str, threshold: Optional[float] = None
    ) -> Optional[Tuple[str, float, str]]:
        """Search cache for a semantically similar query.
        Returns (response_text, similarity_score, canonical_query) if found, else None.
        """
        if not query or not query.strip():
            return None

        effective_threshold = threshold if threshold is not None else self.threshold
        query_vec = self.engine.embed(query)
        query_entities = self._extract_entities(query)

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None

        for entry_id, candidate_vec in self._cached_vectors.items():
            entry = self.entries[entry_id]

            # Entity compatibility check:
            # If candidate requires specific product or policy entities, query must match them!
            if entry.required_entities:
                entry_ents = set(entry.required_entities)
                # Check product entity requirement
                entry_products = entry_ents & PRODUCT_ENTITIES
                if entry_products:
                    query_products = query_entities & PRODUCT_ENTITIES
                    if not (query_products & entry_products):
                        continue

                # Check policy entity requirement
                entry_policies = entry_ents & POLICY_ENTITIES
                if entry_policies:
                    query_policies = query_entities & POLICY_ENTITIES
                    if not (query_policies & entry_policies):
                        continue

            # Compare against canonical vector
            sim = self.engine.cosine_similarity(query_vec, candidate_vec)

            # Also compare against all exemplars for max similarity
            ex_vecs = self._example_vectors.get(entry_id, [])
            for ev in ex_vecs:
                ex_sim = self.engine.cosine_similarity(query_vec, ev)
                if ex_sim > sim:
                    sim = ex_sim

            # Exact query match boost
            if query.lower().strip() == entry.canonical_query.lower().strip():
                sim = 1.0

            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= effective_threshold:
            best_entry.hit_count += 1
            best_entry.last_hit_at = time.time()
            return CacheHitResult(
                best_entry.response,
                round(best_score, 4),
                best_entry.canonical_query,
                category=best_entry.category,
            )

        return None

    def set(
        self,
        query: str,
        response: str,
        category: str = "dynamic",
        required_entities: Optional[List[str]] = None,
    ):
        """Store a new query and response into cache, compute vector embedding, and persist."""
        if not query or not response or len(response.strip()) < 10:
            return

        # Check if identical or very close query already exists
        existing = self.get(query, threshold=0.92)
        if existing:
            return

        entry_id = f"dyn_{hashlib.md5(query.lower().encode('utf-8')).hexdigest()[:10]}"
        vec = self.engine.embed(query)
        entities = required_entities or list(self._extract_entities(query))

        entry = CacheEntry(
            id=entry_id,
            canonical_query=query.strip(),
            response=response.strip(),
            category=category,
            examples=[query.strip()],
            embedding=vec.tolist(),
            example_embeddings=[vec.tolist()],
            required_entities=entities,
            hit_count=1,
            created_at=time.time(),
            last_hit_at=time.time(),
        )

        self.entries[entry_id] = entry
        self._cached_vectors[entry_id] = vec
        self._example_vectors[entry_id] = [vec]
        logger.info(f"💾 [SemanticCache] Stored new dynamic entry for: '{query}' ({len(self.entries)} total)")
        self.persist()

    def stats(self) -> Dict[str, Any]:
        """Return cache hit statistics."""
        total_hits = sum(e.hit_count for e in self.entries.values())
        return {
            "total_entries": len(self.entries),
            "total_hits": total_hits,
            "categories": {
                cat: sum(1 for e in self.entries.values() if e.category == cat)
                for cat in {"policy", "product", "company", "comparison", "dynamic"}
            },
        }


# Singleton cache instance
_GLOBAL_CACHE: Optional[SemanticResponseCache] = None


def get_semantic_cache() -> SemanticResponseCache:
    """Get or create singleton SemanticResponseCache instance."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = SemanticResponseCache()
    return _GLOBAL_CACHE

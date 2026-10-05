"""tests/test_semantic_cache.py

Comprehensive test suite for Semantic Response & Vector Embedding Cache.
Verifies embedding speed, cosine similarity matching, entity constraints,
cache hit/miss handling, dynamic learning, and Pipecat frame bypass behavior.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_cache import CacheHitFrame, SemanticResponseCache, get_semantic_cache


def test_semantic_cache_retrieval():
    print("\n--- 1. Testing Semantic Cache Retrieval & Latency ---")
    cache = get_semantic_cache()
    stats = cache.stats()
    print(f"Total entries loaded: {stats['total_entries']}")
    assert stats["total_entries"] >= 16, "Expected at least 16 pre-warmed entries"

    test_cases = [
        # (Query variations, Expected canonical id)
        ("What is your refund policy?", "policy_refund"),
        ("Can I get my money back?", "policy_refund"),
        ("Do you have a refund policy?", "policy_refund"),
        ("Do you offer a free trial?", "policy_free_trial"),
        ("Can I try Nimbus for free?", "policy_free_trial"),
        ("Where is your company located?", "company_overview"),
        ("Where are your headquarters?", "company_overview"),
        ("What are your cheapest products?", "comp_cheapest"),
        ("Which tools are most affordable?", "comp_cheapest"),
        ("How much does Nimbus CRM cost?", "prod_nimbus_crm"),
        ("What is the price of Nimbus CRM?", "prod_nimbus_crm"),
        ("How much is Nimbus CRM?", "prod_nimbus_crm"),
        ("How much does Nimbus Payroll cost?", "prod_nimbus_payroll"),
        ("How much is payroll?", "prod_nimbus_payroll"),
        ("Tell me about Nimbus Desk pricing", "prod_nimbus_desk"),
        ("How much is Nimbus Leads?", "prod_nimbus_leads"),
    ]

    for query, expected_id in test_cases:
        t0 = time.perf_counter()
        result = cache.get(query)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"\nQuery: '{query}'")
        print(f"Latency: {latency_ms:.3f} ms")
        assert result is not None, f"Expected cache HIT for '{query}', got None"
        response_text, sim, canonical = result
        print(f"Hit canonical: '{canonical}' (similarity: {sim:.3f})")
        print(f"Spoken response preview: {response_text[:80]}...")
        assert sim >= 0.70, f"Expected similarity >= 0.70, got {sim}"
        assert latency_ms < 5.0, f"Expected sub-5ms lookup, got {latency_ms:.2f}ms"
        print("  -> [PASS]")

    print("\n--- 2. Testing Cache Miss on Irrelevant Queries ---")
    irrelevant_queries = [
        "What is the weather today?",
        "Can I order a pepperoni pizza?",
        "Wait one second please",
        "Who was the 16th president of the United States?",
    ]
    for q in irrelevant_queries:
        result = cache.get(q)
        print(f"Query: '{q}' -> Result: {result}")
        assert result is None, f"Expected cache MISS for '{q}', got {result}"
        print("  -> [PASS]")


def test_cross_product_entity_guard():
    print("\n--- 3. Testing Cross-Product Entity Guard ---")
    cache = get_semantic_cache()
    # A query specifically asking about Payroll must NEVER hit CRM
    crm_result = cache.get("What is the price of Nimbus CRM?")
    payroll_result = cache.get("What is the price of Nimbus Payroll?")

    assert crm_result is not None and "CRM" in crm_result[0]
    assert payroll_result is not None and "Payroll" in payroll_result[0]
    assert crm_result[0] != payroll_result[0], "CRM and Payroll responses must not collide"
    print("Cross-product entity isolation: [PASS]")


def test_dynamic_learning():
    print("\n--- 4. Testing Dynamic Learning (Store & Reuse) ---")
    cache = get_semantic_cache()
    novel_query = "Do you integrate with Salesforce?"
    novel_response = "Yes, Nimbus CRM includes 1-click import and bi-directional sync with Salesforce."

    cache.set(novel_query, novel_response, category="dynamic")

    # Immediate hit
    hit = cache.get("Do you integrate with Salesforce?")
    assert hit is not None
    assert hit[0] == novel_response
    print("Dynamic exact hit: [PASS]")

    # Paraphrased hit
    para_hit = cache.get("Can I connect Salesforce to Nimbus?")
    if para_hit:
        print(f"Dynamic paraphrase hit: {para_hit[1]:.3f} [PASS]")
    else:
        print("Dynamic paraphrase did not cross threshold (expected with single exemplar) [OK]")


def test_pipeline_processor_frame_bypass():
    print("\n--- 5. Testing Pipeline Frame Bypass in SemanticRAGProcessor ---")
    from bot import SemanticRAGProcessor
    from pipecat.frames.frames import (
        LLMContextFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.frame_processor import FrameDirection
    from rag import get_rag

    async def _test():
        rag = get_rag()
        cache = get_semantic_cache()
        processor = SemanticRAGProcessor(rag, cache)

        pushed_frames = []

        async def mock_push(frame, direction):
            pushed_frames.append(frame)

        processor.push_frame = mock_push

        # ── Test A: Cache HIT (Should bypass LLMContextFrame completely!) ──
        context = LLMContext([{"role": "user", "content": "What is your refund policy?"}])
        context_frame = LLMContextFrame(context=context)

        pushed_frames.clear()
        await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)

        frame_types = [type(f).__name__ for f in pushed_frames]
        print(f"Pushed frame sequence on Cache Hit: {frame_types}")

        # Verify LLMContextFrame was DROPPED (not pushed to LLM!)
        assert "LLMContextFrame" not in frame_types, "LLMContextFrame must NOT be pushed on cache hit!"

        # Verify speech frames were pushed downstream to TTS
        assert "CacheHitFrame" in frame_types, "CacheHitFrame must be emitted for telemetry"
        assert "LLMFullResponseStartFrame" in frame_types, "Start frame must be emitted"
        assert "TextFrame" in frame_types, "TextFrame with response must be emitted"
        assert "LLMFullResponseEndFrame" in frame_types, "End frame must be emitted"

        text_frames = [f for f in pushed_frames if isinstance(f, TextFrame)]
        assert "30-day money-back guarantee" in text_frames[0].text
        print("Cache HIT frame bypass test: [PASS]")

        # ── Test B: Cache MISS (Must push LLMContextFrame to LLM!) ──
        miss_context = LLMContext([{"role": "user", "content": "What is your data retention policy for enterprise?"}])
        miss_frame = LLMContextFrame(context=miss_context)

        pushed_frames.clear()
        await processor.process_frame(miss_frame, FrameDirection.DOWNSTREAM)

        miss_frame_types = [type(f).__name__ for f in pushed_frames]
        print(f"Pushed frame sequence on Cache Miss: {miss_frame_types}")

        # Verify LLMContextFrame WAS pushed downstream to LLM
        assert "LLMContextFrame" in miss_frame_types, "LLMContextFrame must be pushed on cache miss!"
        print("Cache MISS fallback test: [PASS]")

    asyncio.run(_test())


if __name__ == "__main__":
    test_semantic_cache_retrieval()
    test_cross_product_entity_guard()
    test_dynamic_learning()
    test_pipeline_processor_frame_bypass()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

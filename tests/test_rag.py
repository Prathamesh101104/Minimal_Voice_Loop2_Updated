import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import get_rag

def test_rag_retrieval():
    rag = get_rag()
    test_cases = [
        ("What is your refund policy?", "Refund Policy"),
        ("Do you offer a free trial?", "Free Trial Policy"),
        ("How much does Nimbus CRM cost?", "Nimbus CRM - Pricing, Features & Details"),
        ("Tell me about Nimbus Payroll", "Nimbus Payroll - Pricing, Features & Details"),
        ("What are your cheapest products?", "Product Pricing Comparison & Rankings (Cheapest to Most Expensive)"),
        ("Which product is most expensive?", "Product Pricing Comparison & Rankings (Cheapest to Most Expensive)"),
        ("Where is your company located?", "Nimbus Company Overview & Headquarters"),
    ]

    print("Running SimpleRAG Verification...")
    for query, expected_title in test_cases:
        t0 = time.perf_counter()
        chunks = rag.retrieve(query, top_k=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        titles = [c.title for c in chunks]
        print(f"\nQuery: '{query}'")
        print(f"Latency: {elapsed_ms:.3f} ms")
        print(f"Retrieved: {titles}")

        assert any(expected_title in t for t in titles), f"Failed for '{query}': expected '{expected_title}' in {titles}"
        print("  -> [PASS]")

    # Check formatted prompt context
    sample_context = rag.format_context_for_prompt("How much is Nimbus CRM?")
    assert "[Retrieved Website Knowledge]:" in sample_context
    assert "Nimbus CRM" in sample_context
    assert "$15" in sample_context
    print("\nFormat context test: [PASS]")


def test_processor_context_injection():
    import asyncio
    from bot import SimpleRAGProcessor
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.frame_processor import FrameDirection

    async def _test():
        rag = get_rag()
        processor = SimpleRAGProcessor(rag)

        # Mock downstream push
        pushed_frames = []
        async def mock_push(frame, direction):
            pushed_frames.append(frame)
        processor.push_frame = mock_push

        # Create context with a query that misses cache to test RAG fallback
        messages = [
            {"role": "system", "content": "You are Nimbus assistant."},
            {"role": "user", "content": "Can I connect custom legacy IoT hardware sensors to Nimbus?"},
        ]
        ctx = LLMContext(messages)
        frame = LLMContextFrame(context=ctx)

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        assert len(pushed_frames) == 1
        augmented_user_msg = ctx.messages[-1]["content"]
        assert "[Retrieved Website Knowledge]:" in augmented_user_msg
        print("\nSimpleRAGProcessor RAG fallback frame injection test: [PASS]")

    asyncio.run(_test())


if __name__ == "__main__":
    test_rag_retrieval()
    test_processor_context_injection()
    print("\nALL TESTS PASSED SUCCESSFULLY!")


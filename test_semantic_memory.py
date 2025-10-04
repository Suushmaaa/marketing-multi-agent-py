#!/usr/bin/env python3
"""
Test script for semantic memory operations
"""
import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.semantic import SemanticMemorySystem
from config.settings import settings


async def test_semantic_memory():
    """Test semantic memory operations"""
    print("Testing Semantic Memory Operations")
    print("=" * 50)

    # Initialize semantic memory
    semantic = SemanticMemorySystem()

    try:
        # Test initialization
        print("1. Testing initialization...")
        success = await semantic.initialize()
        if success:
            print("✓ Semantic memory initialized successfully")
        else:
            print("✗ Semantic memory initialization failed")
            return

        # Test storing triples
        print("\n2. Testing triple storage...")
        test_triples = [
            {
                "subject": "Marketing",
                "predicate": "includes",
                "object": "Email Campaigns",
                "weight": 0.9,
                "source": "test"
            },
            {
                "subject": "Email Campaigns",
                "predicate": "uses",
                "object": "Templates",
                "weight": 0.8,
                "source": "test"
            },
            {
                "subject": "Templates",
                "predicate": "contains",
                "object": "Personalization",
                "weight": 0.7,
                "source": "test"
            }
        ]

        for i, triple in enumerate(test_triples):
            triple_id = f"test_triple_{i}"
            success = await semantic.store(triple_id, triple)
            if success:
                print(f"✓ Stored triple: {triple['subject']} -> {triple['predicate']} -> {triple['object']}")
            else:
                print(f"✗ Failed to store triple {i}")

        # Test retrieving triples
        print("\n3. Testing triple retrieval...")
        for i in range(len(test_triples)):
            triple_id = f"test_triple_{i}"
            retrieved = await semantic.retrieve(triple_id)
            if retrieved:
                print(f"✓ Retrieved triple {i}: {retrieved}")
            else:
                print(f"✗ Failed to retrieve triple {i}")

        # Test searching
        print("\n4. Testing search operations...")
        results = await semantic.search({"subject": "Marketing"})
        print(f"✓ Found {len(results)} triples with subject 'Marketing'")

        results = await semantic.search({"min_weight": 0.8})
        print(f"✓ Found {len(results)} triples with weight >= 0.8")

        # Test neighbor queries
        print("\n5. Testing neighbor queries...")
        neighbors = await semantic.query_neighbors("Marketing", depth=2)
        print(f"✓ Found {len(neighbors)} neighbors for 'Marketing'")

        # Test path finding
        print("\n6. Testing path finding...")
        path = await semantic.find_path("Marketing", "Personalization")
        if path:
            print(f"✓ Found path: {' -> '.join([p['subject'] + '-' + p['predicate'] + '->' + p['object'] for p in path])}")
        else:
            print("✗ No path found between Marketing and Personalization")

        # Test statistics
        print("\n7. Testing statistics...")
        stats = await semantic.get_stats()
        print(f"✓ Semantic memory stats: {stats}")

        # Test cleanup
        print("\n8. Testing cleanup...")
        success = await semantic.cleanup()
        if success:
            print("✓ Cleanup completed successfully")
        else:
            print("✗ Cleanup failed")

        print("\n" + "=" * 50)
        print("Semantic Memory Testing Completed Successfully!")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close connection
        await semantic.close()


if __name__ == "__main__":
    asyncio.run(test_semantic_memory())

from intelligence.rag.retriever import RAGRetriever


def main():

    retriever = RAGRetriever(
        knowledge_directory="knowledge_base"
    )

    print("\nIndexing knowledge base...")

    retriever.index_knowledge()

    print("Indexing completed.")

    query = """
    A tabular dataset contains a column named age.
    Values represent the age of people in years.
    One observed value is 999.

    Find knowledge relevant to understanding whether
    this value may represent a data quality problem.
    """

    results = retriever.retrieve(
        query=query,
        top_k=10,
        categories=[
            "semantic_concept",
            "data_quality",
            "domain",
        ],
    )

    print("\nTOP RETRIEVED KNOWLEDGE")
    print("=" * 60)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Category: {result['category']}")
        print(f"Topic: {result['topic']}")
        print(f"Section: {result['section']}")
        print(f"Source: {result['source']}")

        print("\nContent:")
        print(result["content"][:500])

        print("-" * 60)


if __name__ == "__main__":
    main()
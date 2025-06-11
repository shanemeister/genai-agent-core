import argparse
from chat.vectorstore_memory import retrieve_similar_context

def main():
    parser = argparse.ArgumentParser(description="Search your vectorstore for similar documents.")
    parser.add_argument("query", type=str, help="The query string to search for")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top similar chunks to return")
    args = parser.parse_args()

    print(f"\n🔍 Searching for: {args.query}\n")
    results = retrieve_similar_context(args.query, k=args.top_k)

    if not results:
        print("⚠️ No similar content found.")
        return

    for i, chunk in enumerate(results, 1):
        print(f"[{i}]\n{'-'*60}\n{chunk[:1000]}\n{'-'*60}\n")

if __name__ == "__main__":
    main()
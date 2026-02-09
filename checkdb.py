# check_db.py
import chromadb
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_collection("openai_research_vault")

count = collection.count()
print(f"📉 You currently have {count} papers stored in your local memory.")

# Show the first few items
if count > 0:
    print(collection.peek())
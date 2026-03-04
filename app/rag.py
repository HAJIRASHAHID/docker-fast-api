# RAG functions (retrieve_top_chunks, ask_rag)
# rag.py
import numpy as np

def retrieve_top_chunks(question: str, index, embedding_model, top_k: int = 3):
    q_embedding = embedding_model.encode([question])[0]
    q_embedding = np.array(q_embedding).astype("float32")
    result = index.query(vector=q_embedding.tolist(), top_k=top_k, include_metadata=True)
    matches = result.get("matches") if isinstance(result, dict) else getattr(result, "matches", None)
    if not matches:
        return []
    return [match.get("metadata", {}).get("text", "") for match in matches]

def ask_rag(question: str, client, index, embedding_model):
    contexts = retrieve_top_chunks(question, index, embedding_model)
    combined_context = "\n\n".join(contexts)

    prompt = f"""
You are a Machine Learning assistant.
Answer ONLY from the context below.
If answer is not in context, say "Not found in document".

Context:
{combined_context}

Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        # handle a couple of response shapes safely
        choice = None
        if isinstance(response, dict):
            choice = response.get("choices", [{}])[0]
            # some clients use nested message
            return choice.get("message", {}).get("content") or choice.get("text")
        else:
            return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
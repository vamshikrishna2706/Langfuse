import os
import uuid

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langfuse import get_client, observe, propagate_attributes

load_dotenv()

client = InferenceClient(provider="featherless-ai", token=os.environ["HF_TOKEN"])
langfuse = get_client()


@observe()
def ask_llm(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    # A session groups multiple separate traces (e.g. conversation turns) together
    # in the Langfuse UI. Each ask_llm() call below still produces its own trace,
    # but tagging both with the same session_id lets you view them as one thread.
    session_id = str(uuid.uuid4())
    print(f"session_id: {session_id}")

    with propagate_attributes(session_id=session_id):
        answer = ask_llm("In one sentence, what is LLM observability?")
        print(answer)

    with propagate_attributes(session_id=session_id):
        follow_up = ask_llm("Now explain that to a five-year-old.")
        print(follow_up)

    langfuse.flush()

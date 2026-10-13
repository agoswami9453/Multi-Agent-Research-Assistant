import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "").strip()

if not api_key:
    print("GROQ_ACTIVE= False")
    print("ERROR= GROQ_API_KEY is missing")
    raise SystemExit(1)

try:
    response = Groq(api_key=api_key).chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_tokens=5,
    )
    print("GROQ_ACTIVE= True")
    print("GROQ_RESPONSE=", response.choices[0].message.content.strip())
except Exception as error:
    print("GROQ_ACTIVE= False")
    print("ERROR=", type(error).__name__, str(error))
    raise SystemExit(1)
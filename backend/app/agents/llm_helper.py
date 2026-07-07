import os
from typing import Dict, Any, Optional
from google import genai
import openai

def get_llm_client() -> Optional[Dict[str, Any]]:
    """
    Initialize and return LLM client based on available environment variables.
    Returns a dict with 'provider' and 'client' or None.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            return {"provider": "gemini", "client": client}
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            openai.api_key = openai_key
            return {"provider": "openai", "client": openai}
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            
    return None

def generate_text(prompt: str, system_instruction: str = "") -> Optional[str]:
    """
    Generate text using the first available LLM provider.
    """
    llm = get_llm_client()
    if not llm:
        return None
        
    try:
        if llm["provider"] == "gemini":
            client = llm["client"]
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.3
                }
            )
            return response.text.strip()
            
        elif llm["provider"] == "openai":
            client = llm["client"]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return None

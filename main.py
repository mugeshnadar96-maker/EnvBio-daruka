"""
main.py
-------
Entrypoints for the environmental knowledge assistant:

  1. CLI (interactive REPL):
         python3 main.py

  2. FastAPI server:
         python3 main.py --serve
     then:
         POST /chat            {"text": "..."}                 (free text)
         POST /chat/json       {<structured JSON per spec>}    (structured)
         GET  /memory           -> current session memory window
"""

import argparse
import sys
import json

from conversation_agent import EnvAgent


def run_cli():
    agent = EnvAgent()
    print("Environmental Biodiversity & Soil Health Assistant")
    print("Type a description of your land, or paste JSON. Type 'exit' to quit.")
    print("-" * 70)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        is_json = user_input.startswith("{")
        payload = json.loads(user_input) if is_json else user_input
        response = agent.handle_turn(payload, is_json=is_json)
        print("\nAssistant:\n" + response)


def run_server():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from pathlib import Path
    import uvicorn

    app = FastAPI(title="Environmental Knowledge Assistant")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    agent = EnvAgent()

    ui_path = Path(__file__).parent / "webapp" / "index.html"

    class TextIn(BaseModel):
        text: str

    @app.get("/", response_class=HTMLResponse)
    def ui():
        if ui_path.exists():
            return ui_path.read_text()
        return HTMLResponse("<p>webapp/index.html not found next to main.py</p>", status_code=404)

    @app.post("/chat")
    def chat(body: TextIn):
        return {"response": agent.handle_turn(body.text, is_json=False)}

    @app.post("/chat/json")
    def chat_json(body: dict):
        return {"response": agent.handle_turn(body, is_json=True)}

    @app.get("/memory")
    def memory():
        return {"turns": agent.memory_summary()}

    print("\nUI:  http://localhost:8000")
    print("API: http://localhost:8000/chat  (POST {\"text\": \"...\"})\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run as FastAPI server instead of CLI")
    args = parser.parse_args()

    if args.serve:
        run_server()
    else:
        run_cli()

import os, json, glob
from langfuse import Langfuse

def sync():
    if os.getenv("TRACE_TO_LANGFUSE") != "true": return
    
    lf = Langfuse()
    files = glob.glob(os.path.expanduser("~/.claude/sessions/*.json"))
    if not files: return
    
    with open(max(files, key=os.path.getmtime), 'r') as f:
        data = json.load(f)
    
    model = os.getenv("MODEL_NAME") or "claude-code"
    
    trace = lf.trace(
        name=f"ClaudeCode_Fix_{os.getenv('RUN_ID', 'unknown')}",
        session_id=data.get("id"),
        metadata={"run_id": os.getenv("RUN_ID")}
    )
    
    for msg in data.get("messages", []):
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "assistant":
            trace.generation(name="reply", output=content, model=model)
        else:
            trace.event(name=role, input=content)
    lf.flush()

if __name__ == "__main__": sync()

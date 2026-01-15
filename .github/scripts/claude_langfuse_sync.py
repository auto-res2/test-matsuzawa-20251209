import os, json
from langfuse import Langfuse

def sync():
    # Actionが保存したログファイルのパス
    log_path = "/home/runner/work/_temp/claude-execution-output.json"
    if not os.path.exists(log_path):
        print("Log file not found.")
        return

    with open(log_path, 'r') as f:
        events = json.load(f)

    lf = Langfuse()
    
    # ログから結果のサマリーと初期情報を抽出
    result = next((x for x in events if x.get("type") == "result"), {})
    init = next((x for x in events if x.get("type") == "system" and x.get("subtype") == "init"), {})
    model = init.get("model", "claude-code")

    trace = lf.trace(
        name=f"ClaudeCode_Fix_{os.getenv('RUN_ID', 'unknown')}",
        metadata={
            "total_cost": result.get("total_cost_usd"),
            "num_turns": result.get("num_turns"),
            "model": model
        }
    )

    # 会話をすべてトレースに送る
    for e in events:
        if e.get("type") == "message":
            role = e.get("role")
            content = str(e.get("content", ""))
            if role == "assistant":
                trace.generation(name="reply", output=content, model=model)
            else:
                trace.event(name=role, input=content)
    
    lf.flush()
    print("✅ Successfully synced to Langfuse")

if __name__ == "__main__":
    sync()

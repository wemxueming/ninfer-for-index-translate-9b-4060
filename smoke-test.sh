#!/usr/bin/env bash
# smoke-test.sh -- verify the ninfer-serve server is answering on 127.0.0.1:8091
set -uo pipefail
BASE="http://127.0.0.1:8091"

echo "== /v1/models =="
curl -sS --max-time 15 "$BASE/v1/models" || { echo "FAIL: /v1/models unreachable"; exit 1; }
echo

echo "== short chat completion =="
BODY='{"model":"ninfer-local","messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}],"max_tokens":64,"stream":false}'
RESP=$(curl -sS --max-time 180 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY") || { echo "FAIL: completion request failed"; exit 1; }
echo "$RESP" | python3 -c '
import json,sys
r = json.load(sys.stdin)
c = r["choices"][0]
print("finish_reason :", c.get("finish_reason"))
print("content       :", repr(c["message"]["content"]))
print("usage         :", r.get("usage"))
'
echo
echo "== done =="

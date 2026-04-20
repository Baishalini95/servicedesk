from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import re
import urllib3
from dotenv import load_dotenv
from groq import Groq

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
import pathlib
pathlib.Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

SN_URL  = os.getenv("SERVICENOW_URL")
SN_USER = os.getenv("SERVICENOW_USERNAME")
SN_PASS = os.getenv("SERVICENOW_PASSWORD")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    agent: str = "incident"   # "incident" or "kb"


# ── ServiceNow helpers ────────────────────────────────────────────────────────

def sn_get(path, params=None):
    r = requests.get(f"{SN_URL}{path}", auth=(SN_USER, SN_PASS),
                     params=params, verify=False, timeout=10)
    r.raise_for_status()
    return r.json().get("result", [])


def sn_patch(path, payload):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    r = requests.patch(f"{SN_URL}{path}", auth=(SN_USER, SN_PASS),
                       json=payload, headers=headers, verify=False, timeout=10)
    r.raise_for_status()
    return r.json().get("result", {})


def sn_post(path, payload):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    r = requests.post(f"{SN_URL}{path}", auth=(SN_USER, SN_PASS),
                      json=payload, headers=headers, verify=False, timeout=10)
    r.raise_for_status()
    return r.json().get("result", {})


# ── Tools ─────────────────────────────────────────────────────────────────────

def tool_fetch_incident(number):
    results = sn_get("/api/now/table/incident", {
        "sysparm_query": f"number={number.upper()}",
        "sysparm_display_value": "true",
        "sysparm_limit": 1
    })
    return results[0] if results else None


def tool_search_kb(description):
    # Extract meaningful keywords (4+ chars, skip stopwords)
    stopwords = {"with","from","this","that","have","been","will","your","their","when","what"}
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", description) if w.lower() not in stopwords]
    if not words:
        return []
    # Try each keyword with OR logic on short_description and text
    clauses = "^".join(f"short_descriptionLIKE{w}^ORtextLIKE{w}" for w in words[:3])
    results = sn_get("/api/now/table/kb_knowledge", {
        "sysparm_query": f"{clauses}^workflow_state=published",
        "sysparm_fields": "number,short_description,text",
        "sysparm_limit": 3
    })
    # Fallback: search just the first keyword if no results
    if not results and words:
        results = sn_get("/api/now/table/kb_knowledge", {
            "sysparm_query": f"short_descriptionLIKE{words[0]}^ORtextLIKE{words[0]}^workflow_state=published",
            "sysparm_fields": "number,short_description,text",
            "sysparm_limit": 3
        })
    return results


def tool_get_assignment_groups():
    return sn_get("/api/now/table/sys_user_group", {
        "sysparm_fields": "sys_id,name",
        "sysparm_limit": 100
    })


def tool_get_group_members(group_sys_id):
    results = sn_get("/api/now/table/sys_user_grmember", {
        "sysparm_query": f"group={group_sys_id}",
        "sysparm_fields": "user,user.name,user.email",
        "sysparm_display_value": "false",
        "sysparm_limit": 10
    })
    # Normalize: ensure user.sys_id is always populated
    for m in results:
        if "user" in m and not m.get("user.sys_id"):
            u = m["user"]
            m["user.sys_id"] = u.get("value", u) if isinstance(u, dict) else u
        if not m.get("user.name"):
            m["user.name"] = m.get("user.display_value", "Unknown")
    return results


def tool_update_incident(incident_sys_id, group_sys_id, user_sys_id):
    return sn_patch(f"/api/now/table/incident/{incident_sys_id}", {
        "assignment_group": group_sys_id,
        "assigned_to": user_sys_id
    })


def tool_verify_incident(number):
    return tool_fetch_incident(number)


def tool_create_kb(short_description, text, category="general"):
    return sn_post("/api/now/table/kb_knowledge", {
        "short_description": short_description,
        "text": text,
        "kb_category": category,
        "workflow_state": "published"
    })


def tool_create_interaction(short_description, description, caller_sys_id="", incident_sys_id=""):
    payload = {
        "short_description": short_description,
        "description":       description,
        "channel":           "chat",
        "state":             "closed_complete",
        "direction":         "inbound",
    }
    if caller_sys_id:
        payload["opened_for"] = caller_sys_id
    if incident_sys_id:
        payload["incident"] = incident_sys_id
    return sn_post("/api/now/table/interaction", payload)


def tool_fetch_interaction(number):
    results = sn_get("/api/now/table/interaction", {
        "sysparm_query": f"number={number.upper()}",
        "sysparm_display_value": "true",
        "sysparm_limit": 1
    })
    return results[0] if results else None


# ── LLM helpers ───────────────────────────────────────────────────────────────

def recommend_group(description):
    groups = tool_get_assignment_groups()
    group_names = [g["name"] for g in groups]
    names_list = "\n".join(f"- {n}" for n in group_names)
    resp = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": (
                "You are an IT service desk assistant. Based on the incident description, "
                "pick the most suitable assignment group from the EXACT list below.\n"
                "You MUST reply with ONLY one name copied exactly from this list, nothing else:\n"
                f"{names_list}"
            )},
            {"role": "user", "content": f"Incident: {description}"}
        ],
        model="llama-3.3-70b-versatile"
    )
    llm_pick = resp.choices[0].message.content.strip()
    # Exact match first
    matched = next((n for n in group_names if n == llm_pick), None)
    # Case-insensitive fallback
    if not matched:
        matched = next((n for n in group_names if n.lower() == llm_pick.lower()), None)
    # Partial fallback
    if not matched:
        matched = next((n for n in group_names if llm_pick.lower() in n.lower()), None)
    if not matched:
        matched = next((n for n in group_names if any(w in n.lower() for w in llm_pick.lower().split())), group_names[0] if group_names else "Service Desk")
    return matched, getattr(resp, "usage", None)


def generate_kb_article(short_description, description):
    resp = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": (
                "You are an IT knowledge base writer. Given an incident title and description, "
                "write a clear, concise KB article with: Problem, Root Cause, and Solution sections. "
                "Keep it under 300 words. Plain text only, no markdown."
            )},
            {"role": "user", "content": f"Title: {short_description}\nDescription: {description}"}
        ],
        model="llama-3.3-70b-versatile"
    )
    return resp.choices[0].message.content.strip(), getattr(resp, "usage", None)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/config")
async def config():
    return {"url": SN_URL, "user": SN_USER}


@app.get("/test-connection")
async def test_connection():
    try:
        all_groups = tool_get_assignment_groups()
        return {"ok": True, "groups": len(all_groups)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class InteractionRequest(BaseModel):
    short_description: str
    description:       str = ""
    caller_sys_id:     str = ""
    incident_sys_id:   str = ""


@app.post("/create-interaction")
async def create_interaction_endpoint(req: InteractionRequest):
    try:
        result = tool_create_interaction(
            short_description = req.short_description,
            description       = req.description,
            caller_sys_id     = req.caller_sys_id,
            incident_sys_id   = req.incident_sys_id
        )
        return {
            "ok":     True,
            "number": result.get("number", "N/A"),
            "sys_id": result.get("sys_id", "N/A"),
            "state":  result.get("state",  "N/A")
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/interaction/{number}")
async def get_interaction(number: str):
    try:
        result = tool_fetch_interaction(number)
        if not result:
            return {"ok": False, "error": f"Interaction {number} not found"}
        return {"ok": True, "interaction": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _merge_usage(acc, usage):
    if not usage: return
    acc["input_tokens"]  += getattr(usage, "prompt_tokens", 0)
    acc["output_tokens"] += getattr(usage, "completion_tokens", 0)
    acc["tool_tokens"]   += getattr(usage, "prompt_tokens", 0)
    acc["total_tokens"]  += getattr(usage, "total_tokens", 0)


@app.post("/chat")
async def chat(req: ChatRequest):
    sid         = req.session_id
    msg         = req.message.strip()
    agent       = req.agent
    tool_calls  = []
    token_usage = {"input_tokens": 0, "output_tokens": 0, "tool_tokens": 0, "total_tokens": 0}

    session_key = f"{sid}_{agent}"
    if session_key not in sessions:
        sessions[session_key] = {"step": "get_incident"}

    session = sessions[session_key]
    step    = session["step"]
    reply   = ""

    # ════════════════════════════════════════════════════════════════════════
    # INCIDENT AGENT
    # ════════════════════════════════════════════════════════════════════════
    if agent == "incident":

        # ── Natural language commands (any step) ─────────────────────────────
        nl = msg.lower()

        def pri_icon(p):
            p = str(p).lower()
            if "1" in p or "critical" in p: return "🔴"
            if "2" in p or "high" in p:     return "🟠"
            if "3" in p or "moderate" in p: return "🟡"
            if "4" in p or "low" in p:      return "🟢"
            return "⚪"

        if any(x in nl for x in ["show all group", "list group", "all group", "assignment group"]):
            groups = tool_get_assignment_groups()
            tool_calls.append({"name": "get_assignment_groups", "args": {}, "result": groups[:5]})
            lines = "\n".join(f"**{i+1}.** {g['name']}" for i, g in enumerate(groups))
            reply = f"\U0001f465 **All Assignment Groups ({len(groups)}):**\n\n{lines}"
            return {"reply": reply, "tool_calls": tool_calls}

        if any(x in nl for x in ["show all incident", "list incident", "all incident", "open incident", "active incident"]):
            incs = sn_get("/api/now/table/incident", {
                "sysparm_query": "state!=6^state!=7",
                "sysparm_fields": "number,short_description,priority,state",
                "sysparm_display_value": "true",
                "sysparm_limit": 20
            })
            tool_calls.append({"name": "fetch_all_incidents", "args": {"filter": "open"}, "result": {"count": len(incs)}})
            lines = "\n".join(f"{pri_icon(i.get('priority',''))} **{i['number']}** -- {i.get('short_description','N/A')}" for i in incs)
            reply = f"\U0001f4cb **Open Incidents ({len(incs)}):**\n\n{lines}"
            return {"reply": reply, "tool_calls": tool_calls}

        if any(x in nl for x in ["closed incident", "resolved incident"]):
            incs = sn_get("/api/now/table/incident", {
                "sysparm_query": "state=6^ORstate=7",
                "sysparm_fields": "number,short_description,priority,resolved_at",
                "sysparm_display_value": "true",
                "sysparm_limit": 20
            })
            tool_calls.append({"name": "fetch_closed_incidents", "args": {"filter": "closed"}, "result": {"count": len(incs)}})
            lines = "\n".join(f"OK **{i['number']}** -- {i.get('short_description','N/A')} *(resolved: {i.get('resolved_at','N/A')})*" for i in incs)
            reply = f"\U0001f4cb **Closed/Resolved Incidents ({len(incs)}):**\n\n{lines}"
            return {"reply": reply, "tool_calls": tool_calls}

        if any(x in nl for x in ["sla", "sla time", "breach", "due time"]):
            incs = sn_get("/api/now/table/incident", {
                "sysparm_query": "state!=6^state!=7",
                "sysparm_fields": "number,short_description,priority,made_sla,sla_due",
                "sysparm_display_value": "true",
                "sysparm_limit": 20
            })
            tool_calls.append({"name": "fetch_sla_status", "args": {}, "result": {"count": len(incs)}})
            breached = [i for i in incs if str(i.get("made_sla","true")).lower() == "false"]
            ok       = [i for i in incs if str(i.get("made_sla","true")).lower() != "false"]
            lines = ""
            if breached:
                lines += "**SLA Breached:**\n" + "\n".join(f"X **{i['number']}** -- {i.get('short_description','N/A')} (due: {i.get('sla_due','N/A')})" for i in breached)
            if ok:
                lines += "\n\n**Within SLA:**\n" + "\n".join(f"OK **{i['number']}** -- {i.get('short_description','N/A')}" for i in ok[:10])
            reply = f"**SLA Status -- Open Incidents:**\n\n**Breached:** {len(breached)}  |  **On Track:** {len(ok)}\n\n{lines}"
            return {"reply": reply, "tool_calls": tool_calls}

        if any(x in nl for x in ["similar", "related kb", "any kb", "kb for this", "kb article"]):
            inc = session.get("incident")
            if inc:
                kb = tool_search_kb(inc.get("short_description", ""))
                tool_calls.append({"name": "search_kb", "args": {"description": inc.get("short_description")}, "result": kb})
                if kb:
                    lines = ""
                    for i, art in enumerate(kb, 1):
                        clean = re.sub(r"<[^>]+>", " ", art.get("text", "")).strip()
                        lines += f"\n**{i}. {art.get('short_description')}** ({art.get('number')})\n{clean[:300]}\n"
                    reply = f"\U0001f4da **Related KB Articles:**\n{lines}"
                else:
                    reply = "📭 No related KB articles found for the current incident."
            else:
                reply = "⚠️ No incident loaded. Please enter an incident number first."
            return {"reply": reply, "tool_calls": tool_calls}

        if any(x in nl for x in ["members of", "who is in", "show members", "list members"]):
            groups = tool_get_assignment_groups()
            tool_calls.append({"name": "get_assignment_groups", "args": {}, "result": groups[:3]})
            matched = next((g for g in groups if g["name"].lower() in nl), None)
            if not matched:
                words = [w for w in nl.split() if len(w) > 3]
                matched = next((g for g in groups if any(w in g["name"].lower() for w in words)), None)
            if matched:
                members = tool_get_group_members(matched["sys_id"])
                tool_calls.append({"name": "get_group_members", "args": {"group_sys_id": matched["sys_id"]}, "result": members})
                if members:
                    lines = "\n".join(f"**{i+1}.** {m.get('user.name','?')} -- {m.get('user.email','?')}" for i, m in enumerate(members))
                    reply = f"\U0001f464 **Members of {matched['name']} ({len(members)}):**\n\n{lines}"
                else:
                    reply = f"⚠️ No members found in **{matched['name']}**."
            else:
                reply = "⚠️ Could not find that group. Try: **members of Service Desk**"
            return {"reply": reply, "tool_calls": tool_calls}

        if any(x in nl for x in ["help", "what can", "commands", "what do"]):
            reply = (
                "\U0001f916 **Incident Agent -- What I can do:**\n\n"
                "- **INC0000018** -> fetch details, search KB, assign\n"
                "- **show all groups** -> list assignment groups\n"
                "- **open incidents** / **all incidents** -> open tickets\n"
                "- **closed incidents** -> resolved tickets\n"
                "- **members of Service Desk** -> list group members\n"
                "- **sla** -> check SLA breach status\n"
                "- **similar kb** -> find KB for current incident"
            )
            return {"reply": reply, "tool_calls": tool_calls}

        if step == "get_incident":
            if not msg:
                reply = "👋 Hello! Please enter the **incident number** you'd like help with (e.g. INC0000018)."
            else:
                match = re.search(r"INC\d+", msg, re.IGNORECASE)
                if not match:
                    reply = "⚠️ Please enter a valid incident number like **INC0000018**."
                else:
                    inc_num = match.group(0).upper()
                    inc = tool_fetch_incident(inc_num)
                    tool_calls.append({"name": "fetch_incident", "args": {"number": inc_num}, "result": inc or {}})

                    if not inc:
                        reply = f"❌ Incident **{inc_num}** not found."
                    else:
                        session["incident"] = inc
                        priority_map = {"1": "🔴 Critical", "2": "🟠 High", "3": "🟡 Moderate", "4": "🟢 Low", "5": "⚪ Planning"}
                        state_map    = {"1": "New", "2": "In Progress", "3": "On Hold", "6": "Resolved", "7": "Closed"}
                        priority = priority_map.get(str(inc.get("priority", "")), inc.get("priority", "N/A"))
                        state    = state_map.get(str(inc.get("state", "")), inc.get("state", "N/A"))

                        reply = (
                            f"📋 **Incident Details — {inc_num}**\n\n"
                            f"**Description:** {inc.get('short_description', 'N/A')}\n"
                            f"**Full Description:** {inc.get('description', 'N/A')}\n"
                            f"**Priority:** {priority}\n"
                            f"**State:** {state}\n"
                            f"**Caller:** {inc.get('caller_id', {}).get('display_value', 'N/A') if isinstance(inc.get('caller_id'), dict) else inc.get('caller_id', 'N/A')}\n"
                            f"**Opened:** {inc.get('opened_at', 'N/A')}\n\n"
                            f"🔍 Searching Knowledge Base..."
                        )

                        kb = tool_search_kb(inc.get("short_description", ""))
                        tool_calls.append({"name": "search_kb", "args": {"description": inc.get("short_description", "")}, "result": kb})

                        if kb:
                            kb_text = "\n\n📚 **Related KB Articles:**\n"
                            for i, art in enumerate(kb, 1):
                                clean = re.sub(r"<[^>]+>", " ", art.get("text", "")).strip()
                                kb_text += f"\n**{i}. {art.get('short_description')}** ({art.get('number')})\n{clean[:300]}{'...' if len(clean) > 300 else ''}\n"
                            kb_text += "\n\n✅ Did this solve your issue? **(yes / no)**"
                            reply += kb_text
                            session["step"] = "kb_solved"
                        else:
                            reply += "\n\n📭 No KB articles found.\n\n"
                            group, usage = recommend_group(inc.get("short_description", ""))
                            _merge_usage(token_usage, usage)
                            session["recommended_group"] = group
                            reply += f"🤖 I recommend assigning to **{group}**.\n\nShall I assign it? **(yes / no)**"
                            session["step"] = "confirm_assign"

        elif step == "kb_solved":
            if msg.lower() in ["yes", "y"]:
                reply = "🎉 Great! Issue resolved via KB. Conversation closed."
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                inc   = session.get("incident", {})
                group, usage = recommend_group(inc.get("short_description", ""))
                _merge_usage(token_usage, usage)
                session["recommended_group"] = group
                reply = f"🤖 I recommend assigning to **{group}**.\n\nShall I assign it? **(yes / no)**"
                session["step"] = "confirm_assign"
            else:
                reply = "Please reply **yes** or **no**."

        elif step == "confirm_assign":
            if msg.lower() in ["no", "n"]:
                reply = "Please suggest a **keyword or group name**:"
                session["step"] = "manual_group"
            elif msg.lower() in ["yes", "y"]:
                session["step"] = "do_assign"

        if step == "manual_group":
            group, usage = recommend_group(msg)
            _merge_usage(token_usage, usage)
            session["recommended_group"] = group
            reply = f"🤖 Based on '{msg}', I suggest **{group}**.\n\nShall I assign it? **(yes / no)**"
            session["step"] = "confirm_assign"

        if step == "do_assign" or (step == "confirm_assign" and msg.lower() in ["yes", "y"]):
            inc         = session.get("incident", {})
            recommended = session.get("recommended_group", "Service Desk")
            sys_id      = inc.get("sys_id", "")
            try:
                groups = tool_get_assignment_groups()
                tool_calls.append({"name": "get_assignment_groups", "args": {}, "result": groups[:3]})

                matched = next((g for g in groups if g["name"] == recommended), None)
                if not matched:
                    matched = next((g for g in groups if g["name"].lower() == recommended.lower()), None)
                if not matched:
                    raise Exception(f"No group found matching '{recommended}'")

                group_sys_id = matched["sys_id"]
                group_name   = matched["name"]

                members = tool_get_group_members(group_sys_id)
                tool_calls.append({"name": "get_group_members", "args": {"group_sys_id": group_sys_id}, "result": members[:2]})

                user_sys_id = ""
                user_name   = "Unassigned"
                if members:
                    member      = members[0]
                    user_sys_id = member.get("user.sys_id", "") or member.get("user", "")
                    user_name   = member.get("user.name", "Unknown")

                update_result = tool_update_incident(sys_id, group_sys_id, user_sys_id)
                tool_calls.append({"name": "update_incident_assignment", "args": {
                    "incident_sys_id": sys_id,
                    "assignment_group_sys_id": group_sys_id,
                    "assigned_to_sys_id": user_sys_id
                }, "result": update_result})

                verified = tool_verify_incident(inc.get("number", ""))
                tool_calls.append({"name": "verify_incident", "args": {"number": inc.get("number")}, "result": verified or {}})

                v_group, v_user = group_name, user_name
                if verified:
                    ag = verified.get("assignment_group", {})
                    at = verified.get("assigned_to", {})
                    v_group = ag.get("display_value", group_name) if isinstance(ag, dict) else ag
                    v_user  = at.get("display_value", user_name)  if isinstance(at, dict) else at

                # ── Create Interaction record ─────────────────────────────
                caller_sys_id = ""
                caller_raw    = inc.get("caller_id", {})
                if isinstance(caller_raw, dict):
                    caller_sys_id = caller_raw.get("value", "")
                try:
                    interaction = tool_create_interaction(
                        short_description = inc.get("short_description", ""),
                        description       = (
                            f"AI Service Desk chat session.\n"
                            f"Incident: {inc.get('number')}\n"
                            f"Assigned to group: {v_group}\n"
                            f"Assigned to: {v_user}"
                        ),
                        caller_sys_id  = caller_sys_id,
                        incident_sys_id = sys_id
                    )
                    tool_calls.append({"name": "create_interaction", "args": {
                        "incident": inc.get("number"),
                        "channel": "chat"
                    }, "result": {"number": interaction.get("number", "N/A")}})
                    interaction_num = interaction.get("number", "N/A")
                except Exception as ie:
                    interaction_num = f"failed ({str(ie)[:40]})"

                reply = (
                    f"✅ **Incident Successfully Assigned!**\n\n"
                    f"**Incident:** {inc.get('number')}\n"
                    f"**Group:** {v_group}\n"
                    f"**Assigned To:** {v_user}\n"
                    f"**Interaction Record:** {interaction_num}"
                )
            except Exception as e:
                reply = f"⚠️ Assignment failed: {str(e)}"
            sessions.pop(session_key, None)

    # ════════════════════════════════════════════════════════════════════════
    # KB AGENT
    # ════════════════════════════════════════════════════════════════════════
    elif agent == "kb":

        if step == "get_incident":
            if not msg:
                reply = (
                    "📚 Hello! I'm the **KB Agent**.\n\n"
                    "You can:\n"
                    "- Enter an **incident number** (e.g. INC0000018) to view details & linked KB articles\n"
                    "- Enter a **KB number** (e.g. KB0000001) to view KB article details\n"
                    "- Type **scan** to find incidents without KB articles\n"
                    "- Type **with kb** to see incidents that already have KB coverage"
                )

            elif any(x in msg.lower() for x in ["with kb", "has kb", "incidents with kb", "show kb", "have kb"]):
                incidents = sn_get("/api/now/table/incident", {
                    "sysparm_query": "state!=6^state!=7",
                    "sysparm_fields": "number,short_description",
                    "sysparm_display_value": "true",
                    "sysparm_limit": 30
                })
                tool_calls.append({"name": "fetch_all_incidents", "args": {"filter": "open"}, "result": {"count": len(incidents)}})
                covered = []
                for inc in incidents:
                    kb = tool_search_kb(inc.get("short_description", ""))
                    if kb:
                        covered.append({"inc": inc, "kb": kb[0]})
                tool_calls.append({"name": "scan_kb_coverage", "args": {}, "result": {"covered": len(covered)}})
                if not covered:
                    reply = "📭 No open incidents found with matching KB articles."
                else:
                    lines = ""
                    for item in covered:
                        lines += f"\n✅ **{item['inc']['number']}** — {item['inc'].get('short_description','N/A')}\n   📚 KB: **{item['kb'].get('short_description','N/A')}** ({item['kb'].get('number','')})"
                    reply = f"📚 **Incidents with KB Coverage ({len(covered)}):**\n{lines}"
                return {"reply": reply, "tool_calls": tool_calls, "token_usage": token_usage}

            elif msg.lower() in ["scan", "scan all", "check all", "all incidents"]:
                incidents = sn_get("/api/now/table/incident", {
                    "sysparm_query": "state!=6^state!=7",
                    "sysparm_fields": "number,short_description,sys_id",
                    "sysparm_display_value": "true",
                    "sysparm_limit": 20
                })
                tool_calls.append({"name": "fetch_all_incidents", "args": {"filter": "open"}, "result": {"count": len(incidents)}})
                missing = []
                for inc in incidents:
                    kb = tool_search_kb(inc.get("short_description", ""))
                    if not kb:
                        missing.append(inc)
                tool_calls.append({"name": "scan_kb_coverage", "args": {"checked": len(incidents)}, "result": {"missing": len(missing)}})
                if not missing:
                    reply = f"✅ All **{len(incidents)}** open incidents have KB coverage!"
                else:
                    session["missing_kb"]  = missing
                    session["missing_idx"] = 0
                    lines = "".join(f"\n**{i}. {inc['number']}** — {inc.get('short_description','N/A')}" for i, inc in enumerate(missing, 1))
                    reply = (
                        f"📋 Found **{len(missing)}** incidents without KB articles (out of {len(incidents)} open):\n"
                        f"{lines}\n\n"
                        f"Shall I auto-create KB articles for all of them? **(yes / no)**"
                    )
                    session["step"] = "confirm_bulk_kb"

            else:
                kb_match  = re.search(r"KB\d+",  msg, re.IGNORECASE)
                inc_match = re.search(r"INC\d+", msg, re.IGNORECASE)

                if kb_match:
                    # ── KB number: show article details ──────────────────────
                    kb_num  = kb_match.group(0).upper()
                    results = sn_get("/api/now/table/kb_knowledge", {
                        "sysparm_query": f"number={kb_num}",
                        "sysparm_fields": "number,short_description,text,workflow_state,kb_category,sys_created_on",
                        "sysparm_display_value": "true",
                        "sysparm_limit": 1
                    })
                    tool_calls.append({"name": "fetch_kb_article", "args": {"number": kb_num}, "result": results[0] if results else {}})
                    if not results:
                        reply = f"❌ KB article **{kb_num}** not found."
                    else:
                        art   = results[0]
                        clean = re.sub(r"<[^>]+>", " ", art.get("text", "")).strip()
                        reply = (
                            f"📚 **KB Article — {kb_num}**\n\n"
                            f"**Title:** {art.get('short_description', 'N/A')}\n"
                            f"**Category:** {art.get('kb_category', 'N/A')}\n"
                            f"**Status:** {art.get('workflow_state', 'N/A')}\n"
                            f"**Created:** {art.get('sys_created_on', 'N/A')}\n\n"
                            f"**Content:**\n{clean[:800]}{'...' if len(clean) > 800 else ''}"
                        )

                elif inc_match:
                    # ── INC number: show incident details + linked KB ─────────
                    inc_num = inc_match.group(0).upper()
                    inc     = tool_fetch_incident(inc_num)
                    tool_calls.append({"name": "fetch_incident", "args": {"number": inc_num}, "result": inc or {}})
                    if not inc:
                        reply = f"❌ Incident **{inc_num}** not found."
                    else:
                        session["incident"] = inc
                        priority_map = {"1": "🔴 Critical", "2": "🟠 High", "3": "🟡 Moderate", "4": "🟢 Low", "5": "⚪ Planning"}
                        state_map    = {"1": "New", "2": "In Progress", "3": "On Hold", "6": "Resolved", "7": "Closed"}
                        caller = inc.get("caller_id", {})
                        caller = caller.get("display_value", "N/A") if isinstance(caller, dict) else caller
                        reply  = (
                            f"📋 **Incident Details — {inc_num}**\n\n"
                            f"**Description:** {inc.get('short_description', 'N/A')}\n"
                            f"**Full Description:** {inc.get('description', 'N/A')}\n"
                            f"**Priority:** {priority_map.get(str(inc.get('priority','')), inc.get('priority','N/A'))}\n"
                            f"**State:** {state_map.get(str(inc.get('state','')), inc.get('state','N/A'))}\n"
                            f"**Caller:** {caller}\n"
                            f"**Opened:** {inc.get('opened_at', 'N/A')}\n"
                        )
                        short_desc = inc.get("short_description", "")
                        kb = tool_search_kb(short_desc)
                        tool_calls.append({"name": "search_kb", "args": {"description": short_desc}, "result": kb})
                        if kb:
                            reply += "\n📚 **Linked KB Articles:**\n"
                            for i, art in enumerate(kb, 1):
                                clean  = re.sub(r"<[^>]+>", " ", art.get("text", "")).strip()
                                reply += (
                                    f"\n**{i}. {art.get('short_description')}** ({art.get('number')})\n"
                                    f"{clean[:400]}{'...' if len(clean) > 400 else ''}\n"
                                )
                            reply += "\nWould you like to **create a new KB** for this incident? **(yes / no)**"
                            session["step"] = "confirm_kb_create_from_inc"
                        else:
                            reply += "\n📭 **No linked KB articles found.**\n\n🤖 Generating KB article..."
                            article_text, usage = generate_kb_article(short_desc, inc.get("description", ""))
                            _merge_usage(token_usage, usage)
                            tool_calls.append({"name": "generate_kb_content", "args": {"short_description": short_desc}, "result": {"preview": article_text[:100]}})
                            session["kb_title"]   = short_desc
                            session["kb_content"] = article_text
                            session["step"]       = "confirm_kb_create"
                            reply += (
                                f"\n\n📝 **Proposed KB Article:**\n\n"
                                f"**Title:** {short_desc}\n\n"
                                f"{article_text}\n\n"
                                f"Shall I create this KB article in ServiceNow? **(yes / no)**"
                            )
                else:
                    reply = "⚠️ Please enter a valid **incident number** (INC0000018) or **KB number** (KB0000001), or type **scan**."

        elif step == "confirm_kb_create_from_inc":
            if msg.lower() in ["yes", "y"]:
                inc        = session.get("incident", {})
                short_desc = inc.get("short_description", "")
                article_text, usage = generate_kb_article(short_desc, inc.get("description", ""))
                _merge_usage(token_usage, usage)
                tool_calls.append({"name": "generate_kb_content", "args": {"short_description": short_desc}, "result": {"preview": article_text[:100]}})
                session["kb_title"]   = short_desc
                session["kb_content"] = article_text
                session["step"]       = "confirm_kb_create"
                reply = (
                    f"📝 **Proposed KB Article:**\n\n"
                    f"**Title:** {short_desc}\n\n"
                    f"{article_text}\n\n"
                    f"Shall I create this KB article in ServiceNow? **(yes / no)**"
                )
            elif msg.lower() in ["no", "n"]:
                reply = "👍 No KB article created."
                sessions.pop(session_key, None)
            else:
                reply = "Please reply **yes** or **no**."

        elif step == "confirm_bulk_kb":
            if msg.lower() in ["yes", "y"]:
                missing        = session.get("missing_kb", [])
                created, failed = [], []
                for inc in missing:
                    try:
                        short_desc   = inc.get("short_description", "")
                        article_text, usage = generate_kb_article(short_desc, inc.get("description", short_desc))
                        _merge_usage(token_usage, usage)
                        result = tool_create_kb(short_desc, article_text)
                        tool_calls.append({"name": "create_kb_article", "args": {"number": inc["number"]}, "result": {"kb_number": result.get("number", "N/A")}})
                        created.append(f"{inc['number']} → {result.get('number', 'N/A')}")
                    except Exception as e:
                        failed.append(f"{inc['number']} ({str(e)[:40]})")
                lines  = "\n".join(f"✅ {c}" for c in created)
                lines += ("\n" + "\n".join(f"❌ {f}" for f in failed)) if failed else ""
                reply  = f"✅ **Bulk KB Creation Complete!**\n\n**Created:** {len(created)}  |  **Failed:** {len(failed)}\n\n{lines}"
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                missing = session.get("missing_kb", [])
                lines   = "".join(f"\n**{i}. {inc['number']}** — {inc.get('short_description','N/A')}" for i, inc in enumerate(missing, 1))
                reply   = f"👍 No problem. Incidents without KB:\n{lines}"
                sessions.pop(session_key, None)
            else:
                reply = "Please reply **yes** or **no**."

        elif step == "confirm_kb_create":
            if msg.lower() in ["yes", "y"]:
                title   = session.get("kb_title", "")
                content = session.get("kb_content", "")
                try:
                    result = tool_create_kb(title, content)
                    tool_calls.append({"name": "create_kb_article", "args": {"short_description": title}, "result": result})
                    reply = (
                        f"✅ **KB Article Created Successfully!**\n\n"
                        f"**KB Number:** {result.get('number', 'N/A')}\n"
                        f"**Title:** {title}\n"
                        f"**Status:** Published"
                    )
                except Exception as e:
                    reply = f"⚠️ Failed to create KB article: {str(e)}"
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                reply = "👍 KB article creation cancelled."
                sessions.pop(session_key, None)
            else:
                reply = "Please reply **yes** or **no**."

    return {"reply": reply, "tool_calls": tool_calls, "token_usage": token_usage}


# ── Recurrence Analysis Endpoint ─────────────────────────────────────────────

@app.get("/recurrence-analysis")
async def recurrence_analysis():
    import re as _re
    from collections import Counter, defaultdict

    # Fetch last 100 incidents
    incidents = sn_get("/api/now/table/incident", {
        "sysparm_fields": "number,short_description,description,category,location,assignment_group,priority,state,opened_at,caller_id",
        "sysparm_display_value": "true",
        "sysparm_limit": 100,
        "sysparm_query": "ORDERBYDESCopened_at"
    })

    if not incidents:
        return {"error": "No incidents found"}

    # ── Keyword extraction ────────────────────────────────────────────────────
    stopwords = {"the","a","an","is","in","on","at","to","for","of","and","or",
                 "not","with","this","that","it","be","are","was","has","have",
                 "i","my","we","our","can","cannot","when","how","what","why",
                 "from","by","as","but","if","so","do","did","been","will","get"}

    keyword_counter = Counter()
    for inc in incidents:
        text = (inc.get("short_description","") + " " + inc.get("description","")).lower()
        words = _re.findall(r"[a-z]{4,}", text)
        for w in words:
            if w not in stopwords:
                keyword_counter[w] += 1

    top_keywords = [{"keyword": k, "count": v} for k, v in keyword_counter.most_common(15)]

    # ── Category distribution ─────────────────────────────────────────────────
    cat_counter = Counter()
    for inc in incidents:
        cat = inc.get("category") or "Unknown"
        if isinstance(cat, dict): cat = cat.get("display_value", "Unknown")
        cat_counter[cat] += 1
    categories = [{"name": k, "count": v} for k, v in cat_counter.most_common(10)]

    # ── Priority distribution ─────────────────────────────────────────────────
    pri_map = {"1": "Critical", "2": "High", "3": "Moderate", "4": "Low", "5": "Planning"}
    pri_counter = Counter()
    for inc in incidents:
        p = str(inc.get("priority",""))
        pri_counter[pri_map.get(p, p or "Unknown")] += 1
    priorities = [{"name": k, "count": v} for k, v in pri_counter.most_common()]

    # ── Location hotspots ─────────────────────────────────────────────────────
    loc_counter = Counter()
    for inc in incidents:
        loc = inc.get("location") or "Unknown"
        if isinstance(loc, dict): loc = loc.get("display_value", "Unknown")
        if loc: loc_counter[loc] += 1
    locations = [{"name": k, "count": v} for k, v in loc_counter.most_common(8)]

    # ── Assignment group load ─────────────────────────────────────────────────
    grp_counter = Counter()
    for inc in incidents:
        grp = inc.get("assignment_group") or "Unassigned"
        if isinstance(grp, dict): grp = grp.get("display_value", "Unassigned")
        grp_counter[grp] += 1
    groups = [{"name": k, "count": v} for k, v in grp_counter.most_common(8)]

    # ── Recurrence clusters (same keyword appears 3+ times) ───────────────────
    clusters = []
    for kw, cnt in keyword_counter.most_common(10):
        if cnt < 2: break
        related = []
        for inc in incidents:
            text = (inc.get("short_description","") + " " + inc.get("description","")).lower()
            if kw in text:
                related.append(inc.get("number",""))
        if len(related) >= 2:
            clusters.append({"keyword": kw, "count": cnt, "incidents": related[:6]})

    # ── Monthly trend ─────────────────────────────────────────────────────────
    month_counter = Counter()
    for inc in incidents:
        opened = inc.get("opened_at","")
        if opened and len(opened) >= 7:
            month_counter[opened[:7]] += 1
    trend = [{"month": k, "count": v} for k, v in sorted(month_counter.items())[-6:]]

    # ── LLM Root Cause Analysis ───────────────────────────────────────────────
    top_cluster_kws = ", ".join(c["keyword"] for c in clusters[:5])
    top_cats        = ", ".join(c["name"] for c in categories[:5])
    rca_prompt = (
        f"You are an IT operations analyst. Based on these recurring incident patterns:\n"
        f"Top keywords: {top_cluster_kws}\n"
        f"Top categories: {top_cats}\n"
        f"Total incidents analyzed: {len(incidents)}\n\n"
        f"Provide a concise root cause analysis with:\n"
        f"1. Top 3 root causes\n2. Risk areas\n3. Recommended preventive actions\n"
        f"Keep it under 200 words. Plain text."
    )
    try:
        rca_resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": rca_prompt}],
            model="llama-3.3-70b-versatile"
        )
        rca = rca_resp.choices[0].message.content.strip()
    except:
        rca = "Root cause analysis unavailable."

    return {
        "total": len(incidents),
        "keywords": top_keywords,
        "categories": categories,
        "priorities": priorities,
        "locations": locations,
        "groups": groups,
        "clusters": clusters,
        "trend": trend,
        "rca": rca
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("templates/dashboard.html", encoding="utf-8") as f:
        return f.read()


@app.get("/interaction-form", response_class=HTMLResponse)
async def interaction_form():
    with open("templates/interaction_form.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/similar-incidents")
async def similar_incidents(description: str = ""):
    if not description or len(description.strip()) < 3:
        return {"incidents": [], "estimate": None}

    stopwords = {"with","from","this","that","have","been","will","your","when","what","the","and","for","issues","issue","problem"}
    words = [w for w in re.findall(r"[a-z]{3,}", description.lower()) if w not in stopwords]
    if not words:
        return {"incidents": [], "estimate": None}

    clauses = "^OR".join(f"short_descriptionLIKE{w}" for w in words[:4])
    incs = sn_get("/api/now/table/incident", {
        "sysparm_query":         clauses + "^ORDERBYDESCopened_at",
        "sysparm_fields":        "number,short_description,priority,state,opened_at,resolved_at,sys_id",
        "sysparm_display_value": "false",
        "sysparm_limit":         5
    })

    pri_map   = {"1":"Critical","2":"High","3":"Moderate","4":"Low"}
    state_map = {"1":"New","2":"In Progress","3":"On Hold","6":"Resolved","7":"Closed"}

    result = []
    total_days, resolved_count = 0, 0
    for inc in incs:
        result.append({
            "number":            inc.get("number"),
            "short_description": inc.get("short_description"),
            "priority":          pri_map.get(str(inc.get("priority","")), "Unknown"),
            "state":             state_map.get(str(inc.get("state","")), "Unknown"),
            "opened_at":         inc.get("opened_at",""),
            "sys_id":            inc.get("sys_id")
        })
        if inc.get("resolved_at") and inc.get("opened_at"):
            from datetime import datetime
            try:
                o  = datetime.strptime(inc["opened_at"],  "%Y-%m-%d %H:%M:%S")
                rv = datetime.strptime(inc["resolved_at"], "%Y-%m-%d %H:%M:%S")
                d  = (rv - o).days
                if d >= 0: total_days += d; resolved_count += 1
            except: pass

    estimate = None
    if resolved_count > 0:
        avg = round(total_days / resolved_count)
        estimate = {
            "label":  "Same day" if avg <= 1 else f"{avg} day(s)",
            "detail": f"Based on {resolved_count} similar resolved incident(s)"
        }

    return {"incidents": result, "estimate": estimate}


@app.get("/api/similar-kb")
async def similar_kb(description: str = ""):
    if not description or len(description.strip()) < 3:
        return {"articles": []}

    stopwords = {"with","from","this","that","have","been","will","your","when","what","the","and","for"}
    words = [w for w in re.findall(r"[a-z]{3,}", description.lower()) if w not in stopwords]
    if not words:
        return {"articles": []}

    clauses = "^OR".join(f"short_descriptionLIKE{w}^ORtextLIKE{w}" for w in words[:3])
    kbs = sn_get("/api/now/table/kb_knowledge", {
        "sysparm_query":  clauses + "^workflow_state=published",
        "sysparm_fields": "number,short_description,kb_category,sys_id",
        "sysparm_limit":  5
    })

    return {"articles": [{
        "number":            kb.get("number"),
        "short_description": kb.get("short_description"),
        "category":          kb.get("kb_category") or "General",
        "sys_id":            kb.get("sys_id")
    } for kb in kbs]}


@app.get("/api/users")
async def get_users(q: str = ""):
    params = {
        "sysparm_fields": "sys_id,name,email",
        "sysparm_display_value": "true",
        "sysparm_limit": 20
    }
    if q:
        params["sysparm_query"] = f"nameLIKE{q}^ORemailLIKE{q}"
    return sn_get("/api/now/table/sys_user", params)


@app.get("/api/groups")
async def get_groups():
    return tool_get_assignment_groups()


class AISuggestRequest(BaseModel):
    description: str


@app.post("/api/ai-suggest")
async def ai_suggest(req: AISuggestRequest):
    desc = req.description.strip()
    if not desc:
        return {"category": "", "assignment_group": "", "short_description": ""}

    groups = tool_get_assignment_groups()
    group_names = "\n".join(f"- {g['name']}" for g in groups)

    resp = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": (
                "You are an IT service desk classifier. Given an interaction description, return a JSON object with exactly 3 keys:\n"
                "1. category: one of [Incident, Requests, Inquiry, Complaint, Praise]\n"
                f"2. assignment_group: pick the best match from this list ONLY:\n{group_names}\n"
                "3. short_description: a concise 1-line summary (max 80 chars)\n"
                "Reply with ONLY valid JSON, no explanation."
            )},
            {"role": "user", "content": f"Description: {desc}"}
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=150
    )
    import json as _json
    try:
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        data = _json.loads(raw)
        # validate assignment_group is real
        group_list = [g["name"] for g in groups]
        ag = data.get("assignment_group", "")
        if ag not in group_list:
            ag = next((n for n in group_list if ag.lower() in n.lower()), group_list[0] if group_list else "")
        data["assignment_group"] = ag
        return data
    except Exception:
        return {"category": "Incident", "assignment_group": groups[0]["name"] if groups else "", "short_description": desc[:80]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

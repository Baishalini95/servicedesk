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
    return sn_get("/api/now/table/kb_knowledge", {
        "sysparm_query": f"textLIKE{description[:80]}^workflow_state=published",
        "sysparm_fields": "number,short_description,text",
        "sysparm_limit": 3
    })


def tool_get_assignment_groups():
    return sn_get("/api/now/table/sys_user_group", {
        "sysparm_fields": "sys_id,name",
        "sysparm_limit": 100
    })


def tool_get_group_members(group_sys_id):
    return sn_get("/api/now/table/sys_user_grmember", {
        "sysparm_query": f"group={group_sys_id}",
        "sysparm_fields": "user.sys_id,user.name,user.email",
        "sysparm_display_value": "true",
        "sysparm_limit": 10
    })


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


# ── LLM helpers ───────────────────────────────────────────────────────────────

def recommend_group(description):
    resp = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": (
                "You are an IT service desk assistant. Based on the incident description, "
                "recommend the most suitable assignment group from this list ONLY: "
                "Service Desk, Network Operations, Application Support, Infrastructure, "
                "Security Operations, Database Administration, IT Operations, Software, "
                "Hardware, Database, Network. Reply with ONLY the group name, nothing else."
            )},
            {"role": "user", "content": f"Incident: {description}"}
        ],
        model="llama-3.3-70b-versatile"
    )
    return resp.choices[0].message.content.strip()


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
    return resp.choices[0].message.content.strip()


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


@app.post("/chat")
async def chat(req: ChatRequest):
    sid        = req.session_id
    msg        = req.message.strip()
    agent      = req.agent
    tool_calls = []

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
                            group = recommend_group(inc.get("short_description", ""))
                            session["recommended_group"] = group
                            reply += f"🤖 I recommend assigning to **{group}**.\n\nShall I assign it? **(yes / no)**"
                            session["step"] = "confirm_assign"

        elif step == "kb_solved":
            if msg.lower() in ["yes", "y"]:
                reply = "🎉 Great! Issue resolved via KB. Conversation closed."
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                inc   = session.get("incident", {})
                group = recommend_group(inc.get("short_description", ""))
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
            group = recommend_group(msg)
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

                matched = next((g for g in groups if recommended.lower() in g["name"].lower()), None)
                if not matched:
                    matched = next((g for g in groups if any(w in g["name"].lower() for w in recommended.lower().split())), groups[0] if groups else None)
                if not matched:
                    raise Exception(f"No group found matching '{recommended}'")

                group_sys_id = matched["sys_id"]
                group_name   = matched["name"]

                members = tool_get_group_members(group_sys_id)
                tool_calls.append({"name": "get_group_members", "args": {"group_sys_id": group_sys_id}, "result": members[:2]})

                if not members:
                    raise Exception(f"No members in group '{group_name}'")

                member      = members[0]
                user_sys_id = member.get("user.sys_id", "")
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

                reply = (
                    f"✅ **Incident Successfully Assigned!**\n\n"
                    f"**Incident:** {inc.get('number')}\n"
                    f"**Group:** {v_group}\n"
                    f"**Assigned To:** {v_user}"
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
                reply = "📚 Hello! I'm the **KB Agent**.\n\nYou can:\n- Enter an **incident number** (e.g. INC0000018) to search/create KB\n- Type **scan** to find all incidents without KB articles"
            elif msg.lower() in ["scan", "scan all", "check all", "all incidents"]:
                # Scan all open incidents for missing KB
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
                    session["missing_kb"] = missing
                    session["missing_idx"] = 0
                    lines = ""
                    for i, inc in enumerate(missing, 1):
                        lines += f"\n**{i}. {inc['number']}** — {inc.get('short_description', 'N/A')}"
                    reply = (
                        f"📋 Found **{len(missing)}** incidents without KB articles (out of {len(incidents)} open):\n"
                        f"{lines}\n\n"
                        f"Shall I auto-create KB articles for all of them? **(yes / no)**"
                    )
                    session["step"] = "confirm_bulk_kb"
            else:
                match = re.search(r"INC\d+", msg, re.IGNORECASE)
                if not match:
                    reply = "⚠️ Please enter a valid incident number like **INC0000018**, or type **scan** to check all incidents."
                else:
                    inc_num = match.group(0).upper()
                    inc = tool_fetch_incident(inc_num)
                    tool_calls.append({"name": "fetch_incident", "args": {"number": inc_num}, "result": inc or {}})

                    if not inc:
                        reply = f"❌ Incident **{inc_num}** not found."
                    else:
                        session["incident"] = inc
                        short_desc = inc.get("short_description", "")
                        description = inc.get("description", "")

                        reply = f"🔍 Searching KB for: **{short_desc}**..."

                        kb = tool_search_kb(short_desc)
                        tool_calls.append({"name": "search_kb", "args": {"description": short_desc}, "result": kb})

                        if kb:
                            kb_text = "\n\n📚 **KB Articles Found:**\n"
                            for i, art in enumerate(kb, 1):
                                clean = re.sub(r"<[^>]+>", " ", art.get("text", "")).strip()
                                kb_text += f"\n**{i}. {art.get('short_description')}** ({art.get('number')})\n{clean[:400]}{'...' if len(clean) > 400 else ''}\n"
                            kb_text += "\n\n✅ Did this solve your issue? **(yes / no)**"
                            reply += kb_text
                            session["step"] = "kb_check"
                        else:
                            reply += "\n\n📭 **No KB articles found.**\n\n🤖 Generating KB article from incident details..."

                            article_text = generate_kb_article(short_desc, description)
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

        elif step == "confirm_bulk_kb":
            if msg.lower() in ["yes", "y"]:
                missing  = session.get("missing_kb", [])
                created  = []
                failed   = []
                for inc in missing:
                    try:
                        short_desc  = inc.get("short_description", "")
                        description = inc.get("description", short_desc)
                        article_text = generate_kb_article(short_desc, description)
                        result = tool_create_kb(short_desc, article_text)
                        tool_calls.append({"name": "create_kb_article", "args": {"number": inc["number"]}, "result": {"kb_number": result.get("number", "N/A")}})
                        created.append(f"{inc['number']} → {result.get('number', 'N/A')}")
                    except Exception as e:
                        failed.append(f"{inc['number']} ({str(e)[:40]})")

                lines = "\n".join(f"✅ {c}" for c in created)
                if failed:
                    lines += "\n" + "\n".join(f"❌ {f}" for f in failed)
                reply = (
                    f"✅ **Bulk KB Creation Complete!**\n\n"
                    f"**Created:** {len(created)}  |  **Failed:** {len(failed)}\n\n"
                    f"{lines}"
                )
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                missing = session.get("missing_kb", [])
                lines   = ""
                for i, inc in enumerate(missing, 1):
                    lines += f"\n**{i}. {inc['number']}** — {inc.get('short_description', 'N/A')}"
                reply = f"👍 No problem. Here are the incidents without KB for your reference:\n{lines}"
                sessions.pop(session_key, None)
            else:
                reply = "Please reply **yes** or **no**."

        elif step == "kb_check":
            if msg.lower() in ["yes", "y"]:
                reply = "🎉 Great! KB article resolved the issue."
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                inc = session.get("incident", {})
                short_desc  = inc.get("short_description", "")
                description = inc.get("description", "")

                reply = "🤖 Generating a new KB article from this incident..."
                article_text = generate_kb_article(short_desc, description)
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
                reply = "Please reply **yes** or **no**."

        elif step == "confirm_kb_create":
            if msg.lower() in ["yes", "y"]:
                title   = session.get("kb_title", "")
                content = session.get("kb_content", "")
                try:
                    result = tool_create_kb(title, content)
                    tool_calls.append({"name": "create_kb_article", "args": {"short_description": title}, "result": result})
                    kb_num = result.get("number", "N/A")
                    reply = (
                        f"✅ **KB Article Created Successfully!**\n\n"
                        f"**KB Number:** {kb_num}\n"
                        f"**Title:** {title}\n"
                        f"**Status:** Published\n\n"
                        f"The article is now available in the ServiceNow Knowledge Base."
                    )
                except Exception as e:
                    reply = f"⚠️ Failed to create KB article: {str(e)}"
                sessions.pop(session_key, None)
            elif msg.lower() in ["no", "n"]:
                reply = "👍 KB article creation cancelled."
                sessions.pop(session_key, None)
            else:
                reply = "Please reply **yes** or **no**."

    return {"reply": reply, "tool_calls": tool_calls}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

// Chat front end for the Agentic RAG API.
// History is kept in sessionStorage, so it survives a page refresh but clears when the tab closes.

const HISTORY_KEY = "agentic-rag-history";

const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");
const form = document.getElementById("chat-form");
const input = document.getElementById("query");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

let history = loadHistory();

function loadHistory() {
  try {
    return JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // storage unavailable (private mode etc.): history just won't survive a refresh
  }
}

function badgeClass(source) {
  const s = (source || "").toLowerCase();
  if (s.includes("q&a")) return "qna";
  if (s.includes("device")) return "device";
  if (s.includes("web")) return "web";
  return "";
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function renderMessage(msg) {
  welcome.hidden = true;
  const wrap = el("div", `msg ${msg.role}${msg.error ? " error" : ""}`);
  wrap.appendChild(el("div", "bubble", msg.text));

  if (msg.role === "bot" && !msg.error) {
    const meta = el("div", "meta");
    meta.appendChild(el("span", `badge ${badgeClass(msg.source)}`, msg.source || "Unknown source"));
    wrap.appendChild(meta);

    if (msg.context && msg.context.trim()) {
      const details = el("details", "context");
      details.appendChild(el("summary", "", "Show retrieved context"));
      msg.context
        .split("\n")
        .map((p) => p.trim())
        .filter(Boolean)
        .forEach((p) => details.appendChild(el("p", "passage", p)));
      wrap.appendChild(details);
    }
  }

  chat.appendChild(wrap);
  scrollToBottom();
}

function showTyping() {
  const wrap = el("div", "msg bot");
  wrap.id = "typing";
  const bubble = el("div", "bubble");
  const typing = el("span", "typing");
  typing.append(el("span"), el("span"), el("span"), el("em", "", "Agent is thinking…"));
  bubble.appendChild(typing);
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  scrollToBottom();
}

function hideTyping() {
  document.getElementById("typing")?.remove();
}

function addMessage(msg) {
  history.push(msg);
  saveHistory();
  renderMessage(msg);
}

async function ask(query) {
  addMessage({ role: "user", text: query });
  sendBtn.disabled = true;
  input.disabled = true;
  showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail : `Request failed (${res.status}).`;
      throw new Error(detail);
    }
    hideTyping();
    addMessage({ role: "bot", text: data.answer || "(No answer returned.)", source: data.source, context: data.context });
  } catch (err) {
    hideTyping();
    addMessage({ role: "bot", error: true, text: `Sorry, something went wrong: ${err.message}` });
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function autoResize() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query || sendBtn.disabled) return;
  input.value = "";
  autoResize();
  ask(query);
});

// Enter sends, Shift+Enter adds a new line
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", autoResize);

document.querySelectorAll(".example").forEach((btn) =>
  btn.addEventListener("click", () => ask(btn.textContent))
);

clearBtn.addEventListener("click", () => {
  history = [];
  saveHistory();
  chat.querySelectorAll(".msg").forEach((m) => m.remove());
  welcome.hidden = false;
  input.focus();
});

history.forEach(renderMessage);
input.focus();

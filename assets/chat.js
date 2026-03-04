(function () {
  const transcriptEl = document.getElementById("transcript");
  const transcriptScrollEl = document.getElementById("transcript-scroll");
  let mathTimer = null;
  let lastTypesetTarget = null;

  function scheduleMathTypeset() {
    if (mathTimer) {
      clearTimeout(mathTimer);
      mathTimer = null;
    }
    mathTimer = setTimeout(() => {
      if (window.MathJax && window.MathJax.typesetPromise) {
        const target = lastTypesetTarget || transcriptEl;
        window.MathJax.typesetPromise([target]).catch(() => {});
      }
    }, 40);
  }

  function decodeRawB64(value) {
    try {
      const bin = atob(String(value || ""));
      const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
      return new TextDecoder("utf-8").decode(bytes);
    } catch (_err) {
      return "";
    }
  }

  function copyFallback(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "true");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (_err) {
      ok = false;
    }
    document.body.removeChild(ta);
    return ok;
  }

  async function copyTextToClipboard(text) {
    const payload = String(text || "");
    if (!payload) return false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(payload);
        return true;
      }
    } catch (_err) {
      return copyFallback(payload);
    }
    return copyFallback(payload);
  }

  function getRawMessageText(bodyNode) {
    if (!bodyNode) return "";
    const b64 = bodyNode.getAttribute("data-raw-b64");
    if (b64) {
      const decoded = decodeRawB64(b64);
      if (decoded) return decoded;
    }
    return bodyNode.textContent || "";
  }

  function decorateCodeBlocks(scopeNode) {
    const root = scopeNode || transcriptEl;
    if (!root) return;
    root.querySelectorAll("pre").forEach((pre) => {
      if (pre.querySelector(".copy-code-btn")) return;
      const code = pre.querySelector("code");
      if (!code) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-code-btn";
      btn.title = "Copy code";
      btn.setAttribute("aria-label", "Copy code");
      btn.textContent = "Copy";
      pre.insertBefore(btn, pre.firstChild);
    });
  }

  function scrollBottom() {
    if (!transcriptScrollEl) {
      window.scrollTo(0, document.body.scrollHeight);
      return;
    }
    transcriptScrollEl.scrollTop = transcriptScrollEl.scrollHeight;
  }

  function latestMathScope() {
    if (!transcriptEl || !transcriptEl.lastElementChild) {
      return transcriptEl;
    }
    const node = transcriptEl.lastElementChild.querySelector(".msg-body");
    return node || transcriptEl.lastElementChild;
  }

  function updateLatestAssistantBody(html, forceTypeset) {
    if (!transcriptEl) {
      return;
    }
    const bodies = transcriptEl.querySelectorAll(".msg.assistant .msg-body");
    if (!bodies.length) {
      return;
    }
    const body = bodies[bodies.length - 1];
    const nextHtml = html || "";
    const unchanged = body.innerHTML === nextHtml;
    if (!unchanged) {
      body.innerHTML = nextHtml;
      decorateCodeBlocks(body);
    }
    lastTypesetTarget = body;
    // Avoid LaTeX flicker during streaming: only typeset on explicit finalization.
    if (forceTypeset) {
      scheduleMathTypeset();
    }
    if (!unchanged || forceTypeset) {
      scrollBottom();
    }
  }

  window.AthenaUI = {
    setTranscriptHtml(html) {
      transcriptEl.innerHTML = html || "";
      decorateCodeBlocks(transcriptEl);
      lastTypesetTarget = latestMathScope();
      scheduleMathTypeset();
      scrollBottom();
    },
    updateLatestAssistantBody(html, forceTypeset) {
      updateLatestAssistantBody(html, !!forceTypeset);
    },
    notifyMathjaxMissing() {
      // Non-fatal: keep plain text/markdown display if MathJax bundle missing.
      console.warn("MathJax assets not found. Rendering without TeX typesetting.");
    },
  };

  // Probe after startup to detect missing local bundle.
  setTimeout(() => {
    if (!window.MathJax || !window.MathJax.typesetPromise) {
      if (window.AthenaUI && window.AthenaUI.notifyMathjaxMissing) {
        window.AthenaUI.notifyMathjaxMissing();
      }
    }
  }, 500);

  transcriptEl.addEventListener("click", async (event) => {
    const target = event.target;
    if (!target || !target.classList) return;

    if (target.classList.contains("copy-msg-btn")) {
      const msg = target.closest(".msg");
      const body = msg ? msg.querySelector(".msg-body") : null;
      const ok = await copyTextToClipboard(getRawMessageText(body));
      console.log(ok ? "message copied" : "copy failed");
      return;
    }

    if (target.classList.contains("copy-code-btn")) {
      const pre = target.closest("pre");
      const code = pre ? pre.querySelector("code") : null;
      const ok = await copyTextToClipboard(code ? code.textContent || "" : "");
      console.log(ok ? "code copied" : "copy failed");
    }
  });
})();

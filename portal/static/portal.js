(function () {
  const runtimeLine = document.getElementById("runtime-line");
  const userLine = document.getElementById("user-line");
  const logoutBtn = document.getElementById("logout-btn");
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-btn");
  const attachBtn = document.getElementById("attach-btn");
  const clearImagesBtn = document.getElementById("clear-images-btn");
  const imageInput = document.getElementById("image-input");
  const imageLine = document.getElementById("image-line");
  const imagePreview = document.getElementById("image-preview");
  const promptInput = document.getElementById("prompt-input");
  const statusBox = document.getElementById("status-box");
  const transcriptEl = document.getElementById("transcript");
  const transcriptScrollEl = document.getElementById("transcript-scroll");

  const state = {
    pathPrefix: (window.ATHENA_PORTAL && window.ATHENA_PORTAL.pathPrefix) || "/AthenaV5",
    history: [],
    config: null,
    me: null,
    busy: false,
    pendingImages: [],
  };

  function setStatus(text) {
    statusBox.textContent = text;
  }

  function decodeRawB64(value) {
    try {
      const bin = atob(String(value || ""));
      const bytes = Uint8Array.from(bin, function (c) {
        return c.charCodeAt(0);
      });
      return new TextDecoder("utf-8").decode(bytes);
    } catch (err) {
      return "";
    }
  }

  async function copyTextToClipboard(text, okMessage) {
    const payload = String(text || "");
    if (!payload) {
      setStatus("Nothing to copy.");
      return;
    }
    try {
      await navigator.clipboard.writeText(payload);
      setStatus(okMessage || "Copied.");
    } catch (err) {
      setStatus("Copy failed: " + err.message);
    }
  }

  function getRawMessageText(bodyNode) {
    if (!bodyNode) return "";
    const rawB64 = bodyNode.getAttribute("data-raw-b64");
    if (rawB64) {
      const decoded = decodeRawB64(rawB64);
      if (decoded) return decoded;
    }
    const datasetRaw = bodyNode.dataset ? bodyNode.dataset.rawText : "";
    if (datasetRaw) return datasetRaw;
    return bodyNode.textContent || "";
  }

  function decorateCodeBlocks(scopeNode) {
    const root = scopeNode || transcriptEl;
    if (!root) return;
    root.querySelectorAll("pre").forEach(function (pre) {
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

  function fileToDataUrl(file) {
    return new Promise(function (resolve, reject) {
      const reader = new FileReader();
      reader.onload = function () {
        resolve(String(reader.result || ""));
      };
      reader.onerror = function () {
        reject(new Error("Failed to read image: " + (file && file.name ? file.name : "unknown")));
      };
      reader.readAsDataURL(file);
    });
  }

  function renderPendingImages() {
    if (!imageLine || !imagePreview) return;
    imagePreview.innerHTML = "";
    const count = state.pendingImages.length;
    imageLine.textContent =
      count > 0
        ? "Images attached: " + count + " (" + state.pendingImages.map(function (x) { return x.name; }).join(", ") + ")"
        : "Images: none";
    if (clearImagesBtn) clearImagesBtn.disabled = count === 0 || state.busy;
    if (count === 0) return;
    state.pendingImages.forEach(function (item) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.loading = "lazy";
      img.decoding = "async";
      img.alt = item.name || "attached image";
      img.src = item.data_url;
      imagePreview.appendChild(img);
    });
  }

  async function addImageFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    const maxImages = 6;
    for (const file of files) {
      if (state.pendingImages.length >= maxImages) {
        setStatus("Image limit reached (" + maxImages + " max per request).");
        break;
      }
      if (!file.type || !file.type.startsWith("image/")) continue;
      if (file.size > 8 * 1024 * 1024) {
        setStatus("Skipped " + file.name + " (over 8MB).");
        continue;
      }
      const dataUrl = await fileToDataUrl(file);
      state.pendingImages.push({
        name: file.name || "image.png",
        content_type: file.type || "image/png",
        data_url: dataUrl,
      });
    }
    renderPendingImages();
  }

  function clearPendingImages() {
    state.pendingImages = [];
    if (imageInput) imageInput.value = "";
    renderPendingImages();
  }

  function scrollBottom() {
    transcriptScrollEl.scrollTop = transcriptScrollEl.scrollHeight;
  }

  function typesetMath() {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise([transcriptEl]).catch(function () {
        return;
      });
    }
  }

  function applyTranscriptHtml(html) {
    transcriptEl.innerHTML = html || "";
    decorateCodeBlocks(transcriptEl);
    scrollBottom();
    typesetMath();
  }

  function roleLabel(role) {
    if (role === "user") return "User";
    if (role === "assistant") return "Athena";
    if (role === "system") return "System";
    return "Message";
  }

  function roleIcon(role) {
    if (role === "user") return "\u{1F9D1}";
    if (role === "assistant") return "\u{1F9E0}";
    if (role === "system") return "\u2699\uFE0F";
    return "\u{1F4AC}";
  }

  function makeLiveMessage(role, text, imageUrls) {
    const article = document.createElement("article");
    article.className = "msg " + role;

    const avatar = document.createElement("aside");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = roleIcon(role);

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const head = document.createElement("div");
    head.className = "bubble-head";

    const pill = document.createElement("span");
    pill.className = "role-pill";
    const icon = document.createElement("span");
    icon.className = "role-icon";
    icon.textContent = roleIcon(role);
    pill.appendChild(icon);
    pill.appendChild(document.createTextNode(roleLabel(role)));
    head.appendChild(pill);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "copy-msg-btn";
    copyBtn.title = "Copy raw message";
    copyBtn.setAttribute("aria-label", "Copy raw message");
    copyBtn.textContent = "Copy";
    head.appendChild(copyBtn);

    const body = document.createElement("section");
    body.className = "msg-body";
    body.style.whiteSpace = "pre-wrap";
    body.textContent = text || "";
    body.dataset.rawText = text || "";
    if (Array.isArray(imageUrls) && imageUrls.length > 0) {
      imageUrls.forEach(function (url, idx) {
        if (!url) return;
        const img = document.createElement("img");
        img.loading = "lazy";
        img.decoding = "async";
        img.alt = "attached image " + String(idx + 1);
        img.src = url;
        body.appendChild(document.createElement("br"));
        body.appendChild(img);
      });
    }

    bubble.appendChild(head);
    bubble.appendChild(body);
    article.appendChild(avatar);
    article.appendChild(bubble);
    return { article, body };
  }

  function setBusy(flag) {
    state.busy = !!flag;
    sendBtn.disabled = state.busy;
    clearBtn.disabled = state.busy;
    promptInput.disabled = state.busy;
    if (attachBtn) attachBtn.disabled = state.busy;
    if (imageInput) imageInput.disabled = state.busy;
    if (clearImagesBtn) clearImagesBtn.disabled = state.busy || state.pendingImages.length === 0;
  }

  function autosizePrompt() {
    const maxPx = 160;
    promptInput.style.height = "auto";
    const next = Math.max(44, Math.min(promptInput.scrollHeight, maxPx));
    promptInput.style.height = String(next) + "px";
  }

  async function apiGet(path) {
    const res = await fetch(state.pathPrefix + path, { credentials: "same-origin" });
    if (!res.ok) {
      const err = new Error("GET " + path + " failed: " + res.status);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  async function apiPost(path, body) {
    const res = await fetch(state.pathPrefix + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body || {}),
    });
    const payload = await res.json().catch(function () {
      return {};
    });
    if (!res.ok) {
      const message = (payload && payload.detail) || "Request failed.";
      const err = new Error(String(message));
      err.status = res.status;
      throw err;
    }
    return payload;
  }

  function handleAuthFailure() {
    setStatus("Authentication required. Redirecting to sign-in page...");
    window.location.href = state.pathPrefix + "/login";
  }

  async function streamChat(prompt, history, images, handlers) {
    const res = await fetch(state.pathPrefix + "/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ prompt: prompt, history: history || [], images: images || [] }),
    });
    if (res.status === 401) {
      handleAuthFailure();
      return;
    }
    if (!res.ok) {
      let message = "Request failed.";
      try {
        const payload = await res.json();
        message = (payload && payload.detail) || message;
      } catch (err) {
        message = "Request failed: " + res.status;
      }
      throw new Error(message);
    }
    if (!res.body) throw new Error("Streaming response body not available.");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    function processEventBlock(block) {
      const raw = (block || "").trim();
      if (!raw) return;
      const dataLines = raw
        .split(/\r?\n/)
        .map(function (line) {
          return line.trim();
        })
        .filter(function (line) {
          return line.startsWith("data:");
        })
        .map(function (line) {
          return line.slice(5).trim();
        });
      if (dataLines.length === 0) return;
      const evt = JSON.parse(dataLines.join("\n"));
      if (evt.type === "delta") {
        handlers.onDelta(evt.text || "");
        return;
      }
      if (evt.type === "done") {
        handlers.onDone(evt);
        return;
      }
      if (evt.type === "error") {
        throw new Error(evt.message || "Streaming failed.");
      }
    }

    while (true) {
      const step = await reader.read();
      if (step.done) break;
      buffer += decoder.decode(step.value, { stream: true });
      let sep = buffer.search(/\r?\n\r?\n/);
      while (sep >= 0) {
        const block = buffer.slice(0, sep);
        const sepLen = buffer.slice(sep, sep + 2) === "\r\n" ? 4 : 2;
        buffer = buffer.slice(sep + sepLen);
        processEventBlock(block);
        sep = buffer.search(/\r?\n\r?\n/);
      }
    }
    if (buffer.trim().length > 0) {
      processEventBlock(buffer);
    }
  }

  async function sendMessage() {
    const prompt = (promptInput.value || "").trim();
    const outgoingImages = state.pendingImages.slice();
    if ((!prompt && outgoingImages.length === 0) || state.busy) return;

    setBusy(true);
    promptInput.value = "";
    autosizePrompt();
    clearPendingImages();

    const userDisplay = prompt || "Image attached.";
    const imageDataUrls = outgoingImages.map(function (x) {
      return x.data_url;
    });

    const userLive = makeLiveMessage("user", userDisplay, imageDataUrls);
    transcriptEl.appendChild(userLive.article);
    const assistantLive = makeLiveMessage("assistant", "");
    transcriptEl.appendChild(assistantLive.article);
    scrollBottom();

    setStatus("Request sent. Waiting for first token...");

    let firstTokenTimer = null;
    let stallInterval = null;
    let lastDeltaAt = Date.now();
    try {
      let sawDelta = false;
      let donePayload = null;

      firstTokenTimer = setTimeout(function () {
        if (!sawDelta) {
          setStatus("Model is loading / preparing first token...");
        }
      }, 3000);

      stallInterval = setInterval(function () {
        if (state.busy && sawDelta && Date.now() - lastDeltaAt > 10000) {
          setStatus("Stream stalled. Waiting for more tokens...");
        }
      }, 1500);

      await streamChat(prompt, state.history, outgoingImages, {
        onDelta: function (txt) {
          sawDelta = true;
          lastDeltaAt = Date.now();
          assistantLive.body.dataset.rawText = (assistantLive.body.dataset.rawText || "") + txt;
          assistantLive.body.textContent += txt;
          scrollBottom();
        },
        onDone: function (evt) {
          donePayload = evt;
        },
      });

      if (!donePayload) {
        throw new Error("Stream disconnected before completion.");
      }
      state.history = donePayload.history || [];
      applyTranscriptHtml(donePayload.transcript_html || "");
      const modeText = donePayload.smoke_mode
        ? "Smoke mode response received."
        : donePayload.model_loaded
          ? "Model response received."
          : "Response received.";
      setStatus(modeText);
    } catch (err) {
      assistantLive.body.textContent = "Request failed.";
      setStatus("Request failed: " + err.message);
    } finally {
      if (firstTokenTimer) clearTimeout(firstTokenTimer);
      if (stallInterval) clearInterval(stallInterval);
      setBusy(false);
      promptInput.focus();
    }
  }

  function clearConversation() {
    state.history = [];
    clearPendingImages();
    applyTranscriptHtml("");
    setStatus("Conversation cleared.");
  }

  async function bootstrap() {
    try {
      state.config = await apiGet("/api/config");
      const smoke = state.config.smoke_mode ? "on" : "off";
      const loaded = state.config.model_loaded ? "loaded" : "not loaded";
      const configuredLabel = state.config.configured_model_label || "unknown";
      const activeLabel = state.config.active_model_label || "none";
      const activeText = state.config.model_loaded ? activeLabel : "not loaded";
      runtimeLine.textContent =
        "Path " +
        state.config.path_prefix +
        " | smoke " +
        smoke +
        " | model " +
        loaded +
        " | configured " +
        configuredLabel +
        " | active " +
        activeText;
      runtimeLine.title =
        "configured=" +
        (state.config.configured_model_dir || "") +
        (state.config.active_model_dir ? "\nactive=" + state.config.active_model_dir : "");

      if (state.config.auth_required) {
        try {
          const me = await apiGet("/api/me");
          state.me = me.user || null;
        } catch (err) {
          if (err.status === 401) {
            handleAuthFailure();
            return;
          }
          throw err;
        }
      } else {
        logoutBtn.style.display = "none";
      }

      if (state.me) {
        const display = state.me.name || state.me.email || "signed in";
        userLine.textContent = "User: " + display;
      } else {
        userLine.textContent = "User: anonymous";
      }

      if (state.config.model_load_error) {
        setStatus("Model load warning: " + state.config.model_load_error);
      } else if (state.config.smoke_mode) {
        setStatus("Smoke mode active. Configured model: " + configuredLabel);
      } else if (state.config.model_loaded) {
        setStatus("Ready. Active model: " + activeLabel);
      } else {
        setStatus("Ready. Model not loaded yet. Configured model: " + configuredLabel);
      }
    } catch (err) {
      runtimeLine.textContent = "Failed to load portal config.";
      setStatus("Bootstrap failed: " + err.message);
      return;
    }
    sendBtn.disabled = false;
    typesetMath();
  }

  async function logout() {
    try {
      await apiPost("/auth/logout", {});
    } catch (err) {
      // continue redirect either way
    }
    window.location.href = state.pathPrefix + "/login";
  }

  sendBtn.addEventListener("click", sendMessage);
  clearBtn.addEventListener("click", clearConversation);
  logoutBtn.addEventListener("click", logout);
  transcriptEl.addEventListener("click", function (event) {
    const target = event.target;
    if (!target || !target.classList) return;
    if (target.classList.contains("copy-msg-btn")) {
      const msg = target.closest(".msg");
      const body = msg ? msg.querySelector(".msg-body") : null;
      copyTextToClipboard(getRawMessageText(body), "Message copied.");
      return;
    }
    if (target.classList.contains("copy-code-btn")) {
      const pre = target.closest("pre");
      const code = pre ? pre.querySelector("code") : null;
      copyTextToClipboard(code ? code.textContent || "" : "", "Code copied.");
    }
  });
  if (attachBtn && imageInput) {
    attachBtn.addEventListener("click", function () {
      if (state.busy) return;
      imageInput.click();
    });
    imageInput.addEventListener("change", async function (event) {
      const files = event.target && event.target.files ? event.target.files : [];
      try {
        await addImageFiles(files);
      } catch (err) {
        setStatus("Image attach failed: " + err.message);
      } finally {
        imageInput.value = "";
      }
    });
  }
  if (clearImagesBtn) {
    clearImagesBtn.addEventListener("click", function () {
      clearPendingImages();
      setStatus("Pending images cleared.");
    });
  }
  promptInput.addEventListener("keydown", function (event) {
    if (event.isComposing) return;
    if ((event.key === "Enter" || event.key === "NumpadEnter") && !event.shiftKey) {
      event.preventDefault();
      if (!state.busy) sendMessage();
    }
  });
  promptInput.addEventListener("paste", async function (event) {
    if (!event.clipboardData || state.busy) return;
    const files = [];
    const items = Array.from(event.clipboardData.items || []);
    items.forEach(function (item) {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    });
    if (files.length === 0) return;
    event.preventDefault();
    try {
      await addImageFiles(files);
      setStatus("Image pasted from clipboard.");
    } catch (err) {
      setStatus("Clipboard image paste failed: " + err.message);
    }
  });
  promptInput.addEventListener("input", autosizePrompt);

  bootstrap();
  autosizePrompt();
  renderPendingImages();
  decorateCodeBlocks(transcriptEl);
})();

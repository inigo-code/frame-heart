(function () {
  const logLines = [];
  let modelKey = "";
  let scaleKey = "";

  function toast(item) {
    const node = document.createElement("div");
    node.className = "toast" + (item.kind ? " " + item.kind : "");
    node.textContent = item.text || "";
    document.getElementById("toasts").appendChild(node);
    setTimeout(function () {
      node.classList.add("out");
      setTimeout(function () { node.remove(); }, 220);
    }, 4200);
  }

  function appendLog(text) {
    logLines.push(String(text));
    if (logLines.length > 400) logLines.shift();
    const pre = document.getElementById("log");
    pre.textContent = logLines.join("\n");
    pre.scrollTop = pre.scrollHeight;
  }

  function setLogOpen(open) {
    const stage = document.getElementById("stage");
    const card = document.getElementById("log-card");
    const button = document.getElementById("log-toggle");
    stage.classList.toggle("is-log-collapsed", !open);
    card.classList.toggle("is-collapsed", !open);
    button.setAttribute("aria-expanded", open ? "true" : "false");
    document.getElementById("log-toggle-label").textContent = open ? "Hide" : "Show";
    try {
      localStorage.setItem("frameheart-log-v2", open ? "open" : "closed");
    } catch (error) {}
    if (open) {
      const pre = document.getElementById("log");
      pre.scrollTop = pre.scrollHeight;
    }
  }

  function setField(id, value) {
    const node = document.getElementById(id);
    if (document.activeElement === node) return;
    if (node.value !== value) node.value = value;
  }

  function renderModels(state) {
    const host = document.getElementById("models");
    const left = host.scrollLeft;
    host.replaceChildren();
    (state.models || []).forEach(function (entry) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "model" + (entry.id === state.model ? " is-on" : "");
      button.dataset.id = entry.id;
      const title = document.createElement("strong");
      title.textContent = entry.title;
      const badge = document.createElement("em");
      badge.textContent = entry.badge;
      if (entry.badge === "Ready" || entry.badge === "Custom") badge.className = "ok";
      const detail = document.createElement("span");
      detail.textContent = entry.detail;
      button.append(title, badge, detail);
      host.appendChild(button);
    });
    host.scrollLeft = left;
  }

  function renderScales(state) {
    const host = document.getElementById("scales");
    host.replaceChildren();
    (state.scales || []).forEach(function (scale) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.scale = String(scale);
      button.textContent = scale + "\u00d7";
      if (Number(scale) === Number(state.scale)) button.classList.add("is-on");
      host.appendChild(button);
    });
  }

  function countLabel(state) {
    if (state.selection) return state.selection;
    const images = Number(state.imageCount || 0);
    const videos = Number(state.videoCount || 0);
    if (!images && !videos) return "0 files";
    const parts = [];
    if (images) parts.push(images + (images === 1 ? " image" : " images"));
    if (videos) parts.push(videos + (videos === 1 ? " video" : " videos"));
    return parts.join(", ");
  }

  function videoNote(state) {
    const videos = Number(state.videoCount || 0);
    if (!videos || state.task === "settings" || state.task === "queue") return "";
    if (state.task === "upscale") return "Videos are not upscaled.";
    if (state.combine) return "Videos are cleaned, not enlarged.";
    return "";
  }

  function paintProgress(ratio, label) {
    const width = Math.max(0, Math.min(100, (Number(ratio) || 0) * 100));
    document.getElementById("bar").style.width = width + "%";
    document.getElementById("progress-pct").textContent = Math.round(width) + "%";
    const node = document.getElementById("progress-label");
    const text = label || "Ready";
    node.textContent = text;
    node.title = text;
  }

  function paintCancel(button, pct) {
    button.hidden = false;
    button.textContent = "Cancel" + (pct == null ? "" : " " + pct + "%");
    button.className = "btn danger";
    button.disabled = false;
  }

  function paintFfmpegAsk(state) {
    const ask = document.getElementById("ffmpeg-ask");
    if (ask.hidden) return;
    if (state.ffmpegReady) {
      ask.hidden = true;
      return;
    }
    const busy = state.downloadKind === "ffmpeg" && state.downloading;
    const pct = state.downloadPct;
    document.getElementById("ffmpeg-ask-copy").textContent = busy
      ? "Installing FFmpeg" + (pct == null ? "..." : " · " + pct + "%")
      : "Frame Heart uses FFmpeg to read and write video. Install it now? It's a free download of about 115 MB, and it stays beside the app.";
    document.getElementById("ffmpeg-ask-get").hidden = busy;
    document.getElementById("ffmpeg-ask-later").textContent = busy ? "Cancel" : "Not now";
    document.getElementById("ffmpeg-ask-later").dataset.busy = busy ? "1" : "";
  }

  function render(state) {
    if (!state || !state.task) return;
    const settings = state.task === "settings";
    const queue = state.task === "queue";
    document.body.dataset.task = state.task;
    document.getElementById("intro").hidden = !!state.seenIntro;
    paintFfmpegAsk(state);
    document.getElementById("title").textContent = state.title;
    if (state.version) document.getElementById("app-version").textContent = "v" + state.version;
    document.querySelectorAll(".tab").forEach(function (button) {
      button.classList.toggle("is-on", button.dataset.task === state.task);
    });
    document.querySelectorAll("#processor button").forEach(function (button) {
      button.classList.toggle("is-on", button.dataset.processor === (state.processor || "gpu"));
    });
    document.getElementById("settings").hidden = !settings;
    document.getElementById("queue-card").hidden = !queue;
    document.getElementById("job-card").hidden = settings || queue;
    document.getElementById("stage").hidden = settings || queue;
    document.getElementById("inspector").hidden = settings || queue;
    document.getElementById("preview").hidden = settings || queue;
    document.getElementById("start").hidden = settings;
    document.getElementById("model-card").hidden = settings || !state.showModels;

    const picturesOnly = state.task === "upscale";
    document.getElementById("input").placeholder = picturesOnly ? "Folder of images" : "Folder of images or videos";
    document.getElementById("drop-layer").textContent = picturesOnly ? "Drop images or a folder" : "Drop images, videos, or a folder";
    setField("input", state.input || "");
    setField("output", state.output || "");
    const recursive = document.getElementById("recursive");
    if (document.activeElement !== recursive) recursive.checked = !!state.recursive;
    const quality = document.getElementById("quality");
    if (document.activeElement !== quality) quality.value = String(state.quality);
    document.getElementById("quality-value").textContent = String(state.quality);
    const crf = document.getElementById("video-crf");
    if (document.activeElement !== crf) crf.value = String(state.videoCrf == null ? 18 : state.videoCrf);
    document.getElementById("crf-value").textContent = String(state.videoCrf == null ? 18 : state.videoCrf);
    const audio = document.getElementById("keep-audio");
    if (document.activeElement !== audio) audio.checked = state.keepAudio !== false;
    const combine = document.getElementById("combine");
    document.getElementById("combine-row").hidden = state.task !== "clean";
    if (document.activeElement !== combine) combine.checked = !!state.combine;
    const count = document.getElementById("count");
    count.textContent = countLabel(state);
    count.title = state.selectionPath || "";
    const videoNoteNode = document.getElementById("video-note");
    const videoNoteText = videoNote(state);
    videoNoteNode.hidden = !videoNoteText;
    videoNoteNode.textContent = videoNoteText;
    const selectedName = state.selection || "";
    const selectedRow = document.getElementById("selection-row");
    selectedRow.hidden = !selectedName;
    const selectionNode = document.getElementById("selection");
    selectionNode.textContent = selectedName;
    selectionNode.title = state.selectionPath || selectedName;
    document.getElementById("ffmpeg-note").textContent = state.ffmpegNote || "";
    const ffmpeg = document.getElementById("download-ffmpeg");
    if (state.downloadKind === "ffmpeg" && state.downloading) {
      paintCancel(ffmpeg, state.downloadPct);
    } else if (state.ffmpegReady) {
      ffmpeg.hidden = true;
    } else {
      ffmpeg.hidden = false;
      ffmpeg.textContent = "Download FFmpeg";
      ffmpeg.className = "btn primary";
      ffmpeg.disabled = !!state.downloading;
    }

    const key = (state.model || "") + "|" + (state.models || []).map(function (entry) {
      return entry.id + ":" + entry.badge + ":" + entry.detail;
    }).join(",");
    if (key !== modelKey) {
      modelKey = key;
      renderModels(state);
    } else {
      document.querySelectorAll(".model").forEach(function (button) {
        button.classList.toggle("is-on", button.dataset.id === state.model);
      });
    }
    const scales = (state.scales || []).join(",") + ":" + state.scale;
    if (scales !== scaleKey) {
      scaleKey = scales;
      renderScales(state);
    }

    const selected = (state.models || []).find(function (entry) { return entry.id === state.model; });
    const download = document.getElementById("download");
    if (state.downloadKind === "model" && state.downloading) {
      paintCancel(download, state.downloadPct);
    } else if (selected && !selected.installed && selected.kind === "download") {
      download.hidden = false;
      download.textContent = "Download";
      download.className = "btn primary";
    } else {
      download.hidden = true;
    }

    const start = document.getElementById("start");
    start.textContent = state.action;
    start.disabled = !!state.stopping;
    start.className = state.running ? "btn danger" : "btn primary";

    const device = document.getElementById("device");
    device.hidden = !state.device;
    device.textContent = state.device || "";
    device.title = state.device || "";

    paintProgress(state.progress || 0, state.progressLabel || "Ready");
    renderQueue(state);
  }

  function renderQueue(state) {
    const host = document.getElementById("queue");
    if (!host) return;
    const items = state.queue || [];
    // Rebuilding a list that hasn't changed loses a click in progress.
    const sig = JSON.stringify([state.running, items.map(function (item) { return [item.id, item.status, item.detail]; })]);
    if (host.dataset.sig === sig) return;
    host.dataset.sig = sig;
    const top = host.scrollTop;
    host.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "queue-empty";
      empty.textContent = "No files in the queue.";
      host.appendChild(empty);
    }
    const labels = {
      waiting: "Waiting",
      running: "Running",
      removed: "Cleaned",
      upscaled: "Upscaled",
      unchanged: "No mark",
      failed: "Failed",
      stopped: "Stopped"
    };
    items.forEach(function (item) {
      const row = document.createElement("div");
      row.className = "queue-row";
      row.dataset.id = item.id || "";
      const name = document.createElement("span");
      name.className = "queue-name";
      name.textContent = item.name || "";
      name.title = item.detail ? item.name + " — " + item.detail : (item.id || item.name || "");
      const status = document.createElement("span");
      status.className = "queue-status " + (item.status || "waiting");
      status.textContent = labels[item.status] || item.status || "Waiting";
      // One button a row. It stops a running file, or takes a file off the list.
      const action = document.createElement("button");
      action.type = "button";
      action.className = "btn";
      const running = item.status === "running";
      action.dataset.act = running ? "cancel" : "remove";
      action.textContent = running ? "Cancel" : "Take off list";
      row.append(name, status, action);
      host.appendChild(row);
    });
    host.scrollTop = top;
    const more = document.getElementById("queue-more");
    const extra = Number(state.queueMore || 0);
    more.hidden = extra <= 0;
    more.textContent = extra > 0 ? extra + " more are not listed. Clear list still takes them off." : "";
  }

  function call(name) {
    const args = Array.prototype.slice.call(arguments, 1);
    if (!window.pywebview || !window.pywebview.api) return;
    window.pywebview.api[name].apply(null, args).then(function (state) {
      render(state);
    }).catch(function () {
      toast({ kind: "critical", text: "That action failed." });
    });
  }

  function bind() {
    document.querySelectorAll(".tab").forEach(function (button) {
      button.addEventListener("click", function () { call("set_task", button.dataset.task); });
    });
    const split = document.getElementById("compare-range");
    split.addEventListener("input", function () {
      document.getElementById("compare").style.setProperty("--pos", split.value + "%");
    });
    document.getElementById("scales").addEventListener("click", function (event) {
      const button = event.target.closest("button");
      if (!button) return;
      call("set_scale", Number(button.dataset.scale));
    });
    document.getElementById("models").addEventListener("click", function (event) {
      const button = event.target.closest(".model");
      if (!button) return;
      call("set_model", button.dataset.id);
    });
    document.getElementById("browse-input").addEventListener("click", function () { call("browse_input"); });
    document.getElementById("browse-output").addEventListener("click", function () { call("browse_output"); });
    document.getElementById("add-files").addEventListener("click", function () { call("add_files"); });
    document.getElementById("add-model").addEventListener("click", function () { call("add_model"); });
    document.getElementById("download").addEventListener("click", function () {
      const button = document.getElementById("download");
      call(button.classList.contains("danger") ? "cancel_download" : "download");
    });
    document.getElementById("preview").addEventListener("click", function () { call("preview"); });
    document.getElementById("start").addEventListener("click", function () { call("toggle"); });
    document.getElementById("open-folder").addEventListener("click", function () { call("open_output"); });
    document.getElementById("input").addEventListener("change", function (event) {
      call("set_input", event.target.value);
    });
    document.getElementById("output").addEventListener("change", function (event) {
      call("set_output", event.target.value);
    });
    document.getElementById("recursive").addEventListener("change", function (event) {
      call("set_recursive", event.target.checked);
    });
    document.getElementById("quality").addEventListener("input", function (event) {
      document.getElementById("quality-value").textContent = event.target.value;
    });
    document.getElementById("quality").addEventListener("change", function (event) {
      call("set_quality", Number(event.target.value));
    });
    document.getElementById("video-crf").addEventListener("input", function (event) {
      document.getElementById("crf-value").textContent = event.target.value;
    });
    document.getElementById("video-crf").addEventListener("change", function (event) {
      call("set_video_crf", Number(event.target.value));
    });
    document.getElementById("keep-audio").addEventListener("change", function (event) {
      call("set_keep_audio", event.target.checked);
    });
    document.getElementById("combine").addEventListener("change", function (event) {
      call("set_combine", event.target.checked);
    });
    document.getElementById("processor").addEventListener("click", function (event) {
      const button = event.target.closest("button");
      if (!button) return;
      call("set_processor", button.dataset.processor);
    });
    document.getElementById("download-ffmpeg").addEventListener("click", function () {
      const button = document.getElementById("download-ffmpeg");
      call(button.classList.contains("danger") ? "cancel_download" : "download_ffmpeg");
    });
    document.getElementById("queue").addEventListener("click", function (event) {
      const button = event.target.closest("button");
      if (!button) return;
      const row = button.closest(".queue-row");
      if (!row) return;
      call(button.dataset.act === "cancel" ? "queue_cancel" : "queue_remove", row.dataset.id || "");
    });
    document.getElementById("queue-clear").addEventListener("click", function () { call("queue_clear_finished"); });
    document.getElementById("queue-remove-all").addEventListener("click", function () { call("queue_remove_all"); });
    document.getElementById("ffmpeg-ask-get").addEventListener("click", function () {
      document.getElementById("ffmpeg-ask-pictures").hidden = true;
      call("get_ffmpeg_and_run");
    });
    document.getElementById("ffmpeg-ask-pictures").addEventListener("click", function () {
      document.getElementById("ffmpeg-ask").hidden = true;
      call("run_pictures_only");
    });
    document.getElementById("ffmpeg-ask-later").addEventListener("click", function () {
      if (this.dataset.busy) {
        call("cancel_download");
        return;
      }
      document.getElementById("ffmpeg-ask").hidden = true;
    });
    document.getElementById("intro-continue").addEventListener("click", function () {
      document.getElementById("intro").hidden = true;
      call("accept_intro");
    });
    let dragDepth = 0;
    function hasFiles(event) {
      const types = event.dataTransfer && event.dataTransfer.types;
      if (!types) return false;
      for (let i = 0; i < types.length; i += 1) {
        if (types[i] === "Files") return true;
      }
      return false;
    }
    document.addEventListener("dragenter", function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth += 1;
      document.body.classList.add("is-dropping");
    });
    document.addEventListener("dragover", function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    });
    document.addEventListener("dragleave", function (event) {
      if (!hasFiles(event)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) document.body.classList.remove("is-dropping");
    });
    document.addEventListener("drop", function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth = 0;
      document.body.classList.remove("is-dropping");
    });
    document.getElementById("log-toggle").addEventListener("click", function () {
      const open = document.getElementById("log-toggle").getAttribute("aria-expanded") === "true";
      setLogOpen(!open);
    });
    var stored = "closed";
    try {
      stored = localStorage.getItem("frameheart-log-v2") || "closed";
    } catch (error) {}
    if (stored === "open") setLogOpen(true);
  }

  window.app = {
    onEvent: function (message) {
      if (!message) return;
      if (message.name === "state") render(message.data);
      else if (message.name === "log") appendLog(message.data);
      else if (message.name === "preview") {
        const data = message.data || {};
        const has = !!data.before;
        const before = document.getElementById("before");
        const after = document.getElementById("after");
        const compare = document.getElementById("compare");
        before.src = has ? data.before : "";
        after.src = has ? data.after : "";
        compare.classList.toggle("is-empty", !has);
        const fit = function () {
          if (has && after.naturalWidth && after.naturalHeight) {
            compare.style.aspectRatio = after.naturalWidth + " / " + after.naturalHeight;
          } else {
            compare.style.aspectRatio = "";
          }
        };
        if (!has || (after.complete && after.naturalWidth)) fit();
        else after.addEventListener("load", fit, { once: true });
        if (!has) {
          compare.style.setProperty("--pos", "50%");
          document.getElementById("compare-range").value = "50";
        }
        const caption = document.getElementById("caption");
        caption.textContent = has ? (data.caption || "") : "";
        caption.hidden = !has || !data.caption;
      } else if (message.name === "toast") toast(message.data);
      else if (message.name === "ffmpegNeeded") {
        document.getElementById("ffmpeg-ask-pictures").hidden = !(message.data && message.data.pictures);
        document.getElementById("ffmpeg-ask-get").hidden = false;
        document.getElementById("ffmpeg-ask").hidden = false;
      } else if (message.name === "download") {
        const ffmpeg = message.data && message.data.kind === "ffmpeg";
        const button = document.getElementById(ffmpeg ? "download-ffmpeg" : "download");
        paintCancel(button, message.data.pct);
        if (ffmpeg && !document.getElementById("ffmpeg-ask").hidden) {
          paintFfmpegAsk({ downloadKind: "ffmpeg", downloading: true, downloadPct: message.data.pct, ffmpegReady: false });
        }
      } else if (message.name === "progress") {
        const data = message.data || {};
        if (typeof data.ratio === "number") {
          paintProgress(data.ratio, data.label || "");
        } else {
          const done = data.done || 0;
          const total = data.total || 0;
          paintProgress(total ? done / total : 0, data.label || (total ? done + " of " + total : "Ready"));
        }
      }
    },
  };

  bind();
  window.addEventListener("pywebviewready", function () {
    window.pywebview.api.bootstrap().then(function (state) {
      render(state);
      return window.pywebview.api.ui_ready();
    }).catch(function () {
      toast({ kind: "critical", text: "The window did not finish opening." });
    });
  });
})();

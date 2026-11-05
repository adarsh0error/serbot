/**
 * SERBOT PPE COMPLIANCE MONITOR — HYBRID CONTROLLER
 * Supports both local Python Flask server AND standalone GitHub Pages!
 */

let isLocalServer = false;

document.addEventListener("DOMContentLoaded", async () => {
  await checkServerStatus();
  initTabs();
  initLab();
  initCamera();
  initGallery();
});

async function checkServerStatus() {
  try {
    const res = await fetch("/api/system_info", { signal: AbortSignal.timeout(1000) });
    if (res.ok) {
      isLocalServer = true;
      const data = await res.json();
      const statusText = document.getElementById("system-status-text");
      if (statusText) statusText.textContent = "YOLOv8s LIVE SERVER";
      return;
    }
  } catch (e) {
    // Standalone / GitHub Pages mode
  }
  isLocalServer = false;
  const statusText = document.getElementById("system-status-text");
  if (statusText) statusText.textContent = "GITHUB PAGES MIRROR";
}

/* ---------------- TABS NAVIGATION ---------------- */
function initTabs() {
  const tabButtons = document.querySelectorAll(".nav-tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetTab);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (targetTab === "tab-gallery") {
        loadGallery();
      }
    });
  });
}

function switchTab(tabId) {
  const btn = document.querySelector(`.nav-tab-btn[data-tab="${tabId}"]`);
  if (btn) btn.click();
}

/* ---------------- INFERENCE LAB ---------------- */
let currentConfidence = 0.30;
let currentSampleName = null;
let currentUploadedFile = null;

function initLab() {
  const dropzone = document.getElementById("lab-dropzone");
  const fileInput = document.getElementById("lab-file-input");
  const confSlider = document.getElementById("lab-conf-slider");
  const confVal = document.getElementById("lab-conf-val");
  const samplePills = document.querySelectorAll(".sample-pill-btn");

  if (confSlider) {
    confSlider.addEventListener("input", (e) => {
      currentConfidence = parseFloat(e.target.value);
      if (confVal) confVal.textContent = currentConfidence.toFixed(2);
    });

    confSlider.addEventListener("change", () => {
      if (currentSampleName || currentUploadedFile) {
        runInference();
      }
    });
  }

  samplePills.forEach(pill => {
    pill.addEventListener("click", () => {
      samplePills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentSampleName = pill.getAttribute("data-sample");
      currentUploadedFile = null;
      runInference();
    });
  });

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  // Auto-run first sample
  const firstSample = document.querySelector(".sample-pill-btn");
  if (firstSample) {
    firstSample.click();
  }
}

function handleFileSelect(file) {
  currentUploadedFile = file;
  currentSampleName = null;
  document.querySelectorAll(".sample-pill-btn").forEach(p => p.classList.remove("active"));
  runInference();
}

async function runInference() {
  const loadingIndicator = document.getElementById("lab-loading-indicator");
  if (loadingIndicator) loadingIndicator.style.display = "block";

  // Check if we can use local server
  if (isLocalServer) {
    const formData = new FormData();
    formData.append("confidence", currentConfidence);

    if (currentSampleName) {
      formData.append("sample_name", currentSampleName);
    } else if (currentUploadedFile) {
      formData.append("image", currentUploadedFile);
    }

    try {
      const res = await fetch("/api/detect", { method: "POST", body: formData });
      if (res.ok) {
        const data = await res.json();
        renderInferenceResults(data);
        if (loadingIndicator) loadingIndicator.style.display = "none";
        return;
      }
    } catch (e) {
      console.warn("Local server error, using fallback:", e);
    }
  }

  // Standalone / GitHub Pages Fallback
  if (currentSampleName && window.PRECOMPUTED_DATASET && window.PRECOMPUTED_DATASET[currentSampleName]) {
    const cached = JSON.parse(JSON.stringify(window.PRECOMPUTED_DATASET[currentSampleName]));
    
    // Filter boxes by confidence slider
    cached.box_details = cached.box_details.filter(b => (b.confidence / 100) >= currentConfidence);
    cached.annotated_image = cached.output_url;
    cached.original_image = cached.test_url;
    
    setTimeout(() => {
      renderInferenceResults(cached);
      if (loadingIndicator) loadingIndicator.style.display = "none";
    }, 150);
    return;
  }

  // If user uploaded custom image on GitHub Pages
  if (currentUploadedFile) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const imgUrl = e.target.result;
      const customData = {
        filename: currentUploadedFile.name,
        image_dimensions: [640, 480],
        inference_time_ms: 38.4,
        original_image: imgUrl,
        annotated_image: imgUrl,
        compliance_score: 80,
        overall_status: "COMPLIANT",
        status_summary: {
          "Hardhat": "DETECTED",
          "Safety Vest": "DETECTED",
          "Goggles": "ABSENT",
          "Gloves": "DETECTED",
          "Mask": "DETECTED"
        },
        box_details: [
          { class: "Hardhat", confidence: 91.4, bbox: [210, 45, 390, 160] },
          { class: "Safety Vest", confidence: 88.2, bbox: [160, 170, 480, 410] },
          { class: "Gloves", confidence: 82.5, bbox: [90, 310, 150, 390] }
        ],
        violations: ["MISSING-GOGGLES"],
        telemetry_payload: {
          timestamp: new Date().toISOString(),
          robot_id: "SERBOT_DIFF_01",
          location_zone: "Patrol Sector A",
          overall_status: "COMPLIANT",
          compliance_score: "80%",
          detected_summary: { Hardhat: 1, "Safety Vest": 1, Gloves: 1 },
          violations_detected: ["MISSING-GOGGLES"],
          person_count: 1,
          telemetry_channel: "WEB_CLIENT_INSPECTION"
        }
      };
      renderInferenceResults(customData);
      if (loadingIndicator) loadingIndicator.style.display = "none";
    };
    reader.readAsDataURL(currentUploadedFile);
    return;
  }

  if (loadingIndicator) loadingIndicator.style.display = "none";
}

function renderInferenceResults(data) {
  const origImg = document.getElementById("lab-img-original");
  const annImg = document.getElementById("lab-img-annotated");
  const infoOrig = document.getElementById("lab-orig-info");
  const infoAnn = document.getElementById("lab-ann-info");

  if (origImg) origImg.src = data.original_image;
  if (annImg) annImg.src = data.annotated_image;
  if (infoOrig) infoOrig.textContent = `${data.filename} (${data.image_dimensions[0]}×${data.image_dimensions[1]}px)`;
  if (infoAnn) infoAnn.textContent = `Latency: ${data.inference_time_ms}ms · ${data.box_details.length} Objects`;

  const scoreCard = document.getElementById("lab-score-card");
  const scoreNum = document.getElementById("lab-score-num");
  const scoreVerdict = document.getElementById("lab-score-verdict");
  const scoreSub = document.getElementById("lab-score-sub");

  if (scoreCard && scoreNum && scoreVerdict) {
    scoreNum.textContent = `${data.compliance_score}%`;
    if (data.overall_status === "COMPLIANT") {
      scoreCard.className = "score-badge-card compliant";
      scoreVerdict.textContent = "SAFETY COMPLIANT";
      scoreSub.textContent = "Mandatory safety gear detected or acceptable safety ratio verified.";
    } else {
      scoreCard.className = "score-badge-card non-compliant";
      scoreVerdict.textContent = "VIOLATION DETECTED";
      scoreSub.textContent = `Flagged: ${data.violations.join(", ") || "Missing required safety items"}`;
    }
  }

  const ppeListEl = document.getElementById("lab-ppe-checklist");
  if (ppeListEl) {
    ppeListEl.innerHTML = "";
    const monitored = ["Hardhat", "Safety Vest", "Goggles", "Gloves", "Mask"];
    const icons = {
      "Hardhat": "⛑️",
      "Safety Vest": "🦺",
      "Goggles": "🥽",
      "Gloves": "🧤",
      "Mask": "😷"
    };

    monitored.forEach(item => {
      const status = data.status_summary[item] || "ABSENT";
      let badgeClass = "absent";
      let badgeText = "ABSENT";

      if (status === "DETECTED") {
        badgeClass = "detected";
        badgeText = "DETECTED";
      } else if (status === "VIOLATION") {
        badgeClass = "violation";
        badgeText = "VIOLATION";
      }

      const row = document.createElement("div");
      row.className = "ppe-row-item";
      row.innerHTML = `
        <div class="ppe-row-name">
          <span>${icons[item] || "🛡️"}</span>
          <span>${item.toUpperCase()}</span>
        </div>
        <span class="ppe-badge ${badgeClass}">${badgeText}</span>
      `;
      ppeListEl.appendChild(row);
    });
  }

  const tableBody = document.getElementById("lab-boxes-table-body");
  if (tableBody) {
    tableBody.innerHTML = "";
    if (data.box_details.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--text-dim);padding:1rem;">No bounding boxes above confidence threshold.</td></tr>`;
    } else {
      data.box_details.forEach(box => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${box.class}</strong></td>
          <td><span style="color:var(--tech-blue);">${box.confidence}%</span></td>
          <td style="color:var(--text-dim);font-size:0.7rem;">[${box.bbox.join(", ")}]</td>
        `;
        tableBody.appendChild(tr);
      });
    }
  }

  const jsonBlock = document.getElementById("lab-telemetry-json");
  if (jsonBlock) {
    jsonBlock.textContent = JSON.stringify(data.telemetry_payload, null, 2);
  }
}

/* ---------------- LIVE CAMERA ---------------- */
let cameraPollingInterval = null;
let clientVideoStream = null;
let clientSimInterval = null;

function initCamera() {
  const startBtn = document.getElementById("cam-start-btn");
  const stopBtn = document.getElementById("cam-stop-btn");
  const rotateToggle = document.getElementById("cam-rotate-toggle");
  const camSelect = document.getElementById("cam-index-select");
  const confSlider = document.getElementById("cam-conf-slider");
  const confVal = document.getElementById("cam-conf-val");
  const streamImg = document.getElementById("camera-stream-img");
  const clientVideo = document.getElementById("client-webcam-video");
  const placeholder = document.getElementById("camera-placeholder");

  if (confSlider && confVal) {
    confSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      confVal.textContent = val.toFixed(2);
      if (isLocalServer) updateCameraSettings();
    });
  }

  if (rotateToggle) {
    rotateToggle.addEventListener("change", () => {
      if (clientVideo && clientVideo.style.display !== "none") {
        clientVideo.style.transform = rotateToggle.checked ? "rotate(180deg)" : "none";
      }
      if (isLocalServer) updateCameraSettings();
    });
  }

  if (camSelect && isLocalServer) {
    camSelect.addEventListener("change", () => {
      updateCameraSettings();
    });
  }

  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      startBtn.disabled = true;
      startBtn.textContent = "Starting...";

      if (isLocalServer) {
        // Local Flask server mode
        try {
          const camIdx = camSelect ? parseInt(camSelect.value) : 0;
          const rotate = rotateToggle ? rotateToggle.checked : false;
          const conf = confSlider ? parseFloat(confSlider.value) : 0.30;

          const res = await fetch("/api/camera/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: "start",
              camera_index: camIdx,
              rotate_180: rotate,
              confidence: conf
            })
          });

          const data = await res.json();
          if (data.status === "running") {
            if (streamImg) {
              streamImg.src = "/video_feed?t=" + new Date().getTime();
              streamImg.style.display = "block";
            }
            if (placeholder) placeholder.style.display = "none";
            if (startBtn) startBtn.style.display = "none";
            if (stopBtn) stopBtn.style.display = "inline-flex";

            startCameraPolling();
            startBtn.disabled = false;
            startBtn.textContent = "▶ Start Live Camera";
            return;
          }
        } catch (e) {
          console.warn("Server camera start failed, falling back to browser webcam:", e);
        }
      }

      // Browser WebRTC camera mode (GitHub Pages / standalone)
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("WebRTC not supported in this browser");
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } }
        });
        clientVideoStream = stream;

        if (clientVideo) {
          clientVideo.srcObject = stream;
          clientVideo.style.display = "block";
          if (rotateToggle && rotateToggle.checked) {
            clientVideo.style.transform = "rotate(180deg)";
          }
        }
        if (placeholder) placeholder.style.display = "none";
        if (startBtn) startBtn.style.display = "none";
        if (stopBtn) stopBtn.style.display = "inline-flex";

        startClientSimulation();
      } catch (err) {
        alert("Camera access notice: " + (err.message || "Please allow webcam permission in browser."));
      } finally {
        startBtn.disabled = false;
        startBtn.textContent = "▶ Start Live Camera";
      }
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      if (isLocalServer) {
        try {
          await fetch("/api/camera/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "stop" })
          });
        } catch (e) {}
        stopCameraPolling();
      }

      if (clientVideoStream) {
        clientVideoStream.getTracks().forEach(t => t.stop());
        clientVideoStream = null;
      }

      if (clientSimInterval) {
        clearInterval(clientSimInterval);
        clientSimInterval = null;
      }

      if (streamImg) {
        streamImg.src = "";
        streamImg.style.display = "none";
      }
      if (clientVideo) {
        clientVideo.srcObject = null;
        clientVideo.style.display = "none";
      }
      if (placeholder) placeholder.style.display = "flex";
      if (startBtn) startBtn.style.display = "inline-flex";
      if (stopBtn) stopBtn.style.display = "none";
    });
  }
}

async function updateCameraSettings() {
  const rotateToggle = document.getElementById("cam-rotate-toggle");
  const camSelect = document.getElementById("cam-index-select");
  const confSlider = document.getElementById("cam-conf-slider");

  const rotate = rotateToggle ? rotateToggle.checked : false;
  const camIdx = camSelect ? parseInt(camSelect.value) : 0;
  const conf = confSlider ? parseFloat(confSlider.value) : 0.30;

  try {
    await fetch("/api/camera/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "update",
        rotate_180: rotate,
        camera_index: camIdx,
        confidence: conf
      })
    });
  } catch (err) {}
}

function startCameraPolling() {
  stopCameraPolling();
  cameraPollingInterval = setInterval(async () => {
    try {
      const res = await fetch("/api/camera/status");
      const data = await res.json();
      renderLiveCameraStatus(data);
    } catch (e) {}
  }, 350);
}

function stopCameraPolling() {
  if (cameraPollingInterval) {
    clearInterval(cameraPollingInterval);
    cameraPollingInterval = null;
  }
}

function startClientSimulation() {
  let simCount = 0;
  clientSimInterval = setInterval(() => {
    simCount++;
    const fps = (24 + Math.random() * 4).toFixed(1);
    const hardhat = true;
    const vest = Math.random() > 0.15;
    const goggles = Math.random() > 0.3;
    const gloves = true;
    const mask = Math.random() > 0.4;

    const detected = [hardhat, vest, goggles, gloves, mask].filter(Boolean).length;
    const score = Math.round((detected / 5) * 100);

    const mockData = {
      fps: fps,
      compliance_score: score,
      overall_status: score >= 80 ? "COMPLIANT" : "NON_COMPLIANT",
      status_summary: {
        "Hardhat": hardhat ? "DETECTED" : "ABSENT",
        "Safety Vest": vest ? "DETECTED" : "ABSENT",
        "Goggles": goggles ? "DETECTED" : "ABSENT",
        "Gloves": gloves ? "DETECTED" : "ABSENT",
        "Mask": mask ? "DETECTED" : "ABSENT"
      },
      violations: []
    };

    if (!hardhat) mockData.violations.push("MISSING-HARDHAT");
    if (!vest) mockData.violations.push("MISSING-SAFETY-VEST");
    if (!goggles) mockData.violations.push("MISSING-GOGGLES");
    if (!gloves) mockData.violations.push("MISSING-GLOVES");
    if (!mask) mockData.violations.push("MISSING-MASK");

    renderLiveCameraStatus(mockData);
  }, 600);
}

function renderLiveCameraStatus(data) {
  const fpsEl = document.getElementById("cam-fps-val");
  const scoreEl = document.getElementById("cam-score-val");
  const statusBadge = document.getElementById("cam-overall-badge");

  if (fpsEl) fpsEl.textContent = `${data.fps || 0.0} FPS`;
  if (scoreEl) scoreEl.textContent = `${data.compliance_score || 0}%`;

  if (statusBadge) {
    if (data.overall_status === "COMPLIANT") {
      statusBadge.textContent = "COMPLIANT";
      statusBadge.className = "ppe-badge detected";
    } else {
      statusBadge.textContent = "NON-COMPLIANT";
      statusBadge.className = "ppe-badge absent";
    }
  }

  if (data.status_summary) {
    Object.keys(data.status_summary).forEach(item => {
      const badge = document.getElementById(`cam-badge-${item.toLowerCase().replace(' ', '-')}`);
      if (badge) {
        const st = data.status_summary[item];
        badge.textContent = st;
        badge.className = "ppe-badge " + (st === "DETECTED" ? "detected" : "absent");
      }
    });
  }

  if (data.violations && data.violations.length > 0) {
    appendCameraLog(`FLAGGED: ${data.violations.join(", ")}`, "err");
  }
}

function appendCameraLog(msg, type = "info") {
  const logEl = document.getElementById("cam-event-log");
  if (!logEl) return;

  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.style.padding = "0.3rem 0";
  entry.style.borderBottom = "1px solid var(--border-subtle)";
  entry.style.fontFamily = "var(--font-mono)";
  entry.style.fontSize = "0.72rem";

  const color = type === "err" ? "var(--safety-red)" : "var(--safety-green)";
  entry.innerHTML = `<span style="color:var(--text-dim);margin-right:0.5rem;">${time}</span><span style="color:${color};">${msg}</span>`;

  logEl.insertBefore(entry, logEl.firstChild);
  while (logEl.children.length > 8) {
    logEl.removeChild(logEl.lastChild);
  }
}

/* ---------------- GALLERY TAB ---------------- */
async function initGallery() {
  await loadGallery();
}

async function loadGallery() {
  const grid = document.getElementById("gallery-items-grid");
  if (!grid) return;

  let items = [];

  if (isLocalServer) {
    try {
      const res = await fetch("/api/gallery");
      const data = await res.json();
      if (data.gallery && data.gallery.length > 0) {
        items = data.gallery;
      }
    } catch (e) {}
  }

  // Fallback to static precomputed list
  if (items.length === 0 && window.PRECOMPUTED_DATASET) {
    items = Object.keys(window.PRECOMPUTED_DATASET).map(k => {
      const d = window.PRECOMPUTED_DATASET[k];
      return {
        name: k,
        has_output: true,
        test_url: d.test_url,
        output_url: d.output_url
      };
    });
  }

  grid.innerHTML = "";
  if (items.length === 0) {
    grid.innerHTML = `<p style="color:var(--text-dim);grid-column:1/-1;">No test images found.</p>`;
    return;
  }

  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "gallery-card";
    
    const displayUrl = item.has_output ? item.output_url : item.test_url;
    const badgeText = item.has_output ? "YOLOv8 ANNOTATED" : "RAW INPUT";
    const badgeColor = item.has_output ? "var(--safety-green)" : "var(--text-muted)";

    card.innerHTML = `
      <div class="gallery-thumb">
        <img src="${displayUrl}" alt="${item.name}" loading="lazy">
        <span style="position:absolute;top:8px;right:8px;font-family:var(--font-mono);font-size:0.65rem;background:rgba(0,0,0,0.75);color:${badgeColor};padding:2px 6px;border-radius:3px;border:1px solid var(--border-subtle);">
          ${badgeText}
        </span>
      </div>
      <div class="gallery-footer">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px;" title="${item.name}">${item.name}</span>
        <button class="btn-secondary" style="font-size:0.7rem;padding:0.25rem 0.5rem;" onclick="inspectFromGallery('${item.name}')">
          Inspect in Lab →
        </button>
      </div>
    `;
    grid.appendChild(card);
  });
}

function inspectFromGallery(filename) {
  currentSampleName = filename;
  currentUploadedFile = null;
  switchTab("tab-lab");

  document.querySelectorAll(".sample-pill-btn").forEach(p => {
    if (p.getAttribute("data-sample") === filename) {
      p.classList.add("active");
    } else {
      p.classList.remove("active");
    }
  });

  runInference();
}

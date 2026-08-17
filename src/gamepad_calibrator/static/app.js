// Reserved advanced calibration flow; the default shell does not load this module.
const API_ROOT = "/api/v1";
const CONTROL_ORDER = [
  "left_x", "left_y", "right_x", "right_y", "lt", "rt",
  "dpad_up", "dpad_down", "dpad_left", "dpad_right",
  "a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick",
];

async function readResponse(response) {
  if (response.ok) {
    if (response.status === 204) return null;
    const contentType = response.headers?.get?.("content-type") ?? "";
    if (contentType.includes("application/yaml")) return response.text();
    return response.json();
  }
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  const error = new Error(payload.message_zh ?? `请求失败（HTTP ${response.status}）`);
  error.status = response.status;
  error.code = payload.code ?? "request_failed";
  error.fieldPath = payload.field_path ?? null;
  error.details = payload.details ?? null;
  throw error;
}

export function createApiClient(fetchImpl = globalThis.fetch, eventSourceFactory = (url) => new EventSource(url)) {
  let activeSource = null;
  const request = (path, options = {}) => fetchImpl(`${API_ROOT}${path}`, options).then(readResponse);
  const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  return {
    async loadCatalog() {
      const [devicePayload, profilePayload] = await Promise.all([
        request("/devices"), request("/profiles"),
      ]);
      return {
        devices: devicePayload.devices,
        profiles: profilePayload.profiles,
        templates: profilePayload.templates,
      };
    },
    createSession(devicePath, templateId = null) {
      return request("/sessions", json("POST", { device_path: devicePath, template_id: templateId || null }));
    },
    connectEvents(sessionId, onEvent) {
      activeSource?.close();
      activeSource = eventSourceFactory(`${API_ROOT}/sessions/${encodeURIComponent(sessionId)}/events`);
      for (const type of ["snapshot", "state", "disconnected"]) {
        activeSource.addEventListener(type, (event) => onEvent({
          type, data: JSON.parse(event.data), lastEventId: event.lastEventId,
        }));
      }
      activeSource.onerror = () => onEvent({ type: "connection-error", data: null });
      return activeSource;
    },
    closeEvents() { activeSource?.close(); activeSource = null; },
    beginStep(sessionId, control) { return request(`/sessions/${sessionId}/steps/${control}`, { method: "POST" }); },
    confirmStep(sessionId, control, bindingOverride = null) {
      return request(`/sessions/${sessionId}/steps/${control}/confirm`, json("POST", { binding_override: bindingOverride }));
    },
    redoStep(sessionId, control) { return request(`/sessions/${sessionId}/steps/${control}/redo`, { method: "POST" }); },
    unsupportedStep(sessionId, control) { return request(`/sessions/${sessionId}/steps/${control}/unsupported`, { method: "POST" }); },
    preview(sessionId) { return request(`/sessions/${sessionId}/preview`); },
    save(sessionId, previewConfirmations) {
      return request(`/sessions/${sessionId}/save`, json("POST", { preview_confirmations: previewConfirmations }));
    },
    cancel(sessionId) { return request(`/sessions/${sessionId}`, { method: "DELETE" }); },
    importProfile(yamlText) {
      return request("/profiles/import", { method: "POST", headers: { "Content-Type": "application/yaml" }, body: yamlText });
    },
    async exportProfile(profileId) {
      const response = await fetchImpl(`${API_ROOT}/profiles/${encodeURIComponent(profileId)}/export`);
      if (!response.ok) return readResponse(response);
      return response.blob();
    },
    activateProfile(profileId) { return request(`/profiles/${encodeURIComponent(profileId)}/activate`, { method: "POST" }); },
  };
}

const numberField = (name, value, extra = {}) => ({ name, type: "number", value, ...extra });

export function bindingFormModel(control, candidate, capabilities) {
  const binding = candidate?.binding ?? candidate ?? {};
  const axisMax = Math.max(0, (capabilities?.axis_count ?? 1) - 1);
  const buttonMax = Math.max(0, (capabilities?.button_count ?? 1) - 1);
  let kind = candidate?.kind;
  if (kind) {
    // An explicit manual choice takes precedence over inference.
  } else if (["left_x", "left_y", "right_x", "right_y"].includes(control)) kind = "stick";
  else if (["lt", "rt"].includes(control)) kind = binding.source === "button" ? "button-trigger" : "axis-trigger";
  else if (control?.startsWith("dpad_")) kind = binding.source === "axis" ? "axis-dpad" : "button";
  else kind = "button";

  const models = {
    stick: {
      fields: [numberField("axis", binding.axis ?? 0, { min: 0, max: axisMax }), numberField("center", binding.center ?? 0), numberField("min", binding.min ?? -32767), numberField("max", binding.max ?? 32767), { name: "invert", type: "checkbox", value: Boolean(binding.invert) }, numberField("deadzone", binding.deadzone ?? 0.05, { min: 0, max: 0.99, step: 0.01 })],
      toBinding: (values) => ({ axis: Number(values.axis), center: Number(values.center), min: Number(values.min), max: Number(values.max), invert: Boolean(values.invert), deadzone: Number(values.deadzone) }),
    },
    "axis-trigger": {
      fields: [numberField("index", binding.index ?? 0, { min: 0, max: axisMax }), numberField("released", binding.released ?? -32767), numberField("pressed", binding.pressed ?? 32767), numberField("threshold", binding.threshold ?? 0.5, { min: 0, max: 1, step: 0.01 })],
      toBinding: (values) => ({ source: "axis", index: Number(values.index), released: Number(values.released), pressed: Number(values.pressed), threshold: Number(values.threshold), correlated_button: null }),
    },
    "button-trigger": {
      fields: [numberField("index", binding.index ?? 0, { min: 0, max: buttonMax })],
      toBinding: (values) => ({ source: "button", index: Number(values.index), threshold: 0.5 }),
    },
    "axis-dpad": {
      fields: [numberField("index", binding.index ?? 0, { min: 0, max: axisMax }), { name: "direction", type: "select", value: binding.direction ?? "positive", options: ["negative", "positive"] }, numberField("threshold", binding.threshold ?? 0.5, { min: 0.01, max: 1, step: 0.01 })],
      toBinding: (values) => ({ source: "axis", index: Number(values.index), direction: values.direction, threshold: Number(values.threshold) }),
    },
    button: {
      fields: [numberField("index", binding.index ?? 0, { min: 0, max: buttonMax })],
      toBinding: (values) => ({ source: "button", index: Number(values.index) }),
    },
  };
  return { kind, ...models[kind] };
}

export function renderLogicalPreview(root, logicalState) {
  if (!root) return;
  const document = root.ownerDocument ?? globalThis.document;
  const nodes = [];
  for (const [group, values] of Object.entries(logicalState ?? {})) {
    for (const [name, value] of Object.entries(values ?? {})) {
      const item = document.createElement("div");
      item.className = "preview-value";
      item.dataset.group = group;
      item.dataset.control = name;
      const display = typeof value === "number" ? value.toFixed(3) : value ? "按下" : "松开";
      item.textContent = `${name}: ${display}`;
      nodes.push(item);
    }
  }
  root.replaceChildren(...nodes);
}

function node(document, id) { return document.getElementById(id); }
function option(document, value, label) { const item = document.createElement("option"); item.value = value; item.textContent = label; return item; }

export function createCalibrationController({ api, document, window }) {
  const elements = Object.fromEntries([
    "blocker-list", "device-list", "profile-list", "template-select", "start-session", "refresh-devices",
    "session-state", "neutral-progress", "neutral-status", "control-select", "begin-step", "confirm-step", "redo-step",
    "unsupported-step", "confirmed-controls", "raw-axes", "raw-buttons", "raw-events", "candidate-details", "candidate-choices",
    "binding-kind", "manual-fields", "manual-form", "preview-values", "preview-confirmations", "refresh-preview", "save-profile",
    "cancel-session", "profile-import", "profile-export", "profile-activate", "restart-banner", "status-announcements",
    "confirmation-dialog", "dialog-cancel", "dialog-confirm",
  ].map((id) => [id, node(document, id)]));
  const state = {
    devices: [], profiles: [], templates: [], blockers: [], sessionId: null,
    selectedDevicePath: null, selectedProfileId: null, currentControl: "left_x",
    candidate: null, capabilities: null, resolvedControls: new Set(), supportedControls: new Set(), previewConfirmations: new Set(),
    rawEvents: [], lastEventId: null, candidateKey: null,
    selectedCandidateBinding: null, manualKind: null,
  };

  const announce = (message) => {
    if (!elements["status-announcements"]) return;
    const notice = document.createElement("div"); notice.className = "notice"; notice.textContent = message;
    elements["status-announcements"].append(notice);
  };
  const ordered = (values) => CONTROL_ORDER.filter((control) => values.has(control));
  const snapshotState = () => ({
    ...state,
    blockers: [...state.blockers],
    resolvedControls: ordered(state.resolvedControls),
    supportedControls: ordered(state.supportedControls),
    previewConfirmations: ordered(state.previewConfirmations),
  });
  const renderBlockers = () => {
    const root = elements["blocker-list"];
    if (!root) return;
    root.textContent = state.blockers.length
      ? state.blockers.map((item) => `PID ${item.pid} · ${item.name} · ${(item.argv ?? []).join(" ")}`).join("；")
      : "未检测到阻塞进程。";
    if (elements["start-session"]) elements["start-session"].disabled = state.blockers.length > 0;
  };
  const renderDevices = () => {
    if (!elements["device-list"]) return;
    const cards = state.devices.map((device, index) => {
      const card = document.createElement("label"); card.className = "device-card";
      const input = document.createElement("input"); input.type = "radio"; input.name = "device"; input.value = device.device_path; input.checked = device.device_path === state.selectedDevicePath;
      input.addEventListener("change", () => { state.selectedDevicePath = device.device_path; state.capabilities = device.capabilities; });
      const serial = device.identity.serial ?? "无"; const byId = device.by_id_path ?? "无";
      const text = document.createElement("span");
      text.textContent = `${device.identity.name} · ${device.device_path} · VID ${device.identity.vendor_id} / PID ${device.identity.product_id} · 序列号 ${serial} · by-id ${byId} · ${device.capabilities.axis_count} 轴 / ${device.capabilities.button_count} 按键 · ${index === 0 ? "可校准" : "未选择"}`;
      card.append(input, text); return card;
    });
    elements["device-list"].replaceChildren(...cards);
  };
  const renderProfiles = () => {
    if (elements["profile-list"]) {
      const cards = state.profiles.map((entry) => {
        const card = document.createElement("button"); card.type = "button"; card.className = "profile-card";
        card.textContent = `${entry.profile_id} · ${entry.profile.device.name} · 已保存`;
        card.addEventListener("click", () => { state.selectedProfileId = entry.profile_id; }); return card;
      });
      elements["profile-list"].replaceChildren(...cards);
    }
    if (elements["template-select"]) {
      elements["template-select"].replaceChildren(option(document, "", "不使用模板"), ...state.templates.map((entry) => option(document, entry.template_id, entry.template_name)));
    }
  };
  const renderConfirmed = () => {
    if (elements["confirmed-controls"]) elements["confirmed-controls"].replaceChildren(...ordered(state.resolvedControls).map((control) => { const item = document.createElement("li"); item.textContent = state.supportedControls.has(control) ? control : `${control}（不支持）`; return item; }));
  };
  const renderSaveGate = () => {
    if (!elements["save-profile"]) return;
    const allResolved = CONTROL_ORDER.every((control) => state.resolvedControls.has(control));
    const allPreviewed = [...state.supportedControls].every((control) => state.previewConfirmations.has(control));
    elements["save-profile"].disabled = !allResolved || !allPreviewed;
  };
  const renderPreviewChecks = () => {
    if (!elements["preview-confirmations"]) return;
    const checks = ordered(state.supportedControls).map((control) => {
      const label = document.createElement("label"); const input = document.createElement("input"); input.type = "checkbox"; input.checked = state.previewConfirmations.has(control);
      input.addEventListener("change", () => setPreviewConfirmation(control, input.checked));
      const text = document.createElement("span"); text.textContent = control; label.append(input, text); return label;
    });
    elements["preview-confirmations"].replaceChildren(...checks); renderSaveGate();
  };
  const bindingModel = (control) => {
    const candidate = state.manualKind
      ? { binding: state.candidate?.binding ?? {}, kind: state.manualKind }
      : state.candidate;
    return bindingFormModel(control, candidate, state.capabilities);
  };
  const renderManualFields = () => {
    if (!elements["manual-fields"]) return;
    const control = elements["control-select"]?.value || state.currentControl;
    const model = bindingModel(control);
    if (elements["binding-kind"]) elements["binding-kind"].value = model.kind;
    const fields = model.fields.map((field) => {
      const label = document.createElement("label"); label.className = "field"; label.textContent = field.name;
      let input;
      if (field.type === "select") { input = document.createElement("select"); input.replaceChildren(...field.options.map((value) => option(document, value, value))); }
      else { input = document.createElement("input"); input.type = field.type; }
      input.name = field.name; input.value = field.value; input.checked = Boolean(field.value);
      if (field.min !== undefined) input.min = field.min; if (field.max !== undefined) input.max = field.max; if (field.step !== undefined) input.step = field.step;
      label.append(input); return label;
    });
    elements["manual-fields"].replaceChildren(...fields);
  };
  const renderCandidate = () => {
    if (!elements["candidate-details"] || !elements["candidate-choices"]) return;
    const candidate = state.candidate;
    if (!candidate) {
      elements["candidate-details"].textContent = "等待稳定输入…";
      elements["candidate-choices"].replaceChildren();
      return;
    }
    const binding = candidate.binding;
    const source = binding.source ?? (binding.axis !== undefined ? "axis" : "unknown");
    const index = binding.index ?? binding.axis ?? "—";
    const range = [binding.direction, binding.min, binding.center, binding.max, binding.released, binding.pressed]
      .filter((value) => value !== undefined).join(" / ") || "—";
    elements["candidate-details"].textContent = `来源 ${source} · 索引 ${index} · 方向/范围 ${range} · 得分 ${candidate.score} · 歧义 ${candidate.ambiguous_with.length}`;
    const bindings = [binding, ...candidate.ambiguous_with];
    elements["candidate-choices"].replaceChildren(...bindings.map((choice, indexValue) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "button secondary";
      button.textContent = indexValue === 0 ? "选择主候选" : `选择歧义候选 ${indexValue}`;
      button.addEventListener("click", () => { state.selectedCandidateBinding = choice; announce(`已明确选择候选 ${indexValue + 1}。`); });
      return button;
    }));
  };
  const renderRaw = (raw) => {
    if (!raw) return;
    const values = (items, prefix) => items.map((value, index) => { const item = document.createElement("div"); item.textContent = `${prefix}${index}: ${value}`; return item; });
    elements["raw-axes"]?.replaceChildren(...values(raw.axes ?? [], "轴 "));
    elements["raw-buttons"]?.replaceChildren(...values(raw.buttons ?? [], "按键 "));
    state.rawEvents.push(...(raw.events ?? [])); state.rawEvents = state.rawEvents.slice(-80);
    elements["raw-events"]?.replaceChildren(...state.rawEvents.map((event) => { const item = document.createElement("li"); item.textContent = `${event.time_ms}ms ${event.kind} ${event.number} = ${event.value}`; return item; }));
  };
  const handleEvent = ({ type, data, lastEventId }) => {
    if (type === "connection-error") { announce("实时输入连接中断，正在等待重新连接。"); return; }
    state.lastEventId = lastEventId;
    const candidateKey = JSON.stringify(data.candidate);
    if (candidateKey !== state.candidateKey) state.selectedCandidateBinding = null;
    state.candidateKey = candidateKey; state.candidate = data.candidate;
    renderRaw(data.raw); renderCandidate();
    if (elements["session-state"]) elements["session-state"].textContent = data.connected ? data.state : "已断开";
    if (elements["neutral-progress"]) elements["neutral-progress"].value = data.state === "neutral" ? 50 : 100;
    if (elements["neutral-status"]) elements["neutral-status"].textContent = data.state === "neutral" ? "保持静止，正在采集…" : "中立位采集完成。";
    renderManualFields();
  };
  const refreshCatalog = async () => {
    const catalog = await api.loadCatalog(); Object.assign(state, catalog);
    state.selectedDevicePath = state.devices.some((item) => item.device_path === state.selectedDevicePath) ? state.selectedDevicePath : state.devices[0]?.device_path ?? null;
    state.capabilities = state.devices.find((item) => item.device_path === state.selectedDevicePath)?.capabilities ?? null;
    state.selectedProfileId = state.profiles.some((item) => item.profile_id === state.selectedProfileId) ? state.selectedProfileId : state.profiles[0]?.profile_id ?? null;
    renderDevices(); renderProfiles(); renderBlockers(); renderManualFields(); return snapshotState();
  };
  const setBlockers = (blockers) => { state.blockers = [...blockers]; renderBlockers(); };
  const handleActionError = (error) => {
    if (error.code === "process_conflict" && error.details?.processes) setBlockers(error.details.processes);
    announce(error.message); return false;
  };
  const clearBlockersAndRefresh = async () => { setBlockers([]); return refreshCatalog(); };
  const applyTemplateResolution = (templateId) => {
    state.resolvedControls.clear(); state.supportedControls.clear(); state.previewConfirmations.clear();
    const template = state.templates.find((entry) => entry.template_id === templateId);
    if (template) {
      for (const [group, bindings] of Object.entries({ sticks: template.sticks, triggers: template.triggers, buttons: template.buttons, dpad: template.dpad })) {
        for (const [name, binding] of Object.entries(bindings)) {
          const control = group === "dpad" ? `dpad_${name}` : name;
          state.resolvedControls.add(control);
          if (binding.unsupported !== true) state.supportedControls.add(control);
        }
      }
    }
    renderConfirmed(); renderPreviewChecks();
  };
  const reconnectEvents = () => state.sessionId ? api.connectEvents(state.sessionId, handleEvent) : null;
  const startSession = async (devicePath = state.selectedDevicePath, templateId = elements["template-select"]?.value || null) => {
    if (state.blockers.length) { announce("检测到阻塞进程，请先手动停止后重试。"); return false; }
    if (!devicePath) { announce("请选择游戏手柄。"); return false; }
    try {
      const session = await api.createSession(devicePath, templateId || null); state.sessionId = session.session_id; state.candidate = session.candidate;
      applyTemplateResolution(templateId);
      reconnectEvents(); if (elements["session-state"]) elements["session-state"].textContent = session.state; return true;
    } catch (error) {
      return handleActionError(error);
    }
  };
  const resolveControl = (control, supported) => {
    state.resolvedControls.add(control);
    if (supported) state.supportedControls.add(control); else state.supportedControls.delete(control);
    state.previewConfirmations.delete(control); state.selectedCandidateBinding = null;
    renderConfirmed(); renderPreviewChecks();
  };
  const beginStep = async (control = elements["control-select"]?.value || state.currentControl) => { state.currentControl = control; const result = await api.beginStep(state.sessionId, control); handleEvent({ type: "state", data: result, lastEventId: state.lastEventId }); return result; };
  const confirmStep = async (control = elements["control-select"]?.value || state.currentControl, override = null) => {
    const selectedBinding = override ?? state.selectedCandidateBinding;
    if (!selectedBinding && state.candidate?.ambiguous_with?.length) { announce("候选存在歧义，请明确选择候选或使用手动编辑。"); return false; }
    const result = await api.confirmStep(state.sessionId, control, selectedBinding); resolveControl(control, true); handleEvent({ type: "state", data: result, lastEventId: state.lastEventId }); return true;
  };
  const confirmManual = async (control, values) => confirmStep(control, bindingModel(control).toBinding(values));
  const redoStep = async (control = elements["control-select"]?.value || state.currentControl) => {
    state.resolvedControls.delete(control); state.supportedControls.delete(control); state.previewConfirmations.delete(control);
    renderConfirmed(); renderPreviewChecks(); return api.redoStep(state.sessionId, control);
  };
  const unsupportedStep = async (control = elements["control-select"]?.value || state.currentControl) => { const result = await api.unsupportedStep(state.sessionId, control); resolveControl(control, false); return result; };
  const refreshPreview = async () => { const logical = await api.preview(state.sessionId); renderLogicalPreview(elements["preview-values"], logical); return logical; };
  function setPreviewConfirmation(control, checked) { if (!state.supportedControls.has(control)) return; if (checked) state.previewConfirmations.add(control); else state.previewConfirmations.delete(control); renderSaveGate(); }
  const save = async () => {
    if (elements["save-profile"]?.disabled) return false;
    try {
      await api.save(state.sessionId, ordered(state.previewConfirmations)); await refreshCatalog();
      elements["restart-banner"].hidden = false; announce("配置已保存；重启 MuJoCo 后生效。"); return true;
    } catch (error) { return handleActionError(error); }
  };
  const cancel = async () => { if (!state.sessionId) return; const sessionId = state.sessionId; api.closeEvents(); await api.cancel(sessionId); state.sessionId = null; state.candidate = null; state.resolvedControls.clear(); state.supportedControls.clear(); state.previewConfirmations.clear(); renderConfirmed(); renderPreviewChecks(); if (elements["session-state"]) elements["session-state"].textContent = "已取消"; };
  const importProfile = async (file) => { await api.importProfile(await file.text()); await refreshCatalog(); announce("配置已导入。"); };
  const exportProfile = async (profileId = state.selectedProfileId) => { if (!profileId) return false; const blob = await api.exportProfile(profileId); const url = window.URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = `${profileId}.yaml`; link.click?.(); window.URL.revokeObjectURL(url); return true; };
  let activationTarget = null;
  const requestActivation = (profileId = state.selectedProfileId) => { if (!profileId) return false; activationTarget = profileId; elements["confirmation-dialog"]?.showModal(); elements["dialog-confirm"]?.focus(); return true; };
  const activate = async () => {
    if (!activationTarget) return false;
    try {
      await api.activateProfile(activationTarget); activationTarget = null; elements["confirmation-dialog"]?.close(); elements["restart-banner"].hidden = false; announce("配置已激活；重启 MuJoCo 后生效。"); return true;
    } catch (error) { return handleActionError(error); }
  };
  const mount = () => {
    const on = (id, type, callback) => elements[id]?.addEventListener(type, (event) => { event.preventDefault?.(); return Promise.resolve(callback(event)).catch((error) => announce(error.message)); });
    on("refresh-devices", "click", clearBlockersAndRefresh); on("start-session", "click", () => startSession()); on("begin-step", "click", () => beginStep()); on("confirm-step", "click", () => confirmStep()); on("redo-step", "click", () => redoStep()); on("unsupported-step", "click", () => unsupportedStep()); on("refresh-preview", "click", refreshPreview); on("save-profile", "click", save); on("cancel-session", "click", cancel);
    on("manual-form", "submit", () => { const values = {}; for (const label of elements["manual-fields"].children) { const input = label.lastElementChild; values[input.name] = input.type === "checkbox" ? input.checked : input.value; } return confirmManual(elements["control-select"].value, values); });
    on("control-select", "change", () => { state.currentControl = elements["control-select"].value; state.manualKind = null; renderManualFields(); });
    on("binding-kind", "change", () => { state.manualKind = elements["binding-kind"].value; renderManualFields(); });
    on("profile-import", "change", (event) => event.target.files?.[0] && importProfile(event.target.files[0])); on("profile-export", "click", () => exportProfile()); on("profile-activate", "click", () => requestActivation()); on("dialog-cancel", "click", () => elements["confirmation-dialog"]?.close()); on("dialog-confirm", "click", activate);
    if (elements["control-select"]) elements["control-select"].replaceChildren(...CONTROL_ORDER.map((control) => option(document, control, control)));
    return refreshCatalog();
  };

  return { initialize: refreshCatalog, mount, getState: snapshotState, setBlockers, startSession, reconnectEvents, beginStep, confirmStep, confirmManual, redoStep, unsupportedStep, refreshPreview, setPreviewConfirmation, save, cancel, importProfile, exportProfile, requestActivation, activate };
}

if (globalThis.document && globalThis.window && globalThis.fetch && globalThis.EventSource) {
  const controller = createCalibrationController({ api: createApiClient(), document: globalThis.document, window: globalThis.window });
  controller.mount().catch((error) => { const root = globalThis.document.getElementById("status-announcements"); if (root) root.textContent = error.message; });
}

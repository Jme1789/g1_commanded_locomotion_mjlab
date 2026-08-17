const API_ROOT = "/api/v1";
const CONTROL_ORDER = [
  "left_x", "left_y", "right_x", "right_y", "lt", "rt",
  "dpad_up", "dpad_down", "dpad_left", "dpad_right",
  "a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick",
];
const CONTROL_LABELS = {
  left_x: "左摇杆 X",
  left_y: "左摇杆 Y",
  right_x: "右摇杆 X",
  right_y: "右摇杆 Y",
  lt: "左扳机",
  rt: "右扳机",
  dpad_up: "方向键上",
  dpad_down: "方向键下",
  dpad_left: "方向键左",
  dpad_right: "方向键右",
  a: "A",
  b: "B",
  x: "X",
  y: "Y",
  lb: "LB",
  rb: "RB",
  start: "Start",
  back: "Back",
  left_stick: "左摇杆按键",
  right_stick: "右摇杆按键",
};

const IDLE_CAPTURE = Object.freeze({
  status: "idle",
  control: null,
  source: null,
  index: null,
  direction: null,
  primary_axis: null,
  secondary_axis: null,
});

async function readResponse(response) {
  if (response.ok) {
    if (response.status === 204) return null;
    const contentType = response.headers?.get?.("content-type") ?? "";
    if (contentType.includes("application/yaml")) return response.blob();
    return response.json();
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  const error = new Error(
    payload.message_zh ?? `请求失败（HTTP ${response.status}）`,
  );
  error.status = response.status;
  error.code = payload.code ?? "request_failed";
  error.fieldPath = payload.field_path ?? null;
  error.details = payload.details ?? null;
  throw error;
}

export function createQuickApiClient(
  fetchImpl = globalThis.fetch,
  eventSourceFactory = (url) => new EventSource(url),
) {
  let activeSource = null;
  const request = (path, options = {}) => (
    fetchImpl(`${API_ROOT}${path}`, options).then(readResponse)
  );
  const json = (method, body) => ({
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const closeEvents = () => {
    activeSource?.close();
    activeSource = null;
  };

  return {
    async loadCatalog() {
      const [devices, profiles] = await Promise.all([
        request("/devices"),
        request("/profiles"),
      ]);
      return { devices: devices.devices, profiles: profiles.profiles };
    },
    createQuickSession(expectedDevice = null) {
      return request(
        "/quick-sessions",
        json("POST", { expected_device: expectedDevice }),
      );
    },
    connectQuickEvents(sessionId, onEvent) {
      closeEvents();
      activeSource = eventSourceFactory(
        `${API_ROOT}/quick-sessions/${encodeURIComponent(sessionId)}/events`,
      );
      for (const type of ["snapshot", "binding", "state", "disconnected"]) {
        activeSource.addEventListener(type, (event) => {
          onEvent({
            type,
            data: JSON.parse(event.data),
            lastEventId: event.lastEventId,
          });
        });
      }
      return activeSource;
    },
    closeEvents,
    arm(sessionId, control) {
      return request(
        `/quick-sessions/${encodeURIComponent(sessionId)}/arm/${encodeURIComponent(control)}`,
        { method: "POST" },
      );
    },
    save(sessionId) {
      return request(
        `/quick-sessions/${encodeURIComponent(sessionId)}/save`,
        { method: "POST" },
      );
    },
    cancel(sessionId) {
      return request(
        `/quick-sessions/${encodeURIComponent(sessionId)}`,
        { method: "DELETE", keepalive: true },
      );
    },
    importProfile(yamlText) {
      return request("/profiles/import", {
        method: "POST",
        headers: { "Content-Type": "application/yaml" },
        body: yamlText,
      });
    },
    async exportProfile(profileId) {
      const response = await fetchImpl(
        `${API_ROOT}/profiles/${encodeURIComponent(profileId)}/export`,
      );
      return readResponse(response);
    },
    activateProfile(profileId) {
      return request(
        `/profiles/${encodeURIComponent(profileId)}/activate`,
        { method: "POST" },
      );
    },
  };
}

function identityMatches(left, right) {
  if (!left || !right) return false;
  return ["vendor_id", "product_id", "name", "serial"]
    .every((field) => left[field] === right[field]);
}

function bindingText(binding) {
  if (!binding || binding.unsupported) return "未绑定";
  if (binding.source) {
    const direction = binding.direction ? ` ${binding.direction}` : "";
    return `${binding.source} ${binding.index}${direction}`;
  }
  if (binding.axis !== undefined) {
    return `axis ${binding.axis}${binding.invert ? " invert" : ""}`;
  }
  return JSON.stringify(binding);
}

function mappingInstruction(control) {
  if (["left_x", "right_x"].includes(control)) {
    return "请向右推动一次后回中。";
  }
  if (["left_y", "right_y"].includes(control)) {
    return "请向上推动一次后回中。";
  }
  if (["lt", "rt"].includes(control)) {
    return "请完整按下扳机并松开。";
  }
  if (control?.startsWith("dpad_")) {
    return "请按下对应方向并松开。";
  }
  return "请按下对应实体按钮后松开。";
}

function processText(error) {
  const processes = error?.details?.processes;
  if (!Array.isArray(processes) || processes.length === 0) {
    return error?.message ?? "请求失败。";
  }
  return processes.map((process) => {
    const argv = Array.isArray(process.argv) ? ` · ${process.argv.join(" ")}` : "";
    return `PID ${process.pid} · ${process.name}${argv}`;
  }).join("；");
}

export function createQuickMappingController({
  api,
  document,
  window,
  timers = {
    setInterval: (...args) => globalThis.setInterval(...args),
    clearInterval: (...args) => globalThis.clearInterval(...args),
  },
}) {
  const ids = [
    "process-interlock", "connection-status", "device-details", "mapping-grid",
    "mapping-guidance",
    "raw-axes", "raw-buttons", "raw-events", "logical-levels", "edge-events",
    "combo-preview", "save-profile", "profile-list", "profile-import",
    "profile-export", "profile-activate", "restart-banner",
    "status-announcements", "confirmation-dialog",
  ];
  const elements = Object.fromEntries(
    ids.map((id) => [id, document.getElementById(id)]),
  );
  const state = {
    devices: [],
    profiles: [],
    selectedProfileId: null,
    sessionId: null,
    connected: false,
    device: null,
    deviceCount: 0,
    expectedIdentity: null,
    mismatchDevice: null,
    armedControl: null,
    capture: IDLE_CAPTURE,
    bindings: {},
    missingRequired: [],
    logical: null,
    edges: null,
    axes: [],
    buttons: [],
    rawTransitions: [],
    pollTimer: null,
    pendingClose: null,
    refreshing: false,
    mounted: false,
  };

  const announce = (message) => {
    elements["status-announcements"].textContent = message;
  };

  const renderProcessError = (error) => {
    elements["process-interlock"].textContent = processText(error);
    announce(error?.message ?? "请求失败。");
  };

  const renderConnection = () => {
    const status = elements["connection-status"];
    const details = elements["device-details"];
    details.textContent = "";
    if (state.connected && state.device) {
      status.textContent = "已连接";
      const path = state.device.device_path?.split("/").at(-1) ?? "未知路径";
      const capability = state.device.capabilities ?? {};
      const count = state.deviceCount > 1
        ? `检测到 ${state.deviceCount} 个；当前使用 `
        : "";
      details.textContent = (
        `${count}${state.device.identity.name} · ${path} · ` +
        `${capability.axis_count} 轴 · ${capability.button_count} 按键`
      );
      return;
    }
    if (state.mismatchDevice) {
      status.textContent = "等待确认";
      details.textContent = (
        `检测到 ${state.mismatchDevice.identity.name}，设备身份不一致。`
      );
      const confirm = document.createElement("button");
      confirm.type = "button";
      confirm.className = "button primary";
      confirm.textContent = "使用新设备";
      confirm.addEventListener("click", async () => {
        state.expectedIdentity = null;
        state.mismatchDevice = null;
        await beginMonitoring(null);
      });
      details.append(confirm);
      return;
    }
    status.textContent = state.devices.length === 0
      ? "未检测到手柄"
      : "已断开";
    if (state.device) {
      details.textContent = `上次设备：${state.device.identity.name}`;
    }
  };

  const mappingCards = new Map();
  const renderMappings = () => {
    const cards = CONTROL_ORDER.map((control) => {
      let button = mappingCards.get(control);
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "mapping-card";
        button.dataset.control = control;
        button.addEventListener("click", async () => {
          await arm(control);
        });
        mappingCards.set(control, button);
      }
      button.dataset.active = String(state.armedControl === control);
      button.disabled = !state.connected || !state.sessionId;
      button.textContent = (
        `${CONTROL_LABELS[control]} · ${control} · ` +
        bindingText(state.bindings[control])
      );
      return button;
    });
    elements["mapping-grid"].replaceChildren(...cards);
  };

  const renderGuidance = () => {
    const capture = state.capture ?? IDLE_CAPTURE;
    const status = capture.status ?? "idle";
    const control = capture.control ?? state.armedControl;
    const label = CONTROL_LABELS[control] ?? control ?? "逻辑控制";
    let message = "点击一个 G1 逻辑控制开始映射。";

    if (status === "armed" && control) {
      message = `已选中${label}，${mappingInstruction(control)}`;
    } else if (status === "collecting") {
      message = "正在比较各轴位移，请完成动作并回中。";
    } else if (status === "ambiguous") {
      message = "检测到斜向输入，无法可靠区分主轴；请回中后沿目标方向重试。";
    } else if (status === "captured" && control) {
      const physical = capture.source && Number.isInteger(capture.index)
        ? `${capture.source} ${capture.index}`
        : "实体输入";
      const direction = capture.direction ? `，${capture.direction}` : "";
      message = `已识别${label}：${physical}${direction}。`;
    }

    elements["mapping-guidance"].dataset.state = status;
    elements["mapping-guidance"].textContent = message;
  };

  const renderRaw = () => {
    elements["raw-axes"].textContent = state.axes.length
      ? state.axes.map((value, index) => `axis ${index}: ${value}`).join(" · ")
      : "—";
    elements["raw-buttons"].textContent = state.buttons.length
      ? state.buttons.map((value, index) => `button ${index}: ${value}`).join(" · ")
      : "—";
    const rows = state.rawTransitions.map((transition) => {
      const item = document.createElement("li");
      item.textContent = (
        `${transition.time_ms} ms · ${transition.kind} ${transition.number} · ` +
        `${transition.old_value} → ${transition.new_value} · ${transition.phase}` +
        `${transition.initial ? " · initial" : ""}`
      );
      return item;
    });
    elements["raw-events"].replaceChildren(...rows);
  };

  const renderLogical = () => {
    if (!state.logical || !state.edges) {
      elements["logical-levels"].textContent = "—";
      elements["edge-events"].textContent = (
        "pressed: 无 · on_pressed: 无 · on_released: 无"
      );
      elements["combo-preview"].textContent = "—";
      return;
    }
    const logical = state.logical;
    const values = [
      ...Object.entries(logical.sticks ?? {}).map(
        ([name, value]) => `${name}: ${Number(value).toFixed(3)}`,
      ),
      ...Object.entries(logical.triggers ?? {}).map(
        ([name, value]) => `${name}: ${Number(value).toFixed(3)}`,
      ),
      ...Object.entries(logical.buttons ?? {}).map(
        ([name, value]) => `${name}: ${value ? "按下" : "松开"}`,
      ),
      ...Object.entries(logical.dpad ?? {}).map(
        ([name, value]) => `dpad_${name}: ${value ? "按下" : "松开"}`,
      ),
    ];
    elements["logical-levels"].textContent = values.join(" · ");
    const pressed = Object.entries(state.edges.pressed ?? {})
      .filter(([, value]) => value)
      .map(([name]) => name);
    const onPressed = state.edges.on_pressed ?? [];
    const onReleased = state.edges.on_released ?? [];
    elements["edge-events"].textContent = (
      `pressed: ${pressed.join(", ") || "无"} · ` +
      `on_pressed: ${onPressed.join(", ") || "无"} · ` +
      `on_released: ${onReleased.join(", ") || "无"}`
    );
    elements["combo-preview"].textContent = Object.entries(
      state.edges.combos ?? {},
    ).map(([name, value]) => `${name}: ${String(value)}`).join(" · ");
  };

  const renderProfiles = () => {
    const cards = state.profiles.map((stored) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "profile-card";
      button.dataset.selected = String(
        stored.profile_id === state.selectedProfileId,
      );
      button.textContent = stored.profile_id;
      button.addEventListener("click", () => {
        state.selectedProfileId = stored.profile_id;
        renderProfiles();
      });
      return button;
    });
    elements["profile-list"].replaceChildren(...cards);
    const missingProfile = !state.selectedProfileId;
    elements["profile-export"].disabled = missingProfile;
    elements["profile-activate"].disabled = missingProfile;
  };

  const renderSave = () => {
    elements["save-profile"].disabled = (
      !state.sessionId ||
      !state.connected ||
      state.missingRequired.length > 0
    );
  };

  const renderAll = () => {
    renderConnection();
    renderGuidance();
    renderMappings();
    renderRaw();
    renderLogical();
    renderProfiles();
    renderSave();
  };

  const applyEnvelope = (payload) => {
    state.sessionId = payload.session_id ?? state.sessionId;
    state.connected = Boolean(payload.connected);
    state.device = payload.device ?? state.device;
    state.deviceCount = payload.device_count ?? state.deviceCount;
    state.armedControl = payload.armed_control ?? null;
    state.capture = payload.capture ?? state.capture;
    state.bindings = payload.bindings ?? state.bindings;
    state.missingRequired = payload.missing_required ?? state.missingRequired;
    state.logical = payload.logical ?? state.logical;
    state.edges = payload.edges ?? state.edges;
    if (payload.raw) {
      state.axes = payload.raw.axes ?? [];
      state.buttons = payload.raw.buttons ?? [];
      state.rawTransitions = [
        ...state.rawTransitions,
        ...(payload.raw.transitions ?? []),
      ].slice(-80);
    }
    if (payload.replacement?.replaced_controls?.length) {
      announce(
        `${payload.replacement.replaced_controls.join("、")} 已被替换并设为未绑定。`,
      );
    }
    renderAll();
  };

  const stopPolling = () => {
    if (state.pollTimer !== null) {
      timers.clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  };

  const startPolling = () => {
    if (state.pollTimer !== null) return;
    state.pollTimer = timers.setInterval(() => refresh(), 1000);
  };

  const handleDisconnected = (payload) => {
    if (payload.bindings) state.bindings = payload.bindings;
    if (payload.missing_required) {
      state.missingRequired = payload.missing_required;
    }
    if (payload.raw) {
      state.axes = payload.raw.axes ?? [];
      state.buttons = payload.raw.buttons ?? [];
      state.rawTransitions = [
        ...state.rawTransitions,
        ...(payload.raw.transitions ?? []),
      ].slice(-80);
    }
    state.connected = false;
    state.armedControl = null;
    state.capture = payload.capture ?? IDLE_CAPTURE;
    state.logical = payload.logical ?? null;
    state.edges = payload.edges ?? null;
    const disconnectedSession = state.sessionId;
    state.sessionId = null;
    api.closeEvents();
    if (disconnectedSession) {
      state.pendingClose = Promise.resolve(api.cancel(disconnectedSession))
        .catch(renderProcessError);
    }
    startPolling();
    renderAll();
    announce("手柄已断开；保留当前绑定并等待同一设备重新连接。");
  };

  const handleEvent = ({ type, data }) => {
    if (type === "disconnected") {
      handleDisconnected(data);
      return;
    }
    applyEnvelope(data);
  };

  async function beginMonitoring(expectedDevice) {
    if (state.pendingClose) {
      await state.pendingClose;
      state.pendingClose = null;
    }
    try {
      const snapshot = await api.createQuickSession(expectedDevice);
      state.expectedIdentity = { ...snapshot.device.identity };
      state.mismatchDevice = null;
      applyEnvelope(snapshot);
      stopPolling();
      elements["process-interlock"].textContent = (
        "未检测到进程冲突；仅监测输入。"
      );
      api.connectQuickEvents(snapshot.session_id, handleEvent);
      return true;
    } catch (error) {
      renderProcessError(error);
      startPolling();
      return false;
    }
  }

  async function refreshProfiles() {
    const catalog = await api.loadCatalog();
    state.profiles = catalog.profiles;
    if (
      !state.profiles.some(
        (profile) => profile.profile_id === state.selectedProfileId,
      )
    ) {
      state.selectedProfileId = state.profiles[0]?.profile_id ?? null;
    }
    renderProfiles();
  }

  async function refresh() {
    if (state.refreshing) return false;
    state.refreshing = true;
    try {
      if (state.pendingClose) {
        await state.pendingClose;
        state.pendingClose = null;
      }
      const catalog = await api.loadCatalog();
      state.devices = catalog.devices;
      state.profiles = catalog.profiles;
      if (
        !state.profiles.some(
          (profile) => profile.profile_id === state.selectedProfileId,
        )
      ) {
        state.selectedProfileId = state.profiles[0]?.profile_id ?? null;
      }
      if (state.connected) {
        renderAll();
        return true;
      }
      if (state.devices.length === 0) {
        state.mismatchDevice = null;
        renderAll();
        startPolling();
        return false;
      }
      const first = state.devices[0];
      if (state.expectedIdentity) {
        const exactMatch = state.devices.find(
          (device) => identityMatches(device.identity, state.expectedIdentity),
        );
        if (!exactMatch) {
          state.mismatchDevice = first;
          renderAll();
          startPolling();
          return false;
        }
      }
      return beginMonitoring(state.expectedIdentity);
    } catch (error) {
      renderProcessError(error);
      startPolling();
      return false;
    } finally {
      state.refreshing = false;
    }
  }

  async function arm(control) {
    if (!state.sessionId || !state.connected) return false;
    try {
      const payload = await api.arm(state.sessionId, control);
      applyEnvelope(payload);
      return true;
    } catch (error) {
      renderProcessError(error);
      return false;
    }
  }

  async function save() {
    if (
      !state.sessionId ||
      state.missingRequired.length > 0 ||
      !state.connected
    ) return false;
    try {
      await api.save(state.sessionId);
      await refreshProfiles();
      elements["restart-banner"].hidden = false;
      announce("配置已保存；重启 MuJoCo 后生效。");
      return true;
    } catch (error) {
      renderProcessError(error);
      return false;
    }
  }

  async function importProfile(file) {
    if (!file) return false;
    try {
      await api.importProfile(await file.text());
      await refreshProfiles();
      announce("配置已导入。");
      return true;
    } catch (error) {
      renderProcessError(error);
      return false;
    }
  }

  async function exportProfile() {
    if (!state.selectedProfileId) return false;
    try {
      const blob = await api.exportProfile(state.selectedProfileId);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${state.selectedProfileId}.yaml`;
      await link.click?.();
      window.URL.revokeObjectURL(url);
      return true;
    } catch (error) {
      renderProcessError(error);
      return false;
    }
  }

  async function activateProfile() {
    if (!state.selectedProfileId) return false;
    try {
      await api.activateProfile(state.selectedProfileId);
      elements["restart-banner"].hidden = false;
      announce("配置已激活；重启 MuJoCo 后生效。");
      return true;
    } catch (error) {
      renderProcessError(error);
      return false;
    }
  }

  async function cancel() {
    stopPolling();
    api.closeEvents();
    if (state.pendingClose) {
      await state.pendingClose;
      state.pendingClose = null;
    }
    const sessionId = state.sessionId;
    state.sessionId = null;
    state.connected = false;
    state.armedControl = null;
    state.capture = IDLE_CAPTURE;
    if (sessionId) await api.cancel(sessionId);
    renderAll();
    return true;
  }

  const releaseForPageExit = () => {
    stopPolling();
    api.closeEvents();
    const sessionId = state.sessionId;
    state.sessionId = null;
    state.connected = false;
    state.armedControl = null;
    state.capture = IDLE_CAPTURE;
    if (sessionId) {
      Promise.resolve(api.cancel(sessionId)).catch(() => {
        // The browser owns keepalive completion after this page is gone.
      });
    }
  };

  async function mount() {
    if (!state.mounted) {
      state.mounted = true;
      window.addEventListener("pagehide", releaseForPageExit);
      elements["save-profile"].addEventListener("click", save);
      elements["profile-import"].addEventListener("change", (event) => (
        importProfile(event.target.files?.[0])
      ));
      elements["profile-export"].addEventListener("click", exportProfile);
      elements["profile-activate"].addEventListener("click", () => {
        if (!state.selectedProfileId) return;
        elements["confirmation-dialog"].showModal();
      });
      elements["confirmation-dialog"].addEventListener("close", () => {
        if (elements["confirmation-dialog"].returnValue === "activate") {
          return activateProfile();
        }
        return undefined;
      });
    }
    renderAll();
    return refresh();
  }

  const getState = () => ({
    devices: [...state.devices],
    profiles: [...state.profiles],
    selectedProfileId: state.selectedProfileId,
    sessionId: state.sessionId,
    connected: state.connected,
    device: state.device,
    deviceCount: state.deviceCount,
    expectedIdentity: state.expectedIdentity,
    armedControl: state.armedControl,
    capture: { ...state.capture },
    bindings: { ...state.bindings },
    missingRequired: [...state.missingRequired],
  });

  return { mount, refresh, arm, save, cancel, getState };
}

if (
  typeof document !== "undefined" &&
  typeof window !== "undefined" &&
  document.getElementById("mapping-grid")
) {
  const controller = createQuickMappingController({
    api: createQuickApiClient(),
    document,
    window,
  });
  controller.mount().catch((error) => {
    document.getElementById("status-announcements").textContent = error.message;
  });
}

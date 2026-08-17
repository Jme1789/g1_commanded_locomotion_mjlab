import assert from "node:assert/strict";
import test from "node:test";

import {
  createQuickApiClient,
  createQuickMappingController,
} from "../../src/gamepad_calibrator/static/quick_app.js";

const CONTROL_ORDER = [
  "left_x", "left_y", "right_x", "right_y", "lt", "rt",
  "dpad_up", "dpad_down", "dpad_left", "dpad_right",
  "a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick",
];
const DEVICE = {
  device_path: "/dev/input/js7",
  by_id_path: "/dev/input/by-id/usb-Test_Pad-joystick",
  identity: {
    vendor_id: "046d",
    product_id: "c216",
    name: "Test Pad",
    serial: "serial-1",
  },
  capabilities: { axis_count: 6, button_count: 14 },
};
const OTHER_DEVICE = {
  ...DEVICE,
  device_path: "/dev/input/js8",
  by_id_path: "/dev/input/by-id/usb-Other_Pad-joystick",
  identity: {
    vendor_id: "1234",
    product_id: "5678",
    name: "Other Pad",
    serial: "serial-2",
  },
};
const PROFILE = {
  profile_id: "046d_c216_test-pad-serial-1",
  profile: { schema_version: 1, device: DEVICE.identity },
};
const EMPTY_LOGICAL = {
  sticks: { left_x: 0, left_y: 0, right_x: 0, right_y: 0 },
  triggers: { lt: 0, rt: 0 },
  buttons: {
    a: false, b: false, x: false, y: false, lb: false, rb: false,
    start: false, back: false, left_stick: false, right_stick: false,
  },
  dpad: { up: false, down: false, left: false, right: false },
};
const EMPTY_EDGES = {
  pressed: Object.fromEntries(
    ["lt", "rt", "dpad_up", "dpad_down", "dpad_left", "dpad_right",
      "a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick"]
      .map((control) => [control, false]),
  ),
  on_pressed: [],
  on_released: [],
  combos: { lt_up: false, rt_a: false, lb: false, rb: false },
};
const EMPTY_CAPTURE = {
  status: "idle",
  control: null,
  source: null,
  index: null,
  direction: null,
  primary_axis: null,
  secondary_axis: null,
};
const QUICK_SESSION = {
  session_id: "quick-1",
  state: "monitoring",
  connected: true,
  device_count: 1,
  device: DEVICE,
  armed_control: null,
  capture: EMPTY_CAPTURE,
  bindings: Object.fromEntries(
    CONTROL_ORDER.map((control) => [control, { unsupported: true }]),
  ),
  missing_required: ["dpad_up"],
  logical: EMPTY_LOGICAL,
  edges: EMPTY_EDGES,
  raw: {
    axes: [0, 0, 0, 0, 0, 0],
    buttons: Array(14).fill(0),
    transitions: [],
  },
  replacement: null,
};

class FixtureHTMLCollection {
  constructor(owner) { this.owner = owner; }
  get length() { return this.owner._children.length; }
  item(index) { return this.owner._children[index] ?? null; }
  [Symbol.iterator]() { return this.owner._children[Symbol.iterator](); }
}

class FixtureElement {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = ownerDocument;
    this._children = [];
    this.children = new FixtureHTMLCollection(this);
    this.dataset = {};
    this.listeners = new Map();
    this._textContent = "";
    this.parentElement = null;
    this.hidden = false;
    this.disabled = false;
    this.files = [];
    this.value = "";
    this.type = this.tagName === "BUTTON" ? "button" : "";
    this.className = "";
  }
  get textContent() {
    return this._textContent + this._children.map((child) => child.textContent).join("");
  }
  set textContent(value) {
    this._textContent = String(value);
    for (const child of this._children) child.parentElement = null;
    this._children = [];
  }
  get lastElementChild() { return this._children.at(-1) ?? null; }
  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) ?? [];
    callbacks.push(callback);
    this.listeners.set(type, callbacks);
  }
  async dispatch(type, extra = {}) {
    if (type === "click" && this.disabled) return;
    const event = {
      preventDefault() {},
      target: this,
      currentTarget: this,
      ...extra,
    };
    for (const callback of this.listeners.get(type) ?? []) await callback(event);
  }
  async click() {
    if (this.tagName === "A") {
      this.ownerDocument.downloads.push({
        href: this.href,
        download: this.download,
      });
    }
    await this.dispatch("click");
  }
  append(...children) {
    for (const child of children) child.parentElement = this;
    this._children.push(...children);
  }
  replaceChildren(...children) {
    for (const child of this._children) child.parentElement = null;
    for (const child of children) child.parentElement = this;
    this._children = children;
  }
  setAttribute(name, value) {
    if (name.startsWith("data-")) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      this.dataset[key] = String(value);
    } else {
      this[name] = String(value);
    }
  }
  removeAttribute(name) { delete this[name]; }
  focus() { this.ownerDocument.activeElement = this; }
  querySelectorAll(selector) {
    const found = [];
    const visit = (node) => {
      for (const child of node._children) {
        if (selector === child.tagName.toLowerCase()) found.push(child);
        visit(child);
      }
    };
    visit(this);
    return found;
  }
}

class FixtureDialog extends FixtureElement {
  constructor(tagName, ownerDocument) {
    super(tagName, ownerDocument);
    this.open = false;
    this.returnValue = "";
  }
  showModal() { this.open = true; }
  close(returnValue = "") {
    this.returnValue = returnValue;
    this.open = false;
  }
}

const MARKUP_TAGS = {
  "app-header": "header",
  "process-interlock": "section",
  "connection-status": "span",
  "device-details": "div",
  "mapping-grid": "div",
  "mapping-guidance": "div",
  "raw-monitor": "section",
  "raw-axes": "div",
  "raw-buttons": "div",
  "raw-events": "ol",
  "logical-preview": "section",
  "logical-levels": "div",
  "edge-events": "div",
  "combo-preview": "div",
  "save-profile": "button",
  "profile-list": "div",
  "profile-import": "input",
  "profile-export": "button",
  "profile-activate": "button",
  "restart-banner": "aside",
  "status-announcements": "div",
};

class FixtureDocument {
  constructor() {
    this.nodes = new Map(
      Object.entries(MARKUP_TAGS).map(
        ([id, tag]) => [id, new FixtureElement(tag, this)],
      ),
    );
    this.nodes.set(
      "confirmation-dialog",
      new FixtureDialog("dialog", this),
    );
    this.root = new FixtureElement("body", this);
    this.root.append(...this.nodes.values());
    this.nodes.get("profile-import").type = "file";
    this.nodes.get("save-profile").disabled = true;
    this.downloads = [];
    this.activeElement = null;
  }
  getElementById(id) { return this.nodes.get(id) ?? null; }
  createElement(tagName) { return new FixtureElement(tagName, this); }
  querySelectorAll(selector) {
    return this.root.querySelectorAll(selector);
  }
}

class FakeEventSource {
  constructor(url = "") {
    this.url = url;
    this.listeners = new Map();
    this.closed = false;
  }
  addEventListener(type, callback) {
    this.listeners.set(type, callback);
  }
  emit(type, data, lastEventId = "1") {
    this.listeners.get(type)?.({
      data: JSON.stringify(data),
      lastEventId,
    });
  }
  close() { this.closed = true; }
}

function response(body, { status = 200, contentType = "application/json" } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get(name) {
        return name.toLowerCase() === "content-type" ? contentType : null;
      },
    },
    async json() { return body; },
    async text() {
      return typeof body === "string" ? body : JSON.stringify(body);
    },
    async blob() { return body; },
  };
}

function processConflict(pid, name) {
  const error = new Error("检测到阻塞进程。");
  error.code = "process_conflict";
  error.details = {
    processes: [{ pid, name, argv: [`/opt/${name}`, "--run"] }],
  };
  return error;
}

function createTimers() {
  const timers = {
    entries: new Map(),
    nextId: 1,
    setInterval(callback, milliseconds) {
      const id = this.nextId++;
      this.entries.set(id, { callback, milliseconds });
      return id;
    },
    clearInterval(id) { this.entries.delete(id); },
    async tick() {
      for (const { callback } of [...this.entries.values()]) await callback();
    },
  };
  return {
    setInterval: timers.setInterval.bind(timers),
    clearInterval: timers.clearInterval.bind(timers),
    tick: timers.tick.bind(timers),
    entries: timers.entries,
  };
}

function createWindowFixture() {
  const window = {
    createdUrls: [],
    revokedUrls: [],
    listeners: new Map(),
    URL: {
      createObjectURL(blob) {
        const url = `blob:test-${blob.yaml.length}`;
        window.createdUrls.push(url);
        return url;
      },
      revokeObjectURL(url) { window.revokedUrls.push(url); },
    },
    addEventListener(type, callback) {
      const callbacks = this.listeners.get(type) ?? [];
      callbacks.push(callback);
      this.listeners.set(type, callbacks);
    },
    async dispatch(type) {
      for (const callback of this.listeners.get(type) ?? []) {
        await callback({ type });
      }
    },
  };
  return window;
}

function createFakeApi({ devices = [DEVICE], profiles = [PROFILE] } = {}) {
  const boundary = {
    devices,
    profiles,
    quickCreates: [],
    arms: [],
    saves: [],
    cancels: [],
    imports: [],
    exports: [],
    activations: [],
    profilesLoads: 0,
    source: null,
    closedEvents: 0,
    failQuickCreate: null,
    exportedBlob: { yaml: "schema_version: 1\n" },
    session: { ...QUICK_SESSION, device: devices[0] ?? DEVICE },
  };
  boundary.api = {
    async loadCatalog() {
      boundary.profilesLoads += 1;
      return { devices: boundary.devices, profiles: boundary.profiles };
    },
    async createQuickSession(expectedDevice) {
      boundary.quickCreates.push(expectedDevice);
      if (boundary.failQuickCreate) throw boundary.failQuickCreate;
      const selectedDevice = expectedDevice
        ? boundary.devices.find((device) => (
          device.identity.vendor_id === expectedDevice.vendor_id &&
          device.identity.product_id === expectedDevice.product_id &&
          device.identity.name === expectedDevice.name &&
          device.identity.serial === expectedDevice.serial
        ))
        : boundary.devices[0];
      boundary.session = {
        ...boundary.session,
        device_count: boundary.devices.length,
        device: selectedDevice,
      };
      return boundary.session;
    },
    connectQuickEvents(sessionId, onEvent) {
      boundary.api.closeEvents();
      const source = new FakeEventSource(
        `/api/v1/quick-sessions/${sessionId}/events`,
      );
      source.emit = (type, data, lastEventId = "1") => {
        onEvent({ type, data, lastEventId });
      };
      boundary.source = source;
      return source;
    },
    closeEvents() {
      if (boundary.source && !boundary.source.closed) {
        boundary.source.close();
        boundary.closedEvents += 1;
      }
    },
    async arm(sessionId, control) {
      boundary.arms.push({ sessionId, control });
      return boundary.armResponse ?? {
        ...boundary.session,
        armed_control: control,
      };
    },
    async save(sessionId) {
      boundary.saves.push(sessionId);
      return PROFILE;
    },
    async cancel(sessionId) { boundary.cancels.push(sessionId); },
    async importProfile(yamlText) {
      boundary.imports.push(yamlText);
      return PROFILE;
    },
    async exportProfile(profileId) {
      boundary.exports.push(profileId);
      return boundary.exportedBlob;
    },
    async activateProfile(profileId) {
      boundary.activations.push(profileId);
      return {
        schema_version: 1,
        profile: `profiles/${profileId}.yaml`,
        device: DEVICE.identity,
      };
    },
  };
  return boundary;
}

async function mountedQuickController(options = {}) {
  const document = new FixtureDocument();
  const boundary = createFakeApi(options);
  const window = createWindowFixture();
  const timers = createTimers();
  const controller = createQuickMappingController({
    api: boundary.api,
    document,
    window,
    timers,
  });
  await controller.mount();
  return { controller, document, boundary, window, timers };
}

async function settle() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

function mappingCard(document, control) {
  return [...document.getElementById("mapping-grid").children]
    .find((element) => element.dataset.control === control);
}

test("quick API client exposes only mapping/profile operations and closes replaced SSE", async () => {
  const requests = [];
  const sources = [];
  const fetchImpl = async (url, options = {}) => {
    requests.push({ url, options });
    if (url === "/api/v1/devices") return response({ devices: [DEVICE] });
    if (url === "/api/v1/profiles") {
      return response({ profiles: [PROFILE], templates: [] });
    }
    if (url.endsWith("/export")) {
      return response({ yaml: "x" }, { contentType: "application/yaml" });
    }
    if (options.method === "DELETE") return response(null, { status: 204 });
    return response(
      url.endsWith("/save") || url.endsWith("/import") ? PROFILE : QUICK_SESSION,
      { status: options.method === "POST" ? 201 : 200 },
    );
  };
  const api = createQuickApiClient(
    fetchImpl,
    (url) => {
      const source = new FakeEventSource(url);
      sources.push(source);
      return source;
    },
  );

  assert.deepEqual(Object.keys(api).sort(), [
    "activateProfile", "arm", "cancel", "closeEvents", "connectQuickEvents",
    "createQuickSession", "exportProfile", "importProfile", "loadCatalog", "save",
  ]);
  assert.deepEqual(await api.loadCatalog(), {
    devices: [DEVICE],
    profiles: [PROFILE],
  });
  await api.createQuickSession(null);
  const received = [];
  api.connectQuickEvents("quick-1", (event) => received.push(event.type));
  api.connectQuickEvents("quick-1", (event) => received.push(event.type));
  assert.equal(sources[0].closed, true);
  for (const type of ["snapshot", "binding", "state", "disconnected"]) {
    sources[1].emit(type, QUICK_SESSION);
  }
  assert.deepEqual(received, ["snapshot", "binding", "state", "disconnected"]);
  await api.arm("quick-1", "dpad_up");
  await api.save("quick-1");
  await api.cancel("quick-1");
  await api.importProfile("schema_version: 1\n");
  await api.exportProfile(PROFILE.profile_id);
  await api.activateProfile(PROFILE.profile_id);

  const create = requests.find(({ url }) => url === "/api/v1/quick-sessions");
  assert.deepEqual(JSON.parse(create.options.body), { expected_device: null });
  assert.ok(requests.some(({ url }) => url.endsWith("/arm/dpad_up")));
  assert.ok(requests.some(({ url }) => url.endsWith("/quick-1/save")));
  const cancel = requests.find(({ url }) => url.endsWith("/quick-1"));
  assert.equal(cancel.options.method, "DELETE");
  assert.equal(cancel.options.keepalive, true);
  assert.equal(
    requests.some(({ url }) => /kill|terminate|stop|mujoco|g1_ctrl|dds|robot|velocity|joint/i.test(url)),
    false,
  );
});

test("default browser timers retain their global Window receiver", async () => {
  const originalSetInterval = globalThis.setInterval;
  const originalClearInterval = globalThis.clearInterval;
  const receivers = [];
  globalThis.setInterval = function setIntervalForBrowserTest() {
    receivers.push(["set", this]);
    return 41;
  };
  globalThis.clearInterval = function clearIntervalForBrowserTest() {
    receivers.push(["clear", this]);
  };
  try {
    const document = new FixtureDocument();
    const boundary = createFakeApi({ devices: [] });
    const controller = createQuickMappingController({
      api: boundary.api,
      document,
      window: createWindowFixture(),
    });

    await controller.mount();
    await controller.cancel();

    assert.deepEqual(receivers, [
      ["set", globalThis],
      ["clear", globalThis],
    ]);
  } finally {
    globalThis.setInterval = originalSetInterval;
    globalThis.clearInterval = originalClearInterval;
  }
});

test("one connected device auto-starts monitoring without a selector", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });

  assert.equal(
    mounted.document.getElementById("connection-status").textContent,
    "已连接",
  );
  assert.match(
    mounted.document.getElementById("device-details").textContent,
    /Test Pad.*js7.*6 轴.*14 按键/,
  );
  assert.equal(mounted.boundary.quickCreates.length, 1);
  assert.equal(
    mounted.document.querySelectorAll("select").length,
    0,
  );
});

test("zero devices polls and later auto-starts the first connected device", async () => {
  const mounted = await mountedQuickController({ devices: [] });
  assert.equal(
    mounted.document.getElementById("connection-status").textContent,
    "未检测到手柄",
  );
  assert.equal(mounted.boundary.quickCreates.length, 0);
  assert.equal([...mounted.timers.entries.values()][0].milliseconds, 1000);

  mounted.boundary.devices = [DEVICE];
  await mounted.timers.tick();
  assert.equal(mounted.boundary.quickCreates.length, 1);
  assert.equal(
    mounted.document.getElementById("connection-status").textContent,
    "已连接",
  );
});

test("multiple devices report the count and monitor the first without controls for choosing", async () => {
  const mounted = await mountedQuickController({
    devices: [DEVICE, OTHER_DEVICE],
  });

  assert.match(
    mounted.document.getElementById("device-details").textContent,
    /检测到 2 个.*Test Pad.*js7/,
  );
  assert.equal(mounted.boundary.quickCreates.length, 1);
  assert.equal(mounted.boundary.quickCreates[0], null);
  assert.equal(mounted.document.querySelectorAll("select").length, 0);
  assert.equal(mounted.document.querySelectorAll("input").length, 1);
});

test("repeated raw frames stay visible but backend edges render once", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    raw: {
      axes: [0, 0, 0, 0, 0, 0],
      buttons: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      transitions: [
        {
          time_ms: 10,
          kind: "button",
          number: 0,
          old_value: 0,
          new_value: 1,
          initial: false,
          phase: "pressed",
        },
        {
          time_ms: 11,
          kind: "button",
          number: 0,
          old_value: 1,
          new_value: 1,
          initial: false,
          phase: "repeat",
        },
      ],
    },
    edges: {
      pressed: { a: true },
      on_pressed: ["a"],
      on_released: [],
      combos: { lt_up: false, rt_a: false, lb: false, rb: false },
    },
  });

  assert.match(
    mounted.document.getElementById("raw-events").textContent,
    /button 0.*pressed.*button 0.*repeat/,
  );
  assert.match(
    mounted.document.getElementById("edge-events").textContent,
    /on_pressed: a/,
  );

  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    raw: {
      ...QUICK_SESSION.raw,
      transitions: [{
        time_ms: 12, kind: "button", number: 1, old_value: 0,
        new_value: 1, initial: false, phase: "pressed",
      }],
    },
    edges: { ...EMPTY_EDGES, on_pressed: [], on_released: [] },
  });
  assert.doesNotMatch(
    mounted.document.getElementById("edge-events").textContent,
    /on_pressed: b/,
  );
});

test("raw monitor retains only the newest eighty backend transitions", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    raw: {
      ...QUICK_SESSION.raw,
      transitions: Array.from({ length: 81 }, (_, number) => ({
        time_ms: number,
        kind: "button",
        number,
        old_value: 0,
        new_value: 1,
        initial: false,
        phase: "pressed",
      })),
    },
  });
  const text = mounted.document.getElementById("raw-events").textContent;
  assert.doesNotMatch(text, /button 0\b/);
  assert.match(text, /button 1\b/);
  assert.match(text, /button 80\b/);
});

test("mapping click arms dpad_up and renders only the backend-returned binding", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.armResponse = {
    ...QUICK_SESSION,
    armed_control: "dpad_up",
    bindings: {
      ...QUICK_SESSION.bindings,
      dpad_up: { source: "button", index: 12 },
    },
  };

  await mappingCard(mounted.document, "dpad_up").click();
  assert.deepEqual(
    mounted.boundary.arms,
    [{ sessionId: "quick-1", control: "dpad_up" }],
  );
  assert.match(
    mappingCard(mounted.document, "dpad_up").textContent,
    /dpad_up.*button 12/,
  );
  assert.equal(mappingCard(mounted.document, "dpad_up").dataset.active, "true");
  assert.notEqual(mappingCard(mounted.document, "a").dataset.active, "true");
});

test("live snapshots preserve mapping card nodes so physical clicks survive", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  const originalCard = mappingCard(mounted.document, "left_x");

  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    raw: {
      ...QUICK_SESSION.raw,
      axes: [12000, 2000, 0, 0, 0, 0],
    },
  });

  assert.equal(
    mappingCard(mounted.document, "left_x") === originalCard,
    true,
  );
  await originalCard.click();
  assert.deepEqual(
    mounted.boundary.arms,
    [{ sessionId: "quick-1", control: "left_x" }],
  );
});

test("dominant-axis guidance stays visible through ambiguous capture and success", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.armResponse = {
    ...QUICK_SESSION,
    armed_control: "left_x",
    capture: {
      ...EMPTY_CAPTURE,
      status: "armed",
      control: "left_x",
    },
  };

  await mappingCard(mounted.document, "left_x").click();
  const guidance = mounted.document.getElementById("mapping-guidance");
  assert.match(guidance.textContent, /已选中左摇杆 X.*向右推动一次后回中/);
  assert.equal(guidance.dataset.state, "armed");
  assert.equal(mappingCard(mounted.document, "left_x").dataset.active, "true");

  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    armed_control: "left_x",
    capture: {
      ...EMPTY_CAPTURE,
      status: "collecting",
      control: "left_x",
      primary_axis: 0,
      secondary_axis: 1,
    },
  });
  assert.match(guidance.textContent, /正在比较各轴位移/);
  assert.equal(guidance.dataset.state, "collecting");

  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    armed_control: "left_x",
    capture: {
      ...EMPTY_CAPTURE,
      status: "ambiguous",
      control: "left_x",
      primary_axis: 0,
      secondary_axis: 1,
    },
  });
  assert.match(guidance.textContent, /检测到斜向输入/);
  assert.match(guidance.textContent, /请回中后沿目标方向重试/);
  assert.equal(guidance.dataset.state, "ambiguous");
  assert.equal(mappingCard(mounted.document, "left_x").dataset.active, "true");

  mounted.boundary.source.emit("binding", {
    ...QUICK_SESSION,
    capture: {
      ...EMPTY_CAPTURE,
      status: "captured",
      control: "left_x",
      source: "axis",
      index: 0,
      direction: "positive",
      primary_axis: 0,
      secondary_axis: 1,
    },
    bindings: {
      ...QUICK_SESSION.bindings,
      left_x: { source: "axis", index: 0, direction: "positive" },
    },
  });
  assert.match(guidance.textContent, /已识别左摇杆 X.*axis 0.*positive/);
  assert.equal(guidance.dataset.state, "captured");
  assert.notEqual(mappingCard(mounted.document, "left_x").dataset.active, "true");
});

test("button mapping click immediately explains the required press and release", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.armResponse = {
    ...QUICK_SESSION,
    armed_control: "a",
    capture: {
      ...EMPTY_CAPTURE,
      status: "armed",
      control: "a",
    },
  };

  await mappingCard(mounted.document, "a").click();

  const guidance = mounted.document.getElementById("mapping-guidance");
  assert.match(guidance.textContent, /已选中A/);
  assert.match(guidance.textContent, /请按下对应实体按钮后松开/);
  assert.equal(guidance.dataset.state, "armed");
  assert.equal(mappingCard(mounted.document, "a").dataset.active, "true");
});

test("backend replacement marks every displaced control unbound", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.source.emit("binding", {
    ...QUICK_SESSION,
    bindings: {
      ...QUICK_SESSION.bindings,
      a: { source: "button", index: 0 },
      b: { unsupported: true },
    },
    replacement: { replaced_controls: ["b"] },
  });

  assert.match(mappingCard(mounted.document, "a").textContent, /button 0/);
  assert.match(mappingCard(mounted.document, "b").textContent, /未绑定/);
  assert.match(
    mounted.document.getElementById("status-announcements").textContent,
    /b.*未绑定/,
  );
});

test("disconnect clears logical pressed and combos but keeps bindings and closes SSE", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  const bound = {
    ...QUICK_SESSION,
    bindings: {
      ...QUICK_SESSION.bindings,
      a: { source: "button", index: 0 },
    },
    logical: {
      ...EMPTY_LOGICAL,
      buttons: { ...EMPTY_LOGICAL.buttons, a: true },
    },
    edges: {
      ...EMPTY_EDGES,
      pressed: { ...EMPTY_EDGES.pressed, a: true },
      combos: { ...EMPTY_EDGES.combos, rt_a: true },
    },
  };
  mounted.boundary.source.emit("snapshot", bound);
  const source = mounted.boundary.source;
  mounted.boundary.source.emit("disconnected", {
    ...bound,
    connected: false,
    state: "disconnected",
    logical: EMPTY_LOGICAL,
    edges: {
      ...EMPTY_EDGES, on_released: ["a"],
    },
  });

  assert.equal(source.closed, true);
  assert.match(mappingCard(mounted.document, "a").textContent, /button 0/);
  assert.doesNotMatch(
    mounted.document.getElementById("logical-levels").textContent,
    /按下|true/,
  );
  assert.doesNotMatch(
    mounted.document.getElementById("combo-preview").textContent,
    /rt_a: true|rt_a: 激活/,
  );
  assert.equal(
    mounted.document.getElementById("edge-events").textContent
      .match(/on_released: a/g)?.length,
    1,
  );
});

test("a different identity after disconnect requires explicit use-new-device confirmation", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.source.emit("disconnected", {
    ...QUICK_SESSION,
    connected: false,
    state: "disconnected",
  });
  mounted.boundary.devices = [OTHER_DEVICE];
  await mounted.timers.tick();

  assert.equal(mounted.boundary.quickCreates.length, 1);
  assert.match(
    mounted.document.getElementById("device-details").textContent,
    /身份不一致/,
  );
  const confirmations = [
    ...mounted.document.getElementById("device-details").children,
  ].filter((child) => child.textContent === "使用新设备");
  assert.equal(confirmations.length, 1);

  await confirmations[0].click();
  assert.equal(mounted.boundary.quickCreates.length, 2);
  assert.equal(mounted.boundary.quickCreates[1], null);
});

test("reconnect finds the original identity when it is not the first catalog device", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.source.emit("disconnected", {
    ...QUICK_SESSION,
    connected: false,
    state: "disconnected",
    logical: EMPTY_LOGICAL,
    edges: EMPTY_EDGES,
  });
  mounted.boundary.devices = [OTHER_DEVICE, DEVICE];

  await mounted.timers.tick();

  assert.equal(mounted.boundary.quickCreates.length, 2);
  assert.deepEqual(mounted.boundary.quickCreates[1], DEVICE.identity);
  assert.equal(
    mounted.document.getElementById("connection-status").textContent,
    "已连接",
  );
  assert.match(
    mounted.document.getElementById("device-details").textContent,
    /Test Pad/,
  );
});

test("save stays disabled until complete then saves and refreshes profiles", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  assert.equal(mounted.document.getElementById("save-profile").disabled, true);

  mounted.boundary.source.emit("snapshot", {
    ...QUICK_SESSION,
    missing_required: [],
  });
  assert.equal(mounted.document.getElementById("save-profile").disabled, false);
  await mounted.document.getElementById("save-profile").click();

  assert.deepEqual(mounted.boundary.saves, ["quick-1"]);
  assert.equal(mounted.boundary.profilesLoads, 2);
  assert.equal(mounted.document.getElementById("restart-banner").hidden, false);
});

test("profile import export and activation retain browser outcomes and confirmation", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  const input = mounted.document.getElementById("profile-import");
  input.files = [{
    async text() { return "schema_version: 1\ndevice: {}\n"; },
  }];
  await input.dispatch("change");
  assert.deepEqual(
    mounted.boundary.imports,
    ["schema_version: 1\ndevice: {}\n"],
  );
  assert.equal(mounted.boundary.profilesLoads, 2);

  await mounted.document.getElementById("profile-export").click();
  assert.deepEqual(mounted.boundary.exports, [PROFILE.profile_id]);
  assert.deepEqual(mounted.document.downloads, [{
    href: "blob:test-18",
    download: `${PROFILE.profile_id}.yaml`,
  }]);
  assert.deepEqual(mounted.window.revokedUrls, ["blob:test-18"]);

  await mounted.document.getElementById("profile-activate").click();
  const dialog = mounted.document.getElementById("confirmation-dialog");
  assert.equal(dialog.open, true);
  dialog.returnValue = "activate";
  dialog.open = false;
  await dialog.dispatch("close");
  assert.deepEqual(mounted.boundary.activations, [PROFILE.profile_id]);
  assert.equal(mounted.document.getElementById("restart-banner").hidden, false);
});

test("process conflicts show PID and name without process-control actions", async () => {
  const document = new FixtureDocument();
  const boundary = createFakeApi({ devices: [DEVICE] });
  boundary.failQuickCreate = processConflict(4451, "unitree_mujoco");
  const controller = createQuickMappingController({
    api: boundary.api,
    document,
    window: createWindowFixture(),
    timers: createTimers(),
  });
  await controller.mount();

  assert.match(
    document.getElementById("process-interlock").textContent,
    /PID 4451.*unitree_mujoco/,
  );
  assert.equal(
    Object.keys(boundary.api).some(
      (name) => /kill|terminate|start|stop|restart|mujoco|robot/i.test(name),
    ),
    false,
  );
});

test("cancel closes SSE deletes the quick session and clears polling", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  mounted.boundary.source.emit("disconnected", {
    ...QUICK_SESSION,
    connected: false,
    state: "disconnected",
  });
  assert.ok(mounted.timers.entries.size > 0);
  const source = mounted.boundary.source;

  await mounted.controller.cancel();

  assert.equal(source.closed, true);
  assert.deepEqual(mounted.boundary.cancels, ["quick-1"]);
  assert.equal(mounted.timers.entries.size, 0);
  assert.equal(mounted.controller.getState().sessionId, null);
});

test("pagehide closes SSE and releases the active quick session", async () => {
  const mounted = await mountedQuickController({ devices: [DEVICE] });
  const source = mounted.boundary.source;

  await mounted.window.dispatch("pagehide");

  assert.equal(source.closed, true);
  assert.deepEqual(mounted.boundary.cancels, ["quick-1"]);
});

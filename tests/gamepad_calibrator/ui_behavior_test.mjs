import assert from "node:assert/strict";
import test from "node:test";

import {
  bindingFormModel,
  createApiClient,
  createCalibrationController,
  renderLogicalPreview,
} from "../../src/gamepad_calibrator/static/app.js";

const CONTROL_ORDER = [
  "left_x", "left_y", "right_x", "right_y", "lt", "rt",
  "dpad_up", "dpad_down", "dpad_left", "dpad_right",
  "a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick",
];
const DEVICE = {
  device_path: "/dev/input/js7",
  by_id_path: "/dev/input/by-id/usb-Test_Pad-joystick",
  identity: { vendor_id: "046d", product_id: "c216", name: "Test Pad", serial: "serial-1" },
  capabilities: { axis_count: 6, button_count: 14 },
};
const TEMPLATE = {
  template_id: "complete-pad",
  schema_version: 1,
  template_name: "漂亮手柄模板",
  sticks: Object.fromEntries(["left_x", "left_y", "right_x", "right_y"].map((name, axis) => [name, { axis, center: 0, min: -32767, max: 32767, invert: name.endsWith("_y"), deadzone: 0.05 }])),
  triggers: {
    lt: { source: "axis", index: 4, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null },
    rt: { source: "axis", index: 5, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null },
  },
  buttons: Object.fromEntries(["a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick"].map((name, index) => [name, name === "x" ? { unsupported: true } : { source: "button", index }])),
  dpad: Object.fromEntries(["up", "down", "left", "right"].map((name, offset) => [name, { source: "button", index: 10 + offset }])),
};
const PROFILE = {
  profile_id: "046d_c216_test-pad-serial-1",
  profile: {
    schema_version: 1,
    device: DEVICE.identity,
    sticks: TEMPLATE.sticks,
    triggers: TEMPLATE.triggers,
    buttons: TEMPLATE.buttons,
    dpad: TEMPLATE.dpad,
  },
};
const SESSION = { session_id: "session-1", state: "neutral", connected: true, candidate: null, device: DEVICE };
const PRIMARY_CANDIDATE_BINDING = { axis: 0, center: 0, min: -32767, max: 32767, invert: false, deadzone: 0.05 };
const ALTERNATE_CANDIDATE_BINDING = { axis: 1, center: 0, min: -32767, max: 32767, invert: false, deadzone: 0.05 };

function ambiguousSnapshot() {
  return {
    session_id: "session-1", state: "capturing", connected: true,
    candidate: {
      control: "left_x", binding: PRIMARY_CANDIDATE_BINDING,
      score: 32000, ambiguous_with: [ALTERNATE_CANDIDATE_BINDING],
    },
    raw: {
      axes: [24576, 16384, 0, 0, -32767, -32767], buttons: Array(14).fill(0),
      events: [{ time_ms: 123, kind: "axis", number: 0, value: 24576 }],
    },
  };
}

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
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this.type = this.tagName === "BUTTON" ? "button" : "";
    this.name = "";
    this.files = [];
    this.listeners = new Map();
    this._textContent = "";
    this.parentElement = null;
  }
  get textContent() { return this._textContent + this._children.map((child) => child.textContent).join(""); }
  set textContent(value) {
    this._textContent = String(value);
    for (const child of this._children) child.parentElement = null;
    this._children = [];
  }
  get lastElementChild() { return this._children.at(-1) ?? null; }
  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) ?? [];
    callbacks.push(callback); this.listeners.set(type, callbacks);
  }
  async dispatch(type, extra = {}) {
    if (type === "click" && this.disabled) return;
    const event = { preventDefault() {}, target: this, currentTarget: this, ...extra };
    for (const callback of this.listeners.get(type) ?? []) await callback(event);
  }
  append(...children) {
    for (const child of children) child.parentElement = this;
    this._children.push(...children);
  }
  replaceChildren(...children) {
    for (const child of this._children) child.parentElement = null;
    for (const child of children) child.parentElement = this;
    this._children = children;
    if (this.tagName === "SELECT" && !children.some((item) => item.value === this.value)) this.value = children[0]?.value ?? "";
  }
  focus() { this.ownerDocument.activeElement = this; }
  setAttribute(name, value) { this[name] = String(value); }
  async click() {
    if (this.tagName === "A") this.ownerDocument.downloads.push({ href: this.href, download: this.download });
    await this.dispatch("click");
  }
}

class FixtureDialog extends FixtureElement {
  showModal() { this.open = true; }
  close() { this.open = false; }
}

const MARKUP_TAGS = {
  "app-header": "header", "process-interlock": "section", "neutral-step": "section", "guided-step": "section",
  "raw-monitor": "article", "candidate-panel": "article", "manual-editor": "section", "logical-preview": "section",
  "save-panel": "section", "preview-checklist": "fieldset", "manual-confirm": "button",
  "blocker-list": "div", "device-list": "div", "profile-list": "div", "template-select": "select",
  "start-session": "button", "refresh-devices": "button", "session-state": "span", "neutral-progress": "progress",
  "neutral-status": "p", "control-select": "select", "begin-step": "button", "confirm-step": "button",
  "redo-step": "button", "unsupported-step": "button", "confirmed-controls": "ol", "raw-axes": "div",
  "raw-buttons": "div", "raw-events": "ol", "candidate-details": "dl", "candidate-choices": "div",
  "binding-kind": "select", "manual-fields": "div", "manual-form": "form", "preview-values": "div",
  "preview-confirmations": "div", "refresh-preview": "button", "save-profile": "button", "cancel-session": "button",
  "profile-import": "input", "profile-export": "button", "profile-activate": "button", "restart-banner": "aside",
  "status-announcements": "div", "dialog-cancel": "button", "dialog-confirm": "button",
};

class FixtureDocument {
  constructor() {
    this.nodes = new Map(Object.entries(MARKUP_TAGS).map(([id, tag]) => [id, new FixtureElement(tag, this)]));
    this.nodes.set("confirmation-dialog", new FixtureDialog("dialog", this));
    this.nodes.get("profile-import").type = "file";
    this.nodes.get("save-profile").disabled = true;
    this.nodes.get("template-select").replaceChildren(this.option("", "不使用模板"));
    this.nodes.get("binding-kind").replaceChildren(
      this.option("button", "按键"), this.option("stick", "摇杆轴"),
      this.option("axis-trigger", "轴扳机"), this.option("button-trigger", "按键扳机"),
      this.option("axis-dpad", "方向轴"),
    );
    this.nodes.get("manual-form").append(
      this.nodes.get("binding-kind"),
      this.nodes.get("manual-fields"),
      this.nodes.get("manual-confirm"),
    );
    this.nodes.get("process-interlock").append(
      this.nodes.get("blocker-list"), this.nodes.get("refresh-devices"),
    );
    this.nodes.get("guided-step").append(
      this.nodes.get("control-select"), this.nodes.get("begin-step"),
      this.nodes.get("confirm-step"), this.nodes.get("redo-step"),
      this.nodes.get("unsupported-step"), this.nodes.get("confirmed-controls"),
    );
    this.nodes.get("candidate-panel").append(
      this.nodes.get("candidate-details"), this.nodes.get("candidate-choices"),
    );
    this.nodes.get("preview-checklist").append(this.nodes.get("preview-confirmations"));
    this.nodes.get("logical-preview").append(
      this.nodes.get("preview-values"), this.nodes.get("refresh-preview"),
      this.nodes.get("preview-checklist"),
    );
    this.nodes.get("save-panel").append(
      this.nodes.get("cancel-session"), this.nodes.get("save-profile"),
    );
    this.nodes.get("confirmation-dialog").append(
      this.nodes.get("dialog-cancel"), this.nodes.get("dialog-confirm"),
    );
    this.downloads = [];
    this.activeElement = null;
  }
  getElementById(id) { return this.nodes.get(id) ?? null; }
  createElement(tagName) { return new FixtureElement(tagName, this); }
  option(value, label) { const item = this.createElement("option"); item.value = value; item.textContent = label; return item; }
}

class FakeEventSource {
  constructor(url) { this.url = url; this.listeners = new Map(); this.closed = false; }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  emit(type, data, lastEventId = "1") { this.listeners.get(type)?.({ data: JSON.stringify(data), lastEventId }); }
  close() { this.closed = true; }
}

function response(body, { status = 200, contentType = "application/json" } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => name.toLowerCase() === "content-type" ? contentType : null },
    async json() { return body; },
    async text() { return typeof body === "string" ? body : JSON.stringify(body); },
    async blob() { return body; },
  };
}

function processConflict(pid, name) {
  return response({ code: "process_conflict", message_zh: "检测到阻塞进程。", field_path: null, details: { processes: [{ pid, name, argv: [`/opt/${name}`, "--run"] }] } }, { status: 409 });
}

function createExternalBoundary() {
  const boundary = {
    requests: [], sources: [], failNextCreate: false, failNextSave: false, failNextActivate: false,
    importedYaml: null, exportedBlob: { yaml: "schema_version: 1\n" }, profilesLoads: 0,
  };
  boundary.fetch = async (url, options = {}) => {
    boundary.requests.push({ url, options });
    if (url === "/api/v1/devices") return response({ devices: [DEVICE] });
    if (url === "/api/v1/profiles") { boundary.profilesLoads += 1; return response({ profiles: [PROFILE], templates: [TEMPLATE] }); }
    if (url === "/api/v1/sessions" && options.method === "POST") {
      if (boundary.failNextCreate) { boundary.failNextCreate = false; return processConflict(4451, "unitree_mujoco"); }
      return response(SESSION, { status: 201 });
    }
    if (url.endsWith("/confirm")) return response({ ...SESSION, state: "review", device: null });
    if (url.endsWith("/redo") || url.endsWith("/unsupported") || /\/steps\//.test(url)) return response({ ...SESSION, state: "review", device: null });
    if (url.endsWith("/preview")) return response({ sticks: { left_x: 1 }, triggers: { lt: 0 }, buttons: { a: false }, dpad: { up: false } });
    if (url.endsWith("/save")) {
      if (boundary.failNextSave) { boundary.failNextSave = false; return processConflict(5510, "g1_ctrl"); }
      return response(PROFILE, { status: 201 });
    }
    if (options.method === "DELETE") return response(null, { status: 204 });
    if (url === "/api/v1/profiles/import") { boundary.importedYaml = options.body; return response(PROFILE, { status: 201 }); }
    if (url.endsWith("/export")) return response(boundary.exportedBlob, { contentType: "application/yaml" });
    if (url.endsWith("/activate")) {
      if (boundary.failNextActivate) { boundary.failNextActivate = false; return processConflict(6610, "unitree_mujoco"); }
      return response({ schema_version: 1, profile: `profiles/${PROFILE.profile_id}.yaml`, device: DEVICE.identity });
    }
    throw new Error(`Unhandled request ${options.method ?? "GET"} ${url}`);
  };
  boundary.eventSourceFactory = (url) => { const source = new FakeEventSource(url); boundary.sources.push(source); return source; };
  return boundary;
}

function createWindowFixture() {
  return {
    createdUrls: [], revokedUrls: [],
    URL: {
      createObjectURL(blob) { const url = `blob:test-${blob.yaml.length}`; this.owner.createdUrls.push(url); return url; },
      revokeObjectURL(url) { this.owner.revokedUrls.push(url); },
      owner: null,
    },
  };
}

async function mountedController() {
  const document = new FixtureDocument();
  const boundary = createExternalBoundary();
  const window = createWindowFixture(); window.URL.owner = window;
  const api = createApiClient(boundary.fetch, boundary.eventSourceFactory);
  const controller = createCalibrationController({ api, document, window });
  await controller.mount();
  return { controller, document, boundary, window };
}

async function settle() { await new Promise((resolve) => setTimeout(resolve, 0)); }
function requests(boundary, suffix) { return boundary.requests.filter(({ url }) => url.endsWith(suffix)); }

async function mountedAmbiguousSession() {
  const mounted = await mountedController();
  await mounted.document.getElementById("start-session").click(); await settle();
  mounted.boundary.sources[0].emit("snapshot", ambiguousSnapshot(), "8");
  return mounted;
}

test("API client closes an old event stream before replacement and sends template filename ID", async () => {
  const boundary = createExternalBoundary();
  const api = createApiClient(boundary.fetch, boundary.eventSourceFactory);
  const catalog = await api.loadCatalog();
  await api.createSession("/dev/input/js7", catalog.templates[0].template_id);
  api.connectEvents("session-1", () => {}); api.connectEvents("session-1", () => {});

  assert.equal(boundary.sources[0].closed, true);
  assert.equal(boundary.sources[1].url, "/api/v1/sessions/session-1/events");
  assert.deepEqual(JSON.parse(requests(boundary, "/sessions")[0].options.body), { device_path: "/dev/input/js7", template_id: "complete-pad" });
});

test("binding model and logical preview render normalized physical values", () => {
  const model = bindingFormModel("left_x", { binding: TEMPLATE.sticks.left_x }, DEVICE.capabilities);
  assert.equal(model.kind, "stick");
  assert.deepEqual(model.toBinding({ axis: "3", center: "0", min: "-32767", max: "32767", invert: true, deadzone: "0.06" }), { axis: 3, center: 0, min: -32767, max: 32767, invert: true, deadzone: 0.06 });
  const document = new FixtureDocument(); const root = document.createElement("div");
  renderLogicalPreview(root, { sticks: { left_x: 0.25 }, triggers: { lt: 0.75 }, buttons: { a: true }, dpad: { up: false } });
  assert.deepEqual([...root.children].map((item) => item.textContent), ["left_x: 0.250", "lt: 0.750", "a: 按下", "up: 松开"]);
});

test("mounted template selection uses ID and keeps all mappings provisional until preview", async () => {
  const { controller, document, boundary } = await mountedController();
  const selector = document.getElementById("template-select");
  assert.equal(selector.children.item(1).value, "complete-pad");
  assert.equal(selector.children.item(1).textContent, "漂亮手柄模板");
  selector.value = "complete-pad";
  await document.getElementById("start-session").click(); await settle();

  const state = controller.getState();
  assert.deepEqual(state.resolvedControls, CONTROL_ORDER);
  assert.equal(state.supportedControls.includes("x"), false);
  assert.equal(document.getElementById("preview-confirmations").children.length, 19);
  assert.equal(document.getElementById("save-profile").disabled, true);
  for (const label of document.getElementById("preview-confirmations").children) {
    const input = label.children.item(0);
    input.checked = true; await input.dispatch("change");
  }
  assert.equal(document.getElementById("save-profile").disabled, false);
  await document.getElementById("save-profile").click(); await settle();
  const payload = JSON.parse(requests(boundary, "/save")[0].options.body);
  assert.equal(payload.preview_confirmations.length, 19);
  assert.equal(payload.preview_confirmations.includes("x"), false);
});

test("mounted manual kind submits button trigger and one resolved control cannot enable save", async () => {
  const { document, boundary } = await mountedController();
  await document.getElementById("start-session").click(); await settle();
  const control = document.getElementById("control-select"); control.value = "lt"; await control.dispatch("change");
  const kind = document.getElementById("binding-kind");
  assert.equal(kind.value, "axis-trigger");
  kind.value = "button-trigger"; await kind.dispatch("change");
  assert.equal(document.getElementById("manual-fields").children.length, 1);
  document.getElementById("manual-fields").children.item(0).lastElementChild.value = "7";
  await document.getElementById("manual-form").dispatch("submit"); await settle();

  const sent = JSON.parse(requests(boundary, "/confirm")[0].options.body);
  assert.deepEqual(sent.binding_override, { source: "button", index: 7, threshold: 0.5 });
  const previewInput = document.getElementById("preview-confirmations").children.item(0).children.item(0);
  previewInput.checked = true; await previewInput.dispatch("change");
  assert.equal(document.getElementById("save-profile").disabled, true);
});

test("mounted reconnect preserves resolved and preview-confirmed UI while closing the old stream", async () => {
  const { controller, document, boundary } = await mountedAmbiguousSession();
  await document.getElementById("candidate-choices").children.item(0).click();
  await document.getElementById("confirm-step").click(); await settle();
  const previewInput = document.getElementById("preview-confirmations").children.item(0).children.item(0);
  previewInput.checked = true; await previewInput.dispatch("change");

  const oldSource = boundary.sources[0];
  controller.reconnectEvents();
  assert.equal(oldSource.closed, true);
  assert.equal(boundary.sources.length, 2);
  boundary.sources[1].emit("snapshot", { ...ambiguousSnapshot(), candidate: null }, "9");

  assert.deepEqual(controller.getState().resolvedControls, ["left_x"]);
  assert.deepEqual(controller.getState().previewConfirmations, ["left_x"]);
  assert.match(document.getElementById("confirmed-controls").textContent, /left_x/);
  assert.equal(document.getElementById("preview-confirmations").children.item(0).children.item(0).checked, true);
  assert.equal(document.getElementById("session-state").textContent, "capturing");
});

test("mounted ambiguous candidate requires an explicit choice and submits the chosen override", async () => {
  const { controller, document, boundary } = await mountedAmbiguousSession();
  await document.getElementById("confirm-step").click(); await settle();

  assert.equal(requests(boundary, "/confirm").length, 0);
  assert.match(document.getElementById("status-announcements").textContent, /候选存在歧义.*明确选择候选/);
  assert.equal(document.getElementById("candidate-choices").children.length, 2);

  await document.getElementById("candidate-choices").children.item(1).click();
  await document.getElementById("confirm-step").click(); await settle();
  const payload = JSON.parse(requests(boundary, "/confirm")[0].options.body);
  assert.deepEqual(payload, { binding_override: ALTERNATE_CANDIDATE_BINDING });
  assert.deepEqual(controller.getState().resolvedControls, ["left_x"]);
  assert.match(document.getElementById("confirmed-controls").textContent, /left_x/);
  assert.equal(document.getElementById("preview-confirmations").children.length, 1);
});

test("mounted cancel deletes the session, closes events, and clears calibration UI without process actions", async () => {
  const { controller, document, boundary } = await mountedAmbiguousSession();
  await document.getElementById("candidate-choices").children.item(0).click();
  await document.getElementById("confirm-step").click(); await settle();
  const previewInput = document.getElementById("preview-confirmations").children.item(0).children.item(0);
  previewInput.checked = true; await previewInput.dispatch("change");
  const source = boundary.sources[0];

  await document.getElementById("cancel-session").click(); await settle();

  const deletes = boundary.requests.filter(({ url, options }) => url === "/api/v1/sessions/session-1" && options.method === "DELETE");
  assert.equal(deletes.length, 1);
  assert.equal(source.closed, true);
  assert.equal(controller.getState().sessionId, null);
  assert.deepEqual(controller.getState().resolvedControls, []);
  assert.deepEqual(controller.getState().supportedControls, []);
  assert.deepEqual(controller.getState().previewConfirmations, []);
  assert.equal(document.getElementById("confirmed-controls").children.length, 0);
  assert.equal(document.getElementById("preview-confirmations").children.length, 0);
  assert.equal(document.getElementById("session-state").textContent, "已取消");
  assert.equal(boundary.requests.some(({ url }) => /kill|terminate|stop/.test(url)), false);
});

test("mounted blocker refresh clears stale display and retries authoritative create", async () => {
  const { document, boundary } = await mountedController(); boundary.failNextCreate = true;
  await document.getElementById("start-session").click(); await settle();
  assert.match(document.getElementById("blocker-list").textContent, /PID 4451.*unitree_mujoco.*--run/);
  assert.equal(document.getElementById("start-session").disabled, true);
  await document.getElementById("refresh-devices").click(); await settle();
  assert.equal(document.getElementById("start-session").disabled, false);
  await document.getElementById("start-session").click(); await settle();
  assert.equal(requests(boundary, "/sessions").length, 2);
});

test("save and activation conflicts share blocker rendering without process actions", async () => {
  const { document, boundary } = await mountedController();
  document.getElementById("template-select").value = "complete-pad";
  await document.getElementById("start-session").click(); await settle();
  for (const label of document.getElementById("preview-confirmations").children) {
    const input = label.children.item(0); input.checked = true; await input.dispatch("change");
  }
  boundary.failNextSave = true; await document.getElementById("save-profile").click(); await settle();
  assert.match(document.getElementById("blocker-list").textContent, /PID 5510.*g1_ctrl/);
  await document.getElementById("refresh-devices").click(); await settle();
  boundary.failNextActivate = true;
  await document.getElementById("profile-activate").click();
  assert.equal(document.getElementById("confirmation-dialog").open, true);
  await document.getElementById("dialog-confirm").click(); await settle();
  assert.match(document.getElementById("blocker-list").textContent, /PID 6610.*unitree_mujoco/);
  assert.equal(boundary.requests.some(({ url }) => /kill|terminate|stop/.test(url)), false);
});

test("mounted import export and activation dialog expose browser outcomes", async () => {
  const { document, boundary, window } = await mountedController();
  const fileInput = document.getElementById("profile-import");
  fileInput.files = [{ async text() { return "schema_version: 1\ndevice: {}\n"; } }];
  await fileInput.dispatch("change"); await settle();
  assert.equal(boundary.importedYaml, "schema_version: 1\ndevice: {}\n");
  assert.ok(boundary.profilesLoads >= 2);

  await document.getElementById("profile-export").click(); await settle();
  assert.deepEqual(document.downloads, [{ href: "blob:test-18", download: `${PROFILE.profile_id}.yaml` }]);
  assert.deepEqual(window.revokedUrls, ["blob:test-18"]);

  await document.getElementById("profile-activate").click();
  assert.equal(document.getElementById("confirmation-dialog").open, true);
  assert.equal(document.activeElement, document.getElementById("dialog-confirm"));
  await document.getElementById("dialog-confirm").click(); await settle();
  assert.equal(document.getElementById("confirmation-dialog").open, false);
  assert.equal(document.getElementById("restart-banner").hidden, false);
});

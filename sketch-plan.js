(() => {
  "use strict";

  const distance = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);

  function segmentDistance(point, start, end) {
    const dx = end.x - start.x, dy = end.y - start.y;
    const length2 = dx * dx + dy * dy;
    const t = length2 ? Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / length2)) : 0;
    return Math.hypot(point.x - start.x - t * dx, point.y - start.y - t * dy);
  }

  function simplify(points, tolerance) {
    let furthest = 0, index = 0;
    for (let i = 1; i < points.length - 1; i++) {
      const error = segmentDistance(points[i], points[0], points.at(-1));
      if (error > furthest) { index = i; furthest = error; }
    }
    if (furthest <= tolerance) return [points[0], points.at(-1)];
    return [...simplify(points.slice(0, index + 1), tolerance).slice(0, -1), ...simplify(points.slice(index), tolerance)];
  }

  function closeGaps(mask, width, height, radius) {
    let input = Uint8Array.from(mask, (pixel) => Number(pixel === 3));
    for (const dilate of [true, false]) {
      for (const horizontal of [true, false]) {
        const output = new Uint8Array(input.length);
        const span = horizontal ? width : height, lanes = horizontal ? height : width;
        const offset = horizontal ? 1 : width, pitch = horizontal ? width : 1;
        for (let lane = 0; lane < lanes; lane++) {
          const start = lane * pitch;
          let total = 0;
          for (let i = 0; i <= radius; i++) total += input[start + i * offset];
          for (let i = 0; i < span; i++) {
            output[start + i * offset] = Number(dilate ? total > 0 : total === radius * 2 + 1);
            if (i - radius >= 0) total -= input[start + (i - radius) * offset];
            if (i + radius + 1 < span) total += input[start + (i + radius + 1) * offset];
          }
        }
        input = output;
      }
    }
    return input;
  }

  function traceInterior(outline, stride) {
    const brush = Math.max(2, Math.min(8, Math.round(Math.min(outline.w, outline.h) * .015)));
    const padding = brush + 2;
    const width = outline.w + padding * 2, height = outline.h + padding * 2;
    if (width * height > 1800000) return null;
    // Use only the detected boundary ink. Filling from outside leaves the enclosed region;
    // taking its perimeter avoids the exterior ends of handwritten tick marks.
    const mask = new Uint8Array(width * height);
    for (const pixel of outline.pixels) {
      const x = pixel % stride - outline.left + padding;
      const y = Math.floor(pixel / stride) - outline.top + padding;
      for (let dy = -brush; dy <= brush; dy++) {
        const reach = Math.round(Math.sqrt(brush * brush - dy * dy));
        mask.fill(1, (y + dy) * width + x - reach, (y + dy) * width + x + reach + 1);
      }
    }
    const queue = new Int32Array(mask.length);
    const fill = (start, tag) => {
      queue[0] = start;
      mask[start] = tag;
      let count = 1;
      for (let i = 0; i < count; i++) {
        const p = queue[i], x = p % width;
        for (const n of [p - width, p + width, x ? p - 1 : -1, x + 1 < width ? p + 1 : -1]) {
          if (n >= 0 && n < mask.length && mask[n] === 0) {
            mask[n] = tag;
            queue[count++] = n;
          }
        }
      }
      return count;
    };
    fill(0, 2);
    let start = -1;
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i] && fill(i, 3) > mask.length * .26) { start = i; break; }
    }
    if (start < 0) return null;
    const clean = closeGaps(mask, width, height, Math.round(Math.min(outline.w, outline.h) * .036));
    start = clean.indexOf(1);
    if (start < 0) return null;
    // Walk the interior's grid edges clockwise, keeping the filled cell to the right.
    const origin = { x: start % width, y: Math.floor(start / width) };
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const isInside = (x, y) => clean[y * width + x] === 1;
    const rightCell = [[0, 0], [-1, 0], [-1, -1], [0, -1]];
    const leftCell = [[0, -1], [0, 0], [-1, 0], [-1, -1]];
    let { x, y } = origin;
    let direction = 0;
    const points = [];
    do {
      points.push({ x: outline.left + x - padding, y: outline.top + y - padding });
      const candidates = [1, 0, 3, 2].map((turn) => (direction + turn) % 4);
      direction = candidates.find((index) => {
        const [rx, ry] = rightCell[index], [lx, ly] = leftCell[index];
        return isInside(x + rx, y + ry) && !isInside(x + lx, y + ly);
      });
      if (direction === undefined || points.length > 4 * mask.length) return null;
      x += directions[direction][0];
      y += directions[direction][1];
    } while (x !== origin.x || y !== origin.y);
    let opposite = 1;
    points.forEach((point, index) => { if (distance(point, points[0]) > distance(points[opposite], points[0])) opposite = index; });
    const tolerance = Math.min(outline.w, outline.h) * .04;
    const polygon = [...simplify(points.slice(0, opposite + 1), tolerance).slice(0, -1),
      ...simplify([...points.slice(opposite), points[0]], tolerance).slice(0, -1)];
    return polygon.filter((point, i) => segmentDistance(point, polygon[(i + polygon.length - 1) % polygon.length],
      polygon[(i + 1) % polygon.length]) > tolerance * .68);
  }

  function edgeKind(a, b) {
    const x = Math.abs(b.x - a.x), y = Math.abs(b.y - a.y);
    if (y < x * .24) return "horizontal";
    if (x < y * .24) return "vertical";
    return "diagonal";
  }

  function matchLabels(edges, marks) {
    const choices = [];
    for (const mark of marks) {
      const { box } = mark;
      const center = { x: box.x + box.w / 2, y: box.y + box.h / 2 };
      for (const edge of edges) {
        const d = segmentDistance(center, edge.a, edge.b);
        const midpoint = { x: (edge.a.x + edge.b.x) / 2, y: (edge.a.y + edge.b.y) / 2 };
        const nearest = Math.min(...[box.x, center.x, box.x + box.w].flatMap((x) =>
          [box.y, center.y, box.y + box.h].map((y) => segmentDistance({ x, y }, edge.a, edge.b))));
        const inchMark = /[″“”"]$/.test(mark.text ?? "");
        const allowance = Math.min(inchMark ? 48 : 100, Math.max(36, edge.length * .6 + 12));
        if (nearest <= allowance) choices.push({ mark, edge, score: d + .62 * distance(center, midpoint) });
      }
    }
    const used = new Set();
    for (const { mark, edge } of choices.sort((a, b) => a.score - b.score)) {
      if (edge.mark || used.has(mark.number)) continue;
      edge.mark = mark.number;
      used.add(mark.number);
    }
  }

  function mainRectanglePlan(edges) {
    if (edges.length !== 8) return null;
    const top = edges.filter((edge) => edge.kind === "horizontal")
      .sort((a, b) => (a.a.y + a.b.y) - (b.a.y + b.b.y))[0];
    if (!top) return null;
    const ordered = edges.map((_, index) => edges[(top.index + index) % edges.length]);
    const pattern = ["horizontal", "vertical", "horizontal", "diagonal", "vertical", "horizontal", "vertical", "diagonal"];
    if (!ordered.every((edge, index) => edge.kind === pattern[index])) return null;
    const [cap, upper, notch, rightSlope, lower, foot, left, leftSlope] = ordered;
    const span = Math.abs(foot.b.x - foot.a.x);
    if (!(cap.b.x > cap.a.x && foot.b.x < foot.a.x && notch.b.x > notch.a.x &&
        rightSlope.b.x < rightSlope.a.x && leftSlope.b.x > leftSlope.a.x &&
        cap.a.x > foot.b.x && cap.b.y < foot.a.y &&
        Math.abs(lower.b.x - upper.a.x) < span * .18)) return null;
    return { cap, upper, notch, rightSlope, lower, foot, left, leftSlope };
  }

  function solve(values) {
    const { cap, upper, notch, rightSlope, lower, foot, left, leftSlope } = values;
    if (cap >= foot) return { error: "顶边应短于底边，请核对这两条边。" };
    const run = foot - cap;
    if (leftSlope <= run) return { error: "左斜边必须长于左上角的水平差，请核对左斜边、顶边和底边。" };
    if (rightSlope <= notch) return { error: "右斜边必须长于凸出段的水平边，请核对这两条边。" };
    const leftRise = Math.sqrt(leftSlope ** 2 - run ** 2);
    const rightRise = Math.sqrt(rightSlope ** 2 - notch ** 2);
    const leftTotal = left + leftRise;
    const rightTotal = upper + lower + rightRise;
    const tolerance = .005; // metres; avoid false conflicts from unit/measurement rounding.
    if (Math.abs(leftTotal - rightTotal) > tolerance) {
      return { conflict: true, leftTotal, rightTotal };
    }
    const total = (leftTotal + rightTotal) / 2;
    return { parts: [
      { title: "A 主矩形", area: foot * total },
      { title: "B 左上三角形", area: -run * leftRise / 2 },
      { title: "C 右侧三角形", area: notch * rightRise / 2 }
    ] };
  }

  const fields = [
    ["foot", "底边"], ["cap", "顶边"], ["left", "左侧直边"], ["leftSlope", "左斜边"],
    ["upper", "凹口上方直边"], ["notch", "凸出段水平边"], ["rightSlope", "右斜边"], ["lower", "右下直边"]
  ];

  function drawDiagram(canvas) {
    const context = canvas.getContext("2d");
    const shape = (points, color) => {
      context.beginPath();
      points.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
      context.closePath();
      context.fillStyle = color;
      context.fill();
      context.strokeStyle = "#183c31";
      context.lineWidth = 2;
      context.stroke();
    };
    context.clearRect(0, 0, canvas.width, canvas.height);
    shape([[55, 27], [210, 27], [210, 240], [55, 240]], "#ceebdf");
    shape([[55, 27], [118, 27], [55, 107]], "#f9dec2");
    shape([[210, 92], [261, 92], [210, 154]], "#cde0f7");
    context.font = "bold 21px sans-serif";
    context.fillStyle = "#183c31";
    context.fillText("A", 125, 150);
    context.fillText("B −", 66, 54);
    context.fillText("C +", 214, 115);
  }

  function setup({ validate, apply, depth, onModeChange }) {
    const get = (id) => document.getElementById("plan-" + id);
    const units = window.concreteUnits;
    const controls = new Map();
    let edges, marks = [], ready;
    const metres = (text) => {
      const value = text.trim();
      if (!value) return null;
      const mm = units.parse(/^(?:\d+(?:\.\d+)?|\.\d+)$/.test(value) ? value + " m" : value);
      return mm > 0 && mm <= 1e7 ? mm / 1000 : null;
    };
    const metric = (value) => units.asInput(value, 3);

    function update() {
      ready = undefined;
      get("calculate").disabled = true;
      get("results").replaceChildren();
      if (!edges || get("panel").hidden) return;
      const values = {};
      const missing = [];
      for (const [key, label] of fields) {
        const control = controls.get(key);
        const number = edges[key].mark;
        const mark = marks.find((candidate) => candidate.number === number);
        if (!control.edited) control.input.value = mark?.confirmed ? mark.text : "";
        values[key] = metres(control.input.value);
        const resolved = Boolean(values[key]);
        const list = resolved ? get("known") : get("missing");
        if (control.row.parentElement !== list) list.append(control.row);
        control.source.textContent = control.edited ? "手动填写" :
          mark?.confirmed ? "已带入标注 " + number :
          mark ? "确认标注 " + number + "，或直接填写。" : "请补填实测尺寸。";
        control.input.setAttribute("aria-invalid", String(Boolean(control.input.value) && !resolved));
        if (!resolved) missing.push(label);
      }
      get("known-section").hidden = !get("known").children.length;
      const needDepth = Boolean(validate("depth", depth.value));
      if (needDepth) missing.push("厚度");
      get("accept").disabled = true;
      if (missing.length > Number(needDepth)) {
        get("status").textContent = "待补或确认：" + missing.join("、") + "。";
        return;
      }
      const result = solve(values);
      if (result.error || result.conflict) {
        get("known-section").open = true;
        get("status").textContent = result.error ||
          "尺寸无法闭合：左侧推得总高约 " + metric(result.leftTotal) + " m，右侧约 " +
          metric(result.rightTotal) + " m。请以实测总高为参照，核对两条斜边及相邻边长。";
        return;
      }
      const area = result.parts.reduce((sum, part) => sum + part.area, 0);
      const formatted = units.asInput(area, 6);
      if (validate("area", formatted)) {
        get("status").textContent = "分块面积超出有效范围，请核对尺寸。";
        return;
      }
      for (const part of result.parts) {
        const item = document.createElement("li");
        item.textContent = part.title + "：" + (part.area < 0 ? "−" : "+") + metric(Math.abs(part.area)) + " m²";
        get("results").append(item);
      }
      get("status").textContent = needDepth ? "请补填厚度。" :
        "合计面积 " + metric(area) + " m²。请确认建议分法与施工图一致。";
      ready = needDepth ? undefined : formatted;
      get("accept").disabled = !ready;
      get("calculate").disabled = !ready || !get("accept").checked;
    }
    for (const [key, label] of fields) {
      const row = document.createElement("label");
      row.append(label);
      const input = document.createElement("input");
      input.autocomplete = "off";
      input.placeholder = "例如 8′ 或 2.44 m";
      const source = document.createElement("span");
      source.className = "hint";
      row.append(input, source);
      const control = { row, input, source, edited: false };
      controls.set(key, control);
      input.addEventListener("input", () => {
        control.edited = true;
        ready = undefined;
        get("accept").checked = false;
        get("accept").disabled = get("calculate").disabled = true;
      });
      input.addEventListener("change", update);
    }
    get("accept").addEventListener("change", update);
    get("manual").addEventListener("click", () => {
      get("panel").hidden = true;
      onModeChange(false);
    });
    get("open").addEventListener("click", () => {
      if (!edges) return;
      get("panel").hidden = false;
      onModeChange(true);
      update();
    });
    get("calculate").addEventListener("click", () => {
      if (!ready || !get("accept").checked) return;
      const volume = apply("area", { area: ready, depth: depth.value });
      get("status").textContent = volume === "—" ? "已填入分块面积，请检查数量。" : "计算结果 " + volume + " m³。";
    });
    drawDiagram(get("diagram"));
    return {
      update,
      showEntry(active, irregular) { get("open").hidden = active || !irregular || !edges; },
      setMarks(next) { marks = next; get("accept").checked = false; update(); },
      show(plan, confirmedMarks) {
        edges = plan;
        marks = confirmedMarks;
        get("open").hidden = !edges;
        get("panel").hidden = !edges;
        onModeChange(Boolean(edges));
        update();
      },
      clear() {
        edges = ready = undefined;
        marks = [];
        get("open").hidden = get("panel").hidden = true;
        get("accept").checked = false;
        get("known-section").open = false;
        for (const control of controls.values()) { control.edited = false; control.input.value = ""; }
        onModeChange(false);
      }
    };
  }

  window.sketchPlan = {
    analyze(outline, stride, marks) {
      if (!outline) return null;
      const vertices = traceInterior(outline, stride);
      if (!vertices || vertices.length < 3 || vertices.length > 12) return null;
      const edges = vertices.map((a, index) => {
        const b = vertices[(index + 1) % vertices.length];
        return { a, b, index, kind: edgeKind(a, b), length: distance(a, b) };
      });
      matchLabels(edges, marks);
      return mainRectanglePlan(edges);
    },
    setup
  };
})();

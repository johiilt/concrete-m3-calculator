(() => {
  "use strict";

  const instructions = [
    "点选卷尺的第一个刻度。",
    "点选卷尺的第二个刻度。",
    "点选浇筑区第一个角。",
    "沿长度方向点选第二个角。",
    "继续沿边缘点选第三个角。",
    "沿边缘点选最后一个角。"
  ];
  const markers = ["尺1", "尺2", "角1", "角2", "角3", "角4"];
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  function estimate(points, referenceMetres) {
    const [firstMark, lastMark, ...corners] = points;
    const edges = corners.map((corner, i) => {
      const next = corners[(i + 1) % 4];
      return { x: next.x - corner.x, y: next.y - corner.y };
    });
    const lengths = edges.map(({ x, y }) => Math.hypot(x, y));
    const turns = edges.map((edge, i) => {
      const next = edges[(i + 1) % 4];
      const divisor = lengths[i] * lengths[(i + 1) % 4];
      return {
        cross: edge.x * next.y - edge.y * next.x,
        cosine: Math.abs((edge.x * next.x + edge.y * next.y) / divisor)
      };
    });
    if (Math.min(...lengths) < 12 || !(turns.every(({ cross }) => cross > 0) || turns.every(({ cross }) => cross < 0))) {
      return { error: "请按浇筑区边缘顺序点选四角，四条边不能交叉。" };
    }
    const oppositeRatio = (a, b) => Math.max(a, b) / Math.min(a, b);
    if (turns.some(({ cosine }) => cosine > 0.34) ||
        oppositeRatio(lengths[0], lengths[2]) > 1.25 ||
        oppositeRatio(lengths[1], lengths[3]) > 1.25) {
      return { error: "四角看起来偏斜。请核对标点，或从更接近正上方的位置重拍。" };
    }
    const pixels = distance(firstMark, lastMark);
    if (pixels < Math.max(16, Math.min(...lengths) * 0.06)) {
      return { error: "卷尺两点在照片中太近，请选相距更远的刻度。" };
    }
    const scale = referenceMetres / pixels;
    return {
      length: (lengths[0] + lengths[2]) / 2 * scale,
      width: (lengths[1] + lengths[3]) / 2 * scale
    };
  }

  window.setupPhotoMeasure = ({ form, validate, apply }) => {
    const get = (id) => document.getElementById("measure-" + id);
    const panel = get("panel");
    const workspace = get("workspace");
    const canvas = get("canvas");
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const units = window.concreteUnits;
    const points = [];
    const reference = get("reference");
    const depth = get("depth");
    const status = get("status");
    const step = get("step");
    const result = get("result");
    const undo = get("undo");
    const zoom = get("zoom");
    let image;
    let imageRequest = 0;
    let cursor;
    let measurement;

    function draw() {
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const scale = canvas.width / canvas.getBoundingClientRect().width;
      function path(vertices, color, close = false) {
        if (vertices.length < 2) return;
        context.beginPath();
        context.moveTo(vertices[0].x, vertices[0].y);
        for (const vertex of vertices.slice(1)) context.lineTo(vertex.x, vertex.y);
        if (close) context.closePath();
        context.lineWidth = 4 * scale;
        context.strokeStyle = "#14251c";
        context.stroke();
        context.lineWidth = 2 * scale;
        context.strokeStyle = color;
        context.stroke();
      }
      path(points.slice(0, 2), "#ffce4d");
      path(points.slice(2), "#a9ffc0", points.length === 6);
      points.forEach(({ x, y }, i) => {
        context.beginPath();
        context.arc(x, y, 5 * scale, 0, 2 * Math.PI);
        context.fillStyle = i < 2 ? "#ffce4d" : "#a9ffc0";
        context.fill();
        context.lineWidth = 1.5 * scale;
        context.strokeStyle = "#14251c";
        context.stroke();
        context.font = "bold " + 12 * scale + "px sans-serif";
        const label = markers[i];
        const labelWidth = context.measureText(label).width;
        const left = x + labelWidth + 12 * scale < canvas.width ? x + 7 * scale : x - labelWidth - 9 * scale;
        const top = y > 16 * scale ? y - 7 * scale : y + 17 * scale;
        context.lineWidth = 3 * scale;
        context.strokeText(label, left, top);
        context.fillStyle = "#fff";
        context.fillText(label, left, top);
      });
      if (cursor && document.activeElement === canvas && points.length < 6) {
        path([{ x: cursor.x - 8 * scale, y: cursor.y }, { x: cursor.x + 8 * scale, y: cursor.y }], "#fff");
        path([{ x: cursor.x, y: cursor.y - 8 * scale }, { x: cursor.x, y: cursor.y + 8 * scale }], "#fff");
      }
    }

    function readDistance(field, defaultUnit) {
      const value = units.normalize(field.value).trim();
      return units.parse(/^[\d.,]+$/.test(value) ? value + " " + defaultUnit : value);
    }

    function update() {
      undo.disabled = !points.length;
      step.textContent = points.length < 6
        ? "第 " + (points.length + 1) + " / 6 步：" + instructions[points.length]
        : "已标记两个刻度和四角。";
      measurement = undefined;
      get("apply-row").hidden = true;
      result.hidden = points.length < 6;
      if (result.hidden) {
        result.textContent = "";
        return;
      }
      const mm = readDistance(reference, "m");
      if (!mm || mm < 10 || mm > 100000) {
        result.textContent = "请输入两刻度之间的实际距离（10 mm 至 100 m）。";
        return;
      }
      const dimensions = estimate(points, mm / 1000);
      if (dimensions.error) {
        result.textContent = dimensions.error;
        return;
      }
      const length = units.asInput(dimensions.length, 3);
      const width = units.asInput(dimensions.width, 3);
      if (validate("length", length) || validate("width", width)) {
        result.textContent = "估算长宽超出计算器范围，请检查标点与卷尺距离。";
        return;
      }
      const thickness = readDistance(depth, "mm");
      const depthValue = thickness === null ? "" : units.asInput(thickness, 4);
      const size = "估算长 " + length + " m × 宽 " + width + " m。";
      if (validate("depth", depthValue)) {
        result.textContent = size + " 请输入 1～10,000 mm 的厚度。";
        return;
      }
      const cubicMetres = Number(length) * Number(width) * Number(depthValue) / 1000;
      result.textContent = size + " 单处约 " + units.asInput(cubicMetres, cubicMetres < 0.001 ? 9 : 3) + " m³。";
      measurement = { length, width, depth: depthValue };
      get("apply-row").hidden = false;
    }

    function mark(point) {
      if (!image || points.length >= 6) return;
      points.push(point);
      draw();
      update();
    }

    function clear() {
      imageRequest += 1;
      image = undefined;
      cursor = undefined;
      points.length = 0;
      measurement = undefined;
      reference.value = "";
      depth.value = "";
      canvas.style.width = "";
      zoom.setAttribute("aria-pressed", "false");
      zoom.textContent = "放大图片";
      workspace.hidden = true;
      status.textContent = "";
      canvas.width = 1;
      canvas.height = 1;
    }

    async function load(file) {
      clear();
      if (!file.size || file.size > 25 * 1024 * 1024 || (file.type && !file.type.startsWith("image/"))) {
        status.textContent = "请选择不超过 25 MB 的图片。";
        return;
      }
      const request = imageRequest;
      const url = URL.createObjectURL(file);
      status.textContent = "正在打开现场照片。";
      try {
        const source = new Image();
        source.src = url;
        await source.decode();
        if (request !== imageRequest) return;
        const scale = Math.min(1, 2400 / Math.max(source.naturalWidth, source.naturalHeight));
        canvas.width = Math.max(1, Math.round(source.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(source.naturalHeight * scale));
        image = source;
        cursor = { x: canvas.width / 2, y: canvas.height / 2 };
        workspace.hidden = false;
        status.textContent = "";
        draw();
        update();
      } catch {
        if (request === imageRequest) status.textContent = "无法打开此图片，请尝试 JPG、PNG 或 WebP。";
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    for (const [button, picker] of [["camera-button", "camera"], ["upload-button", "file"]]) {
      get(button).addEventListener("click", () => get(picker).click());
      get(picker).addEventListener("change", ({ target }) => {
        const file = target.files[0];
        target.value = "";
        if (file) void load(file);
      });
    }
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      cursor = {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height
      };
      mark(cursor);
    });
    canvas.addEventListener("keydown", (event) => {
      if (!cursor || points.length >= 6) return;
      const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      if (event.key === "Enter") {
        event.preventDefault();
        mark(cursor);
      } else if (moves[event.key]) {
        event.preventDefault();
        const [dx, dy] = moves[event.key];
        const amount = (event.shiftKey ? 10 : 2) * canvas.width / canvas.getBoundingClientRect().width;
        cursor = {
          x: Math.min(canvas.width - 1, Math.max(0, cursor.x + dx * amount)),
          y: Math.min(canvas.height - 1, Math.max(0, cursor.y + dy * amount))
        };
        draw();
      }
    });
    canvas.addEventListener("focus", () => { if (image) draw(); });
    canvas.addEventListener("blur", () => { if (image) draw(); });
    panel.addEventListener("toggle", () => { if (panel.open && image) draw(); });
    for (const input of [reference, depth]) {
      input.addEventListener("input", update);
      input.addEventListener("keydown", (event) => { if (event.key === "Enter") event.preventDefault(); });
    }
    undo.addEventListener("click", () => {
      points.pop();
      draw();
      update();
    });
    zoom.addEventListener("click", () => {
      const active = zoom.getAttribute("aria-pressed") !== "true";
      zoom.setAttribute("aria-pressed", String(active));
      zoom.textContent = active ? "缩回原大小" : "放大图片";
      canvas.style.width = active ? "200%" : "";
      draw();
    });
    get("clear").addEventListener("click", () => {
      clear();
      get("upload-button").focus();
    });
    get("apply").addEventListener("click", () => {
      if (measurement) apply("rect", measurement);
    });
    form.addEventListener("reset", clear);
    panel.hidden = false;
  };
})();

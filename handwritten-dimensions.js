(() => {
  "use strict";
  const weightsUrl = new URL("./ocr/mnist-weights.bin", document.currentScript.src);
  let model;

  function loadModel() {
    model ??= fetch(weightsUrl).then((response) => {
      if (!response.ok) throw new Error("Handwriting model unavailable");
      return response.arrayBuffer();
    }).then((data) => {
      const weights = new Float32Array(data);
      if (weights.length !== 5994) throw new Error("Invalid handwriting model");
      return weights;
    }).catch((error) => {
      model = undefined;
      throw error;
    });
    return model;
  }

  function convolve(input, size, channels, count, weights, offset, bias) {
    const output = new Float32Array(size * size * count);
    for (let f = 0; f < count; f++) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          let sum = weights[bias + f];
          for (let c = 0; c < channels; c++) {
            for (let dy = -2; dy <= 2; dy++) {
              if (y + dy < 0 || y + dy >= size) continue;
              for (let dx = -2; dx <= 2; dx++) {
                if (x + dx < 0 || x + dx >= size) continue;
                const pixel = input[(c * size + y + dy) * size + x + dx];
                sum += pixel * weights[offset + (f * channels + c) * 25 + (dy + 2) * 5 + dx + 2];
              }
            }
          }
          output[(f * size + y) * size + x] = Math.max(sum, 0);
        }
      }
    }
    return output;
  }

  function pool(input, size, channels, step) {
    const next = Math.floor(size / step);
    const output = new Float32Array(channels * next * next);
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < next; y++) {
        for (let x = 0; x < next; x++) {
          let peak = 0;
          for (let dy = 0; dy < step; dy++) {
            for (let dx = 0; dx < step; dx++) {
              peak = Math.max(peak, input[(c * size + y * step + dy) * size + x * step + dx]);
            }
          }
          output[(c * next + y) * next + x] = peak;
        }
      }
    }
    return output;
  }

  function rankDigit(pixels, weights) {
    // Topology and tensor offsets match the ONNX Model Zoo MNIST-8 model; see ocr/MNIST-NOTICE.md.
    const first = pool(convolve(pixels, 28, 1, 8, weights, 0, 200), 28, 8, 2);
    const second = pool(convolve(first, 14, 8, 16, weights, 208, 3408), 14, 16, 3);
    return Array.from({ length: 10 }, (_, digit) => {
      let score = weights[5984 + digit];
      for (let i = 0; i < second.length; i++) score += second[i] * weights[3424 + i * 10 + digit];
      return { digit, score };
    }).sort((a, b) => b.score - a.score);
  }

  function findInk(image) {
    const { width, height } = image;
    const values = new Uint8Array(width * height);
    const histogram = new Uint32Array(256);
    for (let i = 0; i < values.length; i++) {
      const p = i * 4;
      const gray = Math.round((image.data[p] + image.data[p + 1] + image.data[p + 2]) / 3);
      values[i] = gray;
      histogram[gray]++;
    }
    const paperPixels = values.length - histogram[255];
    let count = 0;
    let background = 0;
    while (background < 255 && (count += histogram[background]) < paperPixels * .62) background++;
    if (paperPixels < values.length * .1) background = 255;
    const foreground = values.map((gray) => Number(gray <= background - Math.max(28, background * .2)));
    const parts = [];
    for (let i = 0; i < foreground.length; i++) {
      if (!foreground[i]) continue;
      foreground[i] = 0;
      const pixels = [i];
      let left = width, right = 0, top = height, bottom = 0;
      for (let j = 0; j < pixels.length; j++) {
        const p = pixels[j];
        const y = Math.floor(p / width);
        const x = p % width;
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
        for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
          for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
            const next = yy * width + xx;
            if (!foreground[next]) continue;
            foreground[next] = 0;
            pixels.push(next);
          }
        }
      }
      if (pixels.length >= 5) parts.push({ left, right, top, bottom, w: right - left + 1, h: bottom - top + 1, pixels });
    }
    return parts;
  }

  function digitsFrom(parts, imageWidth, weights) {
    const glyph = document.createElement("canvas");
    const fitted = document.createElement("canvas");
    fitted.width = fitted.height = 28;
    const context = fitted.getContext("2d", { willReadFrequently: true });
    return parts.map((part) => {
      // A single straight pen stroke often represents 1.
      if (part.w / part.h < .28) return { digit: 1, margin: Infinity };
      glyph.width = part.w;
      glyph.height = part.h;
      const source = glyph.getContext("2d");
      const data = source.createImageData(part.w, part.h);
      for (const index of part.pixels) {
        const x = index % imageWidth - part.left;
        const y = Math.floor(index / imageWidth) - part.top;
        data.data.fill(255, (y * part.w + x) * 4, (y * part.w + x + 1) * 4);
      }
      source.putImageData(data, 0, 0);
      const scale = 20 / Math.max(part.w, part.h);
      const width = part.w * scale;
      const height = part.h * scale;
      context.fillStyle = "#000";
      context.fillRect(0, 0, 28, 28);
      context.drawImage(glyph, (28 - width) / 2, (28 - height) / 2, width, height);
      const rgba = context.getImageData(0, 0, 28, 28).data;
      const input = Float32Array.from({ length: 784 }, (_, i) => rgba[i * 4] / 255);
      const [first, second] = rankDigit(input, weights);
      return { digit: first.digit, margin: first.score - second.score };
    });
  }

  function turnImage(source, degrees) {
    const rotated = document.createElement("canvas");
    const radians = degrees * Math.PI / 180;
    const sine = Math.abs(Math.sin(radians)), cosine = Math.abs(Math.cos(radians));
    rotated.width = Math.ceil(source.width * cosine + source.height * sine - 1e-8);
    rotated.height = Math.ceil(source.width * sine + source.height * cosine - 1e-8);
    const context = rotated.getContext("2d");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, rotated.width, rotated.height);
    context.translate(rotated.width / 2, rotated.height / 2);
    context.rotate(radians);
    context.drawImage(source, -source.width / 2, -source.height / 2);
    return rotated;
  }

  function strokeTilt(part, width) {
    const ends = (top) => part.pixels.filter((pixel) => {
      const y = Math.floor(pixel / width);
      return top ? y <= part.top + part.h * .25 : y >= part.bottom - part.h * .25;
    });
    const meanX = (pixels) => pixels.reduce((sum, p) => sum + p % width, 0) / pixels.length;
    return Math.atan2(meanX(ends(false)) - meanX(ends(true)), part.h * .75) * 180 / Math.PI;
  }

  async function readMark(crop, straighten = true) {
    const context = crop.getContext("2d");
    const parts = findInk(context.getImageData(0, 0, crop.width, crop.height));
    if (!parts.length) return { text: "", note: "没有找到笔迹，请重新框选或手动录入。" };
    const largest = Math.max(...parts.map((part) => part.pixels.length));
    const ink = parts.filter((part) => part.pixels.length >= Math.max(6, largest * .034));
    const height = Math.max(...ink.map((part) => part.h));
    const baseline = Math.max(...ink.filter((part) => part.h >= height * .72).map((part) => part.bottom));
    const digits = ink.filter((part) => part.h >= height * .52 &&
      (part.bottom >= baseline - height * .42 || part.w >= height * .32)).sort((a, b) => a.left - b.left);
    if (!digits.length || digits.length > 4) return { text: "", note: "请只框一个标注，并避开轮廓线。" };
    const tail = digits.at(-1);
    const primes = ink.filter((part) => !digits.includes(part) && part.left > tail.right - height * .34 &&
      part.h >= height * .12 && part.bottom <= baseline - height * .42 && part.w <= height * .35);
    if (straighten && primes.length === 2) {
      const [first, second] = primes.map((part) => strokeTilt(part, crop.width));
      const correction = (first + second) / 2;
      if (Math.abs(first - second) < 12 && Math.abs(correction) >= 8 && Math.abs(correction) <= 25) {
        const adjusted = await readMark(turnImage(crop, correction), false);
        if (adjusted.text.endsWith("″")) return adjusted;
      }
    }
    const dots = ink.filter((part) => part.h < height * .28 && part.w < height * .3 &&
      part.top > baseline - height * .35 && part.left > digits[0].right - height * .1 && part.right < tail.left);
    const predictions = digitsFrom(digits, crop.width, await loadModel());
    const text = predictions.map(({ digit }, i) => {
      const point = i > 0 && dots.some((part) => part.left >= digits[i - 1].right - height * .1 && part.right < digits[i].left);
      return (point ? "." : "") + digit;
    }).join("");
    const unit = primes.length === 1 ? "′" : primes.length === 2 ? "″" : "";
    if (predictions.some(({ margin }) => margin < 1)) {
      return { text: "", note: "数字笔迹有歧义，请对照缩图录入，或旋转选区重读。" };
    }
    return { text: text + unit, note: unit
      ? (predictions.some(({ margin }) => margin < 2.5) ? "数字较难辨认，请重点核对。" : "请核对数字和撇号。")
      : "撇号未能确认，请手动补上 ′ 或 ″。" };
  }

  // Review and calculations are initialized by photo-dimensions.js when an image is loaded.
  window.setupHandwrittenDimensions = ({ form, validate, apply }) => {
    const byId = (id) => document.getElementById(id);
    const panel = byId("handwritten-review");
    const canvas = byId("handwritten-canvas");
    const status = byId("handwritten-status");
    const rows = byId("handwritten-rows");
    const pieces = byId("sketch-pieces");
    const shape = byId("sketch-shape");
    const area = byId("sketch-area");
    const depth = byId("sketch-depth");
    const note = byId("sketch-status");
    const calculate = byId("sketch-apply");
    const read = byId("handwritten-read");
    const zoom = byId("handwritten-zoom"), pan = byId("handwritten-pan");
    const units = window.concreteUnits;
    let image, selected, drag, draft;
    let revision = 0;

    const draw = () => {
      if (!image) return;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      if (!selected) return;
      context.fillStyle = "#139c5433";
      context.fillRect(selected.x, selected.y, selected.w, selected.h);
      context.lineWidth = Math.max(2, canvas.width / 300);
      context.strokeStyle = "#007444";
      context.strokeRect(selected.x, selected.y, selected.w, selected.h);
    };
    const setImage = (source) => {
      revision++;
      image = source;
      selected = drag = undefined;
      canvas.width = source.width;
      canvas.height = source.height;
      read.disabled = true;
      draw();
    };
    const position = (event) => {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(canvas.width, Math.round((event.clientX - rect.left) * canvas.width / rect.width))),
        y: Math.max(0, Math.min(canvas.height, Math.round((event.clientY - rect.top) * canvas.height / rect.height)))
      };
    };
    canvas.addEventListener("pointerdown", (event) => {
      if (!image || !event.isPrimary || event.button !== 0 || canvas.classList.contains("pan-image")) return;
      canvas.setPointerCapture(event.pointerId);
      drag = position(event);
      selected = undefined;
      read.disabled = true;
      draw();
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag || !event.isPrimary) return;
      const p = position(event);
      selected = { x: Math.min(p.x, drag.x), y: Math.min(p.y, drag.y), w: Math.abs(p.x - drag.x), h: Math.abs(p.y - drag.y) };
      draw();
    });
    const stopDrag = () => {
      drag = undefined;
      read.disabled = !(selected?.w >= 8 && selected?.h >= 8);
    };
    canvas.addEventListener("pointerup", stopDrag);
    canvas.addEventListener("pointercancel", stopDrag);
    byId("handwritten-turn").addEventListener("click", () => { if (image) setImage(turnImage(image, -90)); });
    zoom.addEventListener("click", () => {
      const enlarged = zoom.getAttribute("aria-pressed") === "false";
      canvas.style.width = enlarged ? "200%" : "";
      zoom.setAttribute("aria-pressed", String(enlarged));
      zoom.textContent = enlarged ? "缩回原大小" : "放大图片";
    });
    pan.addEventListener("click", () => {
      const enabled = canvas.classList.toggle("pan-image");
      pan.setAttribute("aria-pressed", String(enabled));
      pan.textContent = enabled ? "切回框选" : "移动图片";
    });

    function readableLength(value, defaultUnit) {
      const text = units.normalize(value).trim();
      return units.parse(/^[\d.]+$/.test(text) ? text + " " + defaultUnit : text);
    }
    function update() {
      const irregular = shape.value === "irregular";
      byId("sketch-irregular").hidden = !irregular;
      byId("sketch-rect").hidden = irregular;
      calculate.disabled = true;
      draft = undefined;
      const depthError = validate("depth", depth.value);
      if (irregular) {
        let netArea = area.value.trim();
        if (!netArea && pieces.children.length) {
          let sum = 0;
          for (const piece of pieces.children) {
            const [baseInput, heightInput, diagonalInput] = piece.querySelectorAll("input");
            const base = readableLength(baseInput.value, "m");
            const height = readableLength(heightInput.value, "m");
            const triangle = piece.dataset.shape === "triangle";
            let rise = height;
            if (triangle && !heightInput.value.trim()) {
              const diagonal = readableLength(diagonalInput.value, "m");
              if (diagonal > base && base > 0) rise = Math.sqrt(diagonal ** 2 - base ** 2);
            }
            if (!(base > 0 && rise > 0)) {
              note.textContent = triangle ? "请补填三角形的垂直高度，或填斜边和一条直角边。" : "请补填矩形分块的长和宽。";
              return;
            }
            const sign = Number(piece.querySelector("select").value);
            sum += sign * base * rise / (triangle ? 2 : 1) / 1e6;
          }
          if (sum > 0 && sum <= 1e8) netArea = units.asInput(sum, 6);
        }
        if (validate("area", netArea)) {
          note.textContent = netArea ? "请核对净面积及单位。" :
            "资料不足。请补填实测净面积，或增加分块并填斜边的水平、垂直投影（或直角边和斜边）。";
          return;
        }
        draft = { mode: "area", values: { area: netArea, depth: depth.value } };
      } else {
        const length = byId("sketch-length").value;
        const width = byId("sketch-width").value;
        if (validate("length", length) || validate("width", width)) {
          note.textContent = "请补齐矩形长宽。";
          return;
        }
        draft = { mode: "rect", values: { length, width, depth: depth.value } };
      }
      note.textContent = depthError ? "请补填有效厚度，例如 4″。" : "尺寸已齐全，可以计算。";
      calculate.disabled = Boolean(depthError);
    }

    function addPiece(type) {
      const triangle = type === "triangle";
      const row = document.createElement("div");
      row.className = "sketch-piece";
      row.dataset.shape = type;
      row.innerHTML = '<strong>' + (triangle ? "直角三角形" : "矩形") + '</strong>' +
        '<div class="sketch-piece-fields"><label>加或减<select><option value="1">加上</option><option value="-1">扣除</option></select></label>' +
        '<label>' + (triangle ? "底边（直角边）" : "长度") + '<input autocomplete="off"></label>' +
        '<label>' + (triangle ? "垂直高度" : "宽度") + '<input autocomplete="off"></label>' +
        (triangle ? '<label>斜边（未填高度时）<input autocomplete="off"></label>' : "") + '</div>' +
        '<div class="photo-actions"><button type="button">移除分块</button></div>';
      row.querySelector("button").addEventListener("click", () => { row.remove(); update(); });
      pieces.append(row);
      update();
      row.querySelector("input").focus();
    }

    function addRow(crop, reading = { text: "", note: "请按原图录入，如 10′ 或 4″。" }) {
      const row = document.createElement("div");
      row.className = "sketch-row";
      row.innerHTML = '<div class="sketch-thumb"></div><div>' +
        '<div class="sketch-row-fields"><label>标注<input type="text" autocomplete="off" placeholder="例如 10′"></label>' +
        '<label>用途<select><option value="">边长／核对</option><option value="depth">厚度</option><option value="length">矩形长度</option><option value="width">矩形宽度</option></select></label></div>' +
        '<p class="hint"></p><p class="sketch-output" aria-live="polite"></p><div class="photo-actions">' +
        '<button type="button" class="sketch-use" hidden>填入用途</button><button type="button" class="sketch-rotate" hidden>转正选区</button>' +
        '<button type="button" class="sketch-delete">移除</button></div></div>';
      const input = row.querySelector("input"), purpose = row.querySelector("select");
      const hint = row.querySelector(".hint"), output = row.querySelector("p.sketch-output");
      const use = row.querySelector(".sketch-use"), turn = row.querySelector(".sketch-rotate");
      const thumb = document.createElement("img");
      thumb.alt = "标注选区";
      const convert = () => {
        const mm = units.parse(input.value);
        const valid = mm > 0;
        output.textContent = valid ? units.asInput(mm / 1000, 6) + " m · " + units.asInput(mm, 4) + " mm" : "请输入带单位的尺寸。";
        input.setAttribute("aria-invalid", String(Boolean(input.value) && !valid));
        use.hidden = !valid || !purpose.value;
      };
      input.value = reading.text;
      hint.textContent = reading.note;
      if (crop) {
        thumb.src = crop.toDataURL();
        row.querySelector(".sketch-thumb").append(thumb);
        turn.hidden = false;
      }
      input.addEventListener("input", convert);
      purpose.addEventListener("change", convert);
      use.addEventListener("click", () => {
        if (purpose.value !== "depth" && shape.value !== "rectangle") {
          hint.textContent = "请先在下方把轮廓类型设为矩形。";
          return;
        }
        byId("sketch-" + purpose.value).value = input.value;
        update();
      });
      row.querySelector(".sketch-delete").addEventListener("click", () => row.remove());
      let turns = 0;
      turn.addEventListener("click", async () => {
        turns++;
        const angle = [15, -15, 30, -30, 90, 180, 270, 0][(turns - 1) % 8];
        const corrected = turnImage(crop, angle);
        thumb.src = corrected.toDataURL();
        turn.disabled = true;
        try {
          const next = await readMark(corrected, false);
          input.value = next.text;
          hint.textContent = next.note;
          convert();
        } catch {
          hint.textContent = "无法识别，请手动修改。";
        } finally { turn.disabled = false; }
      });
      rows.append(row);
      convert();
    }
    read.addEventListener("click", async () => {
      if (!selected || !image) return;
      const ticket = revision;
      const crop = document.createElement("canvas");
      const scale = Math.min(1, 420 / Math.max(selected.w, selected.h));
      crop.width = Math.max(1, Math.round(selected.w * scale));
      crop.height = Math.max(1, Math.round(selected.h * scale));
      crop.getContext("2d").drawImage(image, selected.x, selected.y, selected.w, selected.h, 0, 0, crop.width, crop.height);
      read.disabled = true;
      status.textContent = "正在识别选区…";
      try {
        const result = await readMark(crop);
        if (ticket !== revision) return;
        addRow(crop, result);
        status.textContent = "已添加选区，请核对。";
      } catch {
        if (ticket === revision) {
          addRow(crop);
          status.textContent = "识别暂不可用，可按缩图手动录入。";
        }
      } finally { if (ticket === revision) read.disabled = false; }
    });
    byId("handwritten-manual").addEventListener("click", () => addRow(null));
    byId("sketch-rectangle").addEventListener("click", () => addPiece("rectangle"));
    byId("sketch-triangle").addEventListener("click", () => addPiece("triangle"));
    byId("sketch-shape").addEventListener("change", update);
    panel.addEventListener("input", (event) => {
      if (event.target.closest(".photo-review")) update();
    });
    panel.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.target.matches("input")) event.preventDefault();
    });
    calculate.addEventListener("click", () => {
      if (calculate.disabled || !draft) return;
      const volume = apply(draft.mode, draft.values);
      note.textContent = volume === "—" ? "已套用尺寸，请检查数量。" : "计算结果 " + volume + " m³。";
    });
    return {
      load(source) { panel.hidden = false; setImage(source); update(); },
      open() { if (!panel.hidden) panel.open = true; },
      clear() {
        revision++;
        image = selected = drag = draft = undefined;
        panel.open = false;
        panel.hidden = true;
        rows.replaceChildren();
        pieces.replaceChildren();
        status.textContent = "";
        canvas.style.width = "";
        canvas.classList.remove("pan-image");
        zoom.setAttribute("aria-pressed", "false");
        zoom.textContent = "放大图片";
        pan.setAttribute("aria-pressed", "false");
        pan.textContent = "移动图片";
        for (const id of ["area", "depth", "length", "width"]) byId("sketch-" + id).value = "";
        shape.value = "irregular";
      }
    };
  };
})();

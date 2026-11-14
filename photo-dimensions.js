(() => {
  "use strict";
  const base = new URL("./ocr/", document.currentScript.src);
  const dimensionLabels = { length: "长度", width: "宽度", area: "单处面积", depth: "厚度" };
  const modeFields = { rect: ["length", "width", "depth"], area: ["area", "depth"] };
  const labels = {
    length: "长度|長度|长|長|\\b(?:length|panjang|L)\\b",
    width: "宽度|寬度|宽|寬|\\b(?:width|lebar|W)\\b",
    depth: "厚度|厚|深度|\\b(?:thickness|depth|tebal|ketebalan|T)\\b",
    area: "单处面积|單處面積|面积|面積|\\b(?:area|keluasan)\\b"
  };
  const number = "(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\\.[0-9]+)?|\\.[0-9]+";
  const separator = "\\s*[:=]?\\s*";
  let engineScript;

  function normalizeMeasurement(dimension, amount, measurementUnit) {
    const raw = amount.replaceAll(",", "");
    if (raw.length > 16) return null;
    const baseUnit = measurementUnit.toLowerCase();
    let metricPower = 0;
    if (/^(mm|毫米)$/.test(baseUnit)) metricPower = 3;
    if (/^(cm|厘米|公分)$/.test(baseUnit)) metricPower = 2;
    const shift = dimension === "depth" ? 3 - metricPower : -metricPower;
    // Move the decimal point using the original digits to preserve exact input precision.
    const [whole, fraction = ""] = raw.split(".");
    const digits = whole + fraction;
    const point = whole.length + shift;
    let integer = point <= 0 ? "0" : digits.slice(0, point).padEnd(point, "0");
    let decimal = point <= 0 ? "0".repeat(-point) + digits : digits.slice(point);
    integer = integer.replace(/^0+(?=\d)/, "");
    decimal = decimal.replace(/0+$/, "");
    return decimal ? integer + "." + decimal : integer;
  }

  function parseDimensions(text) {
    const units = window.concreteUnits;
    const normalized = units.normalize(text)
      .replace(/(?<=\p{Script=Han})[ \t]+(?=\p{Script=Han})/gu, "");
    const matches = {};
    const add = (dimension, amount, measurementUnit) => {
      const value = normalizeMeasurement(dimension, amount, measurementUnit);
      if (value !== null) (matches[dimension] ??= new Set()).add(value);
    };
    const addLength = (dimension, reading) => {
      const metric = new RegExp("^(" + number + ")[ \\t]*(毫米|厘米|公分|公尺|米|mm|cm|m)$", "iu").exec(reading);
      if (metric) {
        add(dimension, metric[1], metric[2]);
        return;
      }
      const mm = units.parse(reading);
      if (mm !== null && mm > 0) {
        const value = dimension === "depth" ? units.asInput(mm, 4) : units.asInput(mm / 1000, 6);
        (matches[dimension] ??= new Set()).add(value);
      }
    };
    for (const [dimension, label] of Object.entries(labels)) {
      const reading = "(" + (dimension === "area" ? units.areaPattern : units.pattern) + ")";
      const pattern = new RegExp("(?:" + label + ")" + separator + reading, "giu");
      for (const [, value] of normalized.matchAll(pattern)) {
        if (dimension !== "area") {
          addLength(dimension, value);
          continue;
        }
        const area = units.parseArea(value);
        if (area > 0 && area <= 1e8) (matches.area ??= new Set()).add(units.asInput(area, 6));
      }
    }
    // A written dimension triplet is conventionally length × width × thickness.
    // Require an explicit unit on each number and send all readings for review.
    const piece = "(" + units.pattern + ")";
    const times = "[ \\t]*[×xX*][ \\t]*";
    const triplets = new RegExp("(?<![0-9.,+\\-])" + piece + times + piece + times + piece, "giu");
    for (const [, length, width, depth] of normalized.matchAll(triplets)) {
      addLength("length", length);
      addLength("width", width);
      addLength("depth", depth);
    }
    return matches;
  }

  function loadEngine() {
    if (window.Tesseract) return Promise.resolve(window.Tesseract);
    if (!engineScript) {
      engineScript = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = new URL("tesseract.min.js", base);
        function fail() {
          clearTimeout(deadline);
          script.remove();
          reject(new Error("OCR assets unavailable"));
        }
        const deadline = setTimeout(fail, 20000);
        script.onload = () => {
          clearTimeout(deadline);
          if (window.Tesseract) resolve(window.Tesseract);
          else fail();
        };
        script.onerror = fail;
        document.head.append(script);
      }).catch((error) => {
        engineScript = undefined;
        throw error;
      });
    }
    return engineScript;
  }

  async function prepareImage(url) {
    const image = new Image();
    image.src = url;
    try {
      await image.decode();
    } catch {
      throw new Error("请选择浏览器可打开的 JPG、PNG 或 WebP 图片。");
    }
    const scale = Math.min(1, 2200 / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d", { alpha: false });
    context.fillStyle = "#fff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  async function recognize(image, signal, progress) {
    const engine = await loadEngine();
    if (signal.aborted) return null;
    let worker;
    let finished = false;
    let deadline;
    let cancel;
    const stop = new Promise((_, reject) => {
      cancel = () => reject(new Error("识别已取消。"));
      signal.addEventListener("abort", cancel, { once: true });
      deadline = setTimeout(() => reject(new Error("识别超时，请裁切到尺寸标注处后重试。")), 75000);
    });
    const job = async () => {
      const options = {
        workerPath: new URL("worker.min.js", base).href,
        corePath: new URL("tesseract-core-lstm.wasm.js", base).href,
        langPath: base.href.replace(/\/$/, "") + "/lang",
        workerBlobURL: false,
        logger: ({ status, progress: fraction }) => {
          if (!finished && status === "recognizing text") progress(fraction);
        }
      };
      const ready = await new Promise((resolve, reject) => {
        engine.createWorker(["chi_sim", "eng"], 1, { ...options, errorHandler: reject }).then(resolve, reject);
      });
      if (finished) {
        await ready.terminate();
        return null;
      }
      worker = ready;
      await worker.setParameters({ tessedit_pageseg_mode: engine.PSM.SINGLE_BLOCK });
      const primary = (await worker.recognize(image)).data.text;
      const primaryReadings = parseDimensions(primary);
      const complete = primaryReadings.depth?.size === 1 && (primaryReadings.area?.size === 1 ||
        (primaryReadings.length?.size === 1 && primaryReadings.width?.size === 1));
      if (complete || !/['‘’′"“”″]/u.test(primary)) return { text: primary, readings: primaryReadings };

      // A second language pass can recover small prime marks omitted by the bilingual OCR.
      await worker.reinitialize("eng");
      await worker.setParameters({ tessedit_pageseg_mode: engine.PSM.SINGLE_BLOCK });
      const enlarged = document.createElement("canvas");
      const scale = Math.min(2, Math.max(1, 3000 / Math.max(image.width, image.height)));
      enlarged.width = Math.round(image.width * scale);
      enlarged.height = Math.round(image.height * scale);
      enlarged.getContext("2d").drawImage(image, 0, 0, enlarged.width, enlarged.height);
      const alternate = (await worker.recognize(enlarged)).data.text;
      const units = window.concreteUnits;
      const alternateNormalized = units.normalize(alternate);
      const linearLabel = "(?:" + labels.length + "|" + labels.width + ")" + separator;
      const possibleFeet = "(" + linearLabel + number + ")[ \\t]*°";
      // Recover a common OCR substitution only in labeled dimensions and only
      // when a feet-and-inches reading provides independent unit context.
      const imperialContext = new RegExp(possibleFeet + "[ \\t]*-?[ \\t]*(" + number + ')[ \\t]*"', "iu");
      const recovered = imperialContext.test(alternateNormalized)
        ? alternateNormalized.replace(new RegExp(possibleFeet + '(?=[ \\t]*(?:-?[ \\t]*' + number + '[ \\t]*"|$))', "gimu"), "$1'")
        : alternateNormalized;
      const alternateReadings = parseDimensions(alternateNormalized);
      const recoveredReadings = recovered === alternateNormalized ? {} : parseDimensions(recovered);
      const uncertain = [];
      for (const id of ["length", "width"]) {
        if (!primaryReadings[id] && !alternateReadings[id] && recoveredReadings[id]?.size === 1) {
          alternateReadings[id] = recoveredReadings[id];
          uncertain.push(id);
        }
      }
      const supplemented = Object.entries(alternateReadings).filter(([id, values]) =>
        !primaryReadings[id] && values.size === 1);
      return {
        text: alternate.trim() === primary.trim() ? primary : primary + "\n\n补充识别：\n" + alternate,
        readings: { ...primaryReadings, ...Object.fromEntries(supplemented) },
        uncertain
      };
    };
    try {
      return await Promise.race([stop, job()]);
    } finally {
      finished = true;
      clearTimeout(deadline);
      signal.removeEventListener("abort", cancel);
      if (worker) await worker.terminate();
    }
  }

  window.setupPhotoDimensions = ({ form, validate, apply }) => {
    const handwritten = window.setupHandwrittenDimensions?.({ form, validate, apply });
    const get = (id) => document.getElementById(id);
    const status = get("photo-status");
    const review = get("photo-review");
    const fields = get("photo-fields");
    const preview = get("photo-preview");
    const camera = get("camera-photo");
    const select = get("select-photo");
    const textDetails = get("photo-text");
    const transcript = get("photo-transcript");
    const remove = get("remove-photo");
    const take = get("take-photo");
    const upload = get("upload-photo");
    const draft = {};
    let currentImage;
    let controller;
    let mode;

    function clearPhoto() {
      handwritten?.clear();
      controller?.abort();
      controller = undefined;
      if (currentImage) URL.revokeObjectURL(currentImage);
      currentImage = undefined;
      preview.removeAttribute("src");
      preview.hidden = true;
      review.hidden = true;
      remove.hidden = true;
      textDetails.hidden = true;
      textDetails.open = false;
      transcript.textContent = "";
      status.textContent = "";
      fields.replaceChildren();
      get("photo-mode-field").hidden = true;
    }

    function renderReview() {
      fields.replaceChildren();
      fields.dataset.mode = mode;
      for (const id of modeFields[mode]) {
        const field = document.createElement("div");
        field.className = "field";
        // Labels and units come from fixed application constants, not from the OCR text.
        const unit = { area: "m²", depth: "mm", length: "m", width: "m" }[id];
        field.innerHTML = '<label for="photo-' + id + '">' + dimensionLabels[id] +
          '</label><div class="control"><input id="photo-' + id +
          '" inputmode="decimal" autocomplete="off" aria-describedby="photo-' + id +
          '-error"><span class="unit">' + unit + '</span></div><p class="error" id="photo-' + id + '-error" hidden></p>';
        const input = field.querySelector("input");
        input.value = draft[id] || "";
        input.addEventListener("input", () => {
          draft[id] = input.value;
          showFieldError(id);
        });
        // Enter reviews and applies photo readings rather than submitting the older main values.
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter" && !event.isComposing) {
            event.preventDefault();
            applyPhoto();
          }
        });
        fields.append(field);
      }
      review.hidden = false;
    }

    function showFieldError(id) {
      const input = get("photo-" + id);
      const error = get("photo-" + id + "-error");
      const message = validate(id, input.value);
      error.textContent = message || "";
      error.hidden = !message;
      if (message) input.setAttribute("aria-invalid", "true");
      else input.removeAttribute("aria-invalid");
      return message;
    }

    function applyPhoto() {
      const ids = modeFields[mode];
      const invalid = ids.filter(showFieldError);
      if (invalid.length) {
        get("photo-" + invalid[0]).focus();
        return;
      }
      const dimensions = Object.fromEntries(ids.map((id) => [id, draft[id]]));
      const volume = apply(mode, dimensions);
      status.textContent = volume === "—"
        ? "已套用图片尺寸。请检查数量后再计算。"
        : "已套用图片尺寸，混凝土用量 " + volume + " m³。";
    }

    async function processFile(file) {
      clearPhoto();
      if (!file.size || file.size > 25 * 1024 * 1024 || (file.type && !file.type.startsWith("image/"))) {
        status.textContent = "请选择一张不超过 25 MB 的图片。";
        return;
      }
      controller = new AbortController();
      const { signal } = controller;
      const url = URL.createObjectURL(file);
      currentImage = url;
      remove.hidden = false;
      status.textContent = "正在准备文字识别。首次使用需要加载识别文件。";
      try {
        const image = await prepareImage(url);
        if (signal.aborted) return;
        preview.src = url;
        preview.hidden = false;
        handwritten?.load(image);
        const recognized = await recognize(image, signal, (progress) => {
          if (!signal.aborted) status.textContent = "正在识别尺寸文字 " + Math.round(progress * 100) + "%";
        });
        if (signal.aborted) return;
        const { text, readings: matches, uncertain = [] } = recognized;
        transcript.textContent = text.trim();
        textDetails.hidden = !transcript.textContent;
        for (const id of Object.keys(dimensionLabels)) {
          const readings = matches[id];
          const value = readings?.size === 1 ? [...readings][0] : "";
          draft[id] = value && !validate(id, value) ? value : "";
        }
        if (!Object.keys(matches).length) {
          status.textContent = "整图没有读到带单位的尺寸。可框选手写标注逐项识别。";
          handwritten?.open();
          return;
        }
        const hasRect = Boolean(matches.length || matches.width);
        const hasArea = Boolean(matches.area);
        mode = hasArea && (!hasRect || !(draft.length && draft.width)) ? "area" : "rect";
        get("photo-mode").value = mode;
        get("photo-mode-field").hidden = !(hasRect && hasArea);
        renderReview();
        const incomplete = modeFields[mode].some((id) => !draft[id]);
        status.textContent = incomplete
          ? "请对照原图补齐空白尺寸。读到多个不同值或超出范围时会留空。"
          : "请对照原图核对尺寸，再套用到计算器。";
        if (uncertain.length) {
          status.textContent += " " + uncertain.map((id) => dimensionLabels[id]).join("、") +
            "的单撇较难辨认，暂按英尺换算，请核对原图。";
        }
      } catch (error) {
        if (!signal.aborted) {
          handwritten?.open();
          status.textContent = error instanceof Error && /[一-龯]/u.test(error.message) ? error.message
            : "识别暂时无法完成，请检查网络后重试，或直接在表单输入尺寸。";
        }
      }
    }

    take.addEventListener("click", () => camera.click());
    upload.addEventListener("click", () => select.click());
    for (const picker of [camera, select]) {
      picker.addEventListener("change", () => {
        const file = picker.files[0];
        picker.value = "";
        if (file) void processFile(file);
      });
    }
    get("photo-mode").addEventListener("change", ({ target }) => {
      mode = target.value;
      renderReview();
    });
    get("apply-photo").addEventListener("click", applyPhoto);
    remove.addEventListener("click", () => {
      clearPhoto();
      upload.focus();
    });
    form.addEventListener("reset", clearPhoto);
    get("photo-panel").hidden = false;
  };
})();

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

  const centerX = (part) => (part.left + part.right) / 2;
  const centerY = (part) => (part.top + part.bottom) / 2;

  function findOutline(parts, width, height) {
    const outlines = parts.filter((part) => part.w > width * .12 && part.h > height * .12 &&
      part.left > width * .015 && part.top > height * .015 &&
      part.right < width * .985 && part.bottom < height * .985 &&
      part.pixels.length < part.w * part.h * .28);
    return outlines.sort((a, b) => b.w * b.h - a.w * a.h)[0];
  }

  function findLabels(parts, outline, width, height) {
    const edge = Math.min(width, height);
    const zone = outline ? Math.max(outline.w, outline.h) * .6 : edge;
    const ink = parts.filter((part) => part !== outline && part.pixels.length > 9 &&
      Math.max(part.w, part.h) < edge * .16 && part.left > 1 && part.top > 1 &&
      part.right < width - 2 && part.bottom < height - 2 &&
      (!outline || (part.left >= outline.left - zone && part.right <= outline.right + zone &&
        part.top >= outline.top - zone && part.bottom <= outline.bottom + zone)));
    if (ink.length > 210) return [];
    const tall = ink.filter((part) => part.h >= edge * .016 && part.h < edge * .115 && part.w / part.h >= .34);
    const labels = [];
    for (const digit of tall) {
      const h = digit.h;
      const possiblePrimes = ink.filter((part) => {
        const dx = (centerX(part) - centerX(digit)) / h;
        const dy = (centerY(digit) - centerY(part)) / h;
        return dx > -.19 && dx < 1.35 && part.left >= digit.left + h * .18 && part.left - digit.right < h * .48 &&
          dy > .34 && dy < 1.58 && part.bottom < digit.bottom - h * .23 &&
          part.h >= h * .19 && part.h <= h * .78 && part.w <= h * .42 &&
          part.w / part.h < .79 && part.pixels.length < digit.pixels.length * .66;
      }).sort((a, b) => centerX(a) - centerX(b));
      if (!possiblePrimes.length) continue;
      const marks = [possiblePrimes[0]];
      const next = possiblePrimes[1];
      if (next && centerX(next) - centerX(marks[0]) < h * .42 &&
          Math.abs(centerY(next) - centerY(marks[0])) < h * .35 &&
          Math.abs(strokeTilt(next, width) - strokeTilt(marks[0], width)) < 18) marks.push(next);
      const digits = [digit];
      const dots = [];
      for (let step = 0; step < 3; step++) {
        const first = digits[0];
        const candidates = ink.filter((part) => {
          const gap = first.left - part.right;
          return !marks.includes(part) && part.h >= h * .48 && part.h <= h * 1.42 &&
            part.w <= h * 1.05 && gap >= -h * .05 && gap <= h * .67 &&
            part.bottom >= first.bottom - h * .48 && part.bottom <= first.bottom + h * 1.12 &&
            part.top > first.top - h * .46 && part.top <= first.bottom + h * .04;
        }).sort((a, b) => b.right - a.right);
        const prior = candidates[0];
        if (!prior) break;
        const dot = ink.find((part) => part.h <= h * .23 && part.w <= h * .28 &&
          centerX(part) > prior.right && centerX(part) < first.left + h * .12 &&
          part.top > Math.min(prior.bottom, first.bottom) - h * .55 &&
          part.bottom < Math.max(prior.bottom, first.bottom) + h * .24 &&
          prior.pixels.length <= digit.pixels.length * 2.5);
        if (dot) dots.push(dot);
        else if (Math.abs(prior.bottom - first.bottom) > h * .36) break;
        digits.unshift(prior);
      }
      labels.push({ digit, digits, dots, marks });
    }
    return labels;
  }

  function labelCrop(image, parts, isolate, pad = 4) {
    const left = Math.min(...parts.map((part) => part.left));
    const top = Math.min(...parts.map((part) => part.top));
    const right = Math.max(...parts.map((part) => part.right));
    const bottom = Math.max(...parts.map((part) => part.bottom));
    const crop = document.createElement("canvas");
    crop.width = right - left + 1 + pad * 2;
    crop.height = bottom - top + 1 + pad * 2;
    const context = crop.getContext("2d");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, crop.width, crop.height);
    if (isolate) {
      context.fillStyle = "#111";
      for (const { pixels } of parts) {
        for (const p of pixels) context.fillRect(p % image.width - left + pad, Math.floor(p / image.width) - top + pad, 1, 1);
      }
    } else {
      context.drawImage(image, pad - left, pad - top);
    }
    return { crop, box: { x: left, y: top, w: right - left + 1, h: bottom - top + 1 } };
  }

  function classifyTurnedDigit(crop, angle, weights) {
    const aligned = turnImage(crop, angle);
    const parts = findInk(aligned.getContext("2d").getImageData(0, 0, aligned.width, aligned.height));
    const largest = parts.sort((a, b) => b.pixels.length - a.pixels.length)[0];
    if (!largest || largest.h < aligned.height * .23 || largest.w > largest.h * 1.65 ||
        largest.pixels.length > aligned.width * aligned.height * .48) return null;
    if (parts.slice(1).some((part) => part.pixels.length >= largest.pixels.length * .45)) return null;
    return digitsFrom([largest], aligned.width, weights)[0];
  }

  function consensus(samples, minimumVotes, minimumGap = 1) {
    const totals = Array.from({ length: 10 }, () => ({ votes: 0, strength: 0 }));
    for (const sample of samples) {
      if (!sample || sample.margin < 1 || !Number.isFinite(sample.margin)) continue;
      totals[sample.digit].votes++;
      totals[sample.digit].strength += Math.min(7, sample.margin);
    }
    const [first, second] = totals.map((item, digit) => ({ ...item, digit }))
      .sort((a, b) => b.strength - a.strength);
    return first.votes >= minimumVotes && first.strength >= second.strength + minimumGap ? first.digit : null;
  }

  function recoverTiltedLabel(image, label, tilt, weights, original) {
    if (label.marks.length !== 1 || Math.abs(tilt) > 63) return null;
    if (label.digits.length > 1 && !label.dots.length) return null;
    const markedTilt = strokeTilt(label.marks[0], image.width);
    const decimal = label.digits.length > 1;
    if (decimal && Math.abs(tilt) < 18) return null;
    const readings = label.digits.map((part, i) => {
      const upright = original[i];
      if (decimal && i === label.digits.length - 1 && upright.margin >= 3.25) return upright.digit;
      const isolated = labelCrop(image, [part], true, Math.ceil(part.h * .28)).crop;
      const angles = decimal
        ? [.44, .5, .56, .62, .68, .78].map((factor) => -tilt * factor)
        : [markedTilt - 36, markedTilt - 24, markedTilt + 24, markedTilt + 36, markedTilt + 48];
      return consensus(angles.map((angle) => classifyTurnedDigit(isolated, angle, weights)), decimal ? 3 : 4, 6);
    });
    if (readings.some((digit) => digit === null)) return null;
    return readings.map((digit, i) => {
      const decimalPoint = i > 0 && label.dots.some((dot) => centerX(dot) > centerX(label.digits[i - 1]) &&
        centerX(dot) < centerX(label.digits[i]));
      return (decimalPoint ? "." : "") + digit;
    }).join("") + "′";
  }

  function recoverJoinedDigit(image, prime, weights) {
    const x = centerX(prime), base = prime.h;
    const samples = [];
    // The detached prime locates the digit even when its ink joins the boundary.
    // Trim progressively toward the prime to drop part of the boundary stroke.
    for (const [width, height] of [[2.12, 2.82], [2.02, 2.68], [1.84, 2.57], [2.18, 2.62]]) {
      const bounds = {
        left: Math.max(0, Math.round(x - base * 1.17)),
        right: Math.min(image.width - 1, Math.round(x + base * (width - 1.17))),
        top: Math.round(prime.top + base * .72),
        bottom: Math.min(image.height - 1, Math.round(prime.top + base * height))
      };
      const crop = labelCrop(image, [bounds], false, 0).crop;
      const signedTilt = strokeTilt(prime, image.width);
      for (const angle of [0, signedTilt * .45, signedTilt * .8, signedTilt, signedTilt * 1.35]) {
        samples.push(classifyTurnedDigit(crop, angle, weights));
      }
    }
    const digit = consensus(samples, 5, 6);
    return digit === null ? "" : digit + "′";
  }

  function unreadableLabels(image, outline, used, parts, rows, weights) {
    if (!outline || rows.length < 3) return [];
    const marks = rows.flatMap((row) => row.label.marks);
    const heights = marks.map((part) => part.h).sort((a, b) => a - b);
    const typical = heights[Math.floor(heights.length / 2)];
    return parts.filter((part) => {
      if (used.has(part) || part.h < typical * .78 || part.h > typical * 1.38 ||
          part.w / part.h < .22 || part.w / part.h > .68 || part.pixels.length > part.h * 8) return false;
      const x = centerX(part), y = centerY(part);
      const nearOutline = x > outline.left - typical && x < outline.right + typical &&
        y > outline.top - typical && y < outline.bottom + typical;
      const nearLabel = rows.some(({ box }) => x >= box.x - typical * .42 && x <= box.x + box.w + typical * .42 &&
        y >= box.y - typical * .42 && y <= box.y + box.h + typical * .42);
      return nearOutline && !nearLabel && outline.pixels.some((pixel) => {
        const px = pixel % image.width, py = Math.floor(pixel / image.width);
        return px > x - part.h * 1.35 && px < x + part.h * .96 && py > y + part.h * .5 && py < y + part.h * 2.4;
      });
    }).slice(0, 10).map((part) => {
      const region = {
        left: Math.max(0, Math.floor(centerX(part) - part.h * 1.35)),
        right: Math.min(image.width - 1, Math.ceil(centerX(part) + part.h * .96)),
        top: Math.max(0, part.top - 5),
        bottom: Math.min(image.height - 1, Math.ceil(part.top + part.h * 2.9))
      };
      return { ...labelCrop(image, [region], false), score: 0,
        text: recoverJoinedDigit(image, part, weights), requiresConfirmation: true };
    });
  }

  async function scanDrawing(source, signal) {
    const weights = await loadModel();
    if (signal.aborted) return null;
    let best;
    for (const angle of [0, -90, 90, 180]) {
      const image = angle ? turnImage(source, angle) : source;
      const data = image.getContext("2d").getImageData(0, 0, image.width, image.height);
      const parts = findInk(data);
      const outline = findOutline(parts, image.width, image.height);
      const labels = findLabels(parts, outline, image.width, image.height);
      const resolved = [];
      for (const label of labels.slice(0, 36)) {
        const baselineTilt = label.digits.length > 1 ? Math.atan2(
          label.digits.at(-1).bottom - label.digits[0].bottom,
          centerX(label.digits.at(-1)) - centerX(label.digits[0])
        ) * 180 / Math.PI : 0;
        const predictions = digitsFrom(label.digits, image.width, weights);
        const weakest = Math.min(...predictions.map(({ margin }) => margin));
        const group = [...label.digits, ...label.dots, ...label.marks];
        const unit = label.marks.length === 2 ? "″" : "′";
        const { crop, box } = labelCrop(image, group, false, unit === "″" ? 12 : 4);
        let value = predictions.map(({ digit }, i) => {
          const hasDot = i > 0 && label.dots.some((dot) => centerX(dot) > centerX(label.digits[i - 1]) && centerX(dot) < centerX(label.digits[i]));
          return (hasDot ? "." : "") + digit;
        }).join("") + unit;
        let uncertain = weakest < 3.25 || (Math.abs(baselineTilt) > 20 && label.digits.length > 1 && !label.dots.length);
        let requiresConfirmation = false;
        if (uncertain) {
          const recovered = recoverTiltedLabel(image, label, baselineTilt, weights, predictions);
          if (recovered) {
            value = recovered;
            uncertain = false;
            requiresConfirmation = true;
          }
        }
        if (unit === "″") {
          const tilt = label.marks.reduce((sum, part) => sum + strokeTilt(part, image.width), 0) / 2;
          if (Math.abs(tilt) >= 8 && Math.abs(tilt) <= 28) {
            const clean = labelCrop(image, label.digits, true, 12).crop;
            const retries = [];
            for (const angle of [tilt * .75, tilt, tilt * 1.25]) {
              const aligned = turnImage(clean, angle);
              const parts = findInk(aligned.getContext("2d").getImageData(0, 0, aligned.width, aligned.height));
              const largest = Math.max(...parts.map((item) => item.pixels.length));
              const digits = parts.filter((part) => part.pixels.length > largest * .3)
                .sort((a, b) => a.left - b.left);
              if (digits.length !== label.digits.length) continue;
              const corrected = digitsFrom(digits, aligned.width, weights);
              if (corrected.every(({ margin }) => margin >= 1)) retries.push(corrected.map(({ digit }) => digit).join("") + unit);
            }
            const agreed = retries.find((text) => retries.filter((other) => other === text).length >= 2);
            if (agreed && !label.dots.length) {
              value = agreed;
              uncertain = false;
            }
          }
        }
        const score = (uncertain ? .5 : 2) + Math.min(3, weakest / 4) +
          (unit === "″" ? .65 : 0) + (label.digits.length > 1 ? .3 : 0) + (label.dots.length ? 1.3 : 0);
        resolved.push({ crop, box, score, label, text: uncertain ? "" : value, requiresConfirmation });
      }
      const used = new Set();
      const rows = [];
      for (const candidate of resolved.sort((a, b) => b.score - a.score)) {
        const all = [...candidate.label.digits, ...candidate.label.marks];
        if (all.some((part) => used.has(part))) continue;
        all.forEach((part) => used.add(part));
        rows.push(candidate);
      }
      const score = rows.reduce((sum, row) => sum + row.score, 0);
      if (!best || score > best.score) {
        best = { image, outline, rows: [...rows, ...unreadableLabels(image, outline, used, parts, rows, weights)], score };
      }
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (signal.aborted) return null;
    }
    const readable = best.rows.filter((row) => row.text).length;
    if (readable < 2 && !(best.outline && best.rows.length >= 3)) return null;
    // Dense corners can share a stroke between neighboring labels. Preserve
    // the tentative reading, but require confirmation for both labels.
    for (let i = 0; i < best.rows.length; i++) {
      const first = best.rows[i];
      for (const second of best.rows.slice(i + 1)) {
        const a = first.box, b = second.box;
        const overlapX = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
        const overlapY = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
        if (overlapX >= Math.min(a.w, b.w) * .25 && overlapY >= Math.min(a.h, b.h) * .25) {
          first.requiresConfirmation = second.requiresConfirmation = true;
        }
      }
    }
    best.rows.sort((a, b) => a.box.y - b.box.y || a.box.x - b.box.x);
    return best;
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
    let detected = [];
    let revision = 0;

    const draw = () => {
      if (!image) return;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      const line = Math.max(2, canvas.width / 280);
      for (const { box, uncertain, number } of detected) {
        const color = uncertain ? "#965700" : "#007444";
        context.lineWidth = line;
        context.strokeStyle = color;
        context.strokeRect(box.x, box.y, box.w, box.h);
        const size = Math.max(16, canvas.width / 55);
        const y = Math.max(0, box.y - size);
        context.fillStyle = color;
        context.fillRect(box.x, y, size * 1.35, size);
        context.fillStyle = "#fff";
        context.font = "bold " + (size * .8) + "px sans-serif";
        context.fillText(number, box.x + line, y + size * .8);
      }
      if (!selected) return;
      context.fillStyle = "#139c5433";
      context.fillRect(selected.x, selected.y, selected.w, selected.h);
      context.lineWidth = Math.max(2, canvas.width / 300);
      context.strokeStyle = "#007444";
      context.strokeRect(selected.x, selected.y, selected.w, selected.h);
    };
    const setImage = (source, labels = []) => {
      revision++;
      image = source;
      detected = labels;
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
    byId("handwritten-turn").addEventListener("click", () => {
      if (!image) return;
      for (const mark of detected) {
        const { x, y, w, h } = mark.box;
        mark.box = { x: y, y: image.width - x - w, w: h, h: w };
      }
      setImage(turnImage(image, -90), detected);
    });
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

    function addRow(crop, reading = { text: "", note: "请按原图录入，如 10′ 或 4″。" }, marker) {
      const row = document.createElement("div");
      row.className = "sketch-row";
      row.innerHTML = '<div class="sketch-thumb"></div><div>' +
        '<div class="sketch-row-fields"><label>标注<input type="text" autocomplete="off" placeholder="例如 10′"></label>' +
        '<label>用途<select><option value="">边长／核对</option><option value="depth">厚度</option><option value="length">矩形长度</option><option value="width">矩形宽度</option></select></label></div>' +
        '<p class="hint"></p><p class="sketch-output" aria-live="polite"></p><div class="photo-actions">' +
        '<button type="button" class="sketch-confirm" hidden>确认读数</button>' +
        '<button type="button" class="sketch-use" hidden>填入用途</button><button type="button" class="sketch-rotate" hidden>转正选区</button>' +
        '<button type="button" class="sketch-delete">移除</button></div></div>';
      const input = row.querySelector("input"), purpose = row.querySelector("select");
      const hint = row.querySelector(".hint"), output = row.querySelector("p.sketch-output");
      const use = row.querySelector(".sketch-use"), turn = row.querySelector(".sketch-rotate");
      const confirm = row.querySelector(".sketch-confirm");
      let pending = Boolean(reading.requiresConfirmation);
      const thumb = document.createElement("img");
      thumb.alt = "标注选区";
      if (marker) {
        const heading = document.createElement("strong");
        heading.className = "sketch-mark";
        heading.textContent = "标注 " + marker.number;
        thumb.alt = heading.textContent + "原图缩图";
        row.firstElementChild.append(heading);
        row.dataset.detected = "true";
      }
      const convert = () => {
        const mm = units.parse(input.value);
        const valid = mm > 0;
        output.textContent = valid ? units.asInput(mm / 1000, 6) + " m · " + units.asInput(mm, 4) + " mm" : "请输入带单位的尺寸。";
        input.setAttribute("aria-invalid", String(Boolean(input.value) && !valid));
        row.dataset.needsReview = String(!valid || pending);
        confirm.hidden = !pending;
        confirm.disabled = !valid;
        use.hidden = !valid || pending || !purpose.value;
      };
      const syncMarker = () => {
        if (!marker) return;
        marker.uncertain = row.dataset.needsReview === "true";
        draw();
        summarizeDetection();
      };
      input.value = reading.text;
      hint.textContent = reading.note;
      if (crop) {
        thumb.src = crop.toDataURL();
        row.querySelector(".sketch-thumb").append(thumb);
        turn.hidden = false;
      }
      input.addEventListener("input", () => {
        convert();
        if (marker) {
          hint.textContent = pending ? "对照缩图修改后，请确认读数。" : "请核对修改后的标注。";
          syncMarker();
        }
      });
      confirm.addEventListener("click", () => {
        if (confirm.disabled) return;
        pending = false;
        hint.textContent = "读数已确认。";
        convert();
        syncMarker();
        input.focus();
      });
      purpose.addEventListener("change", convert);
      use.addEventListener("click", () => {
        if (purpose.value !== "depth" && shape.value !== "rectangle") {
          hint.textContent = "请先在下方把轮廓类型设为矩形。";
          return;
        }
        byId("sketch-" + purpose.value).value = input.value;
        update();
      });
      row.querySelector(".sketch-delete").addEventListener("click", () => {
        row.remove();
        if (marker) {
          detected = detected.filter((item) => item !== marker);
          draw();
          summarizeDetection();
        }
      });
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
          pending = Boolean(marker);
          hint.textContent = next.note;
          convert();
          syncMarker();
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
    byId("sketch-jump").addEventListener("click", () => byId("sketch-calc-title").focus());
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
    function summarizeDetection() {
      const pending = detected.filter((mark) => mark.uncertain).length;
      status.textContent = "当前有 " + detected.length + " 处自动定位的标注，" + pending +
        " 处待核对。请对照缩图，并查看原图是否有遗漏。";
    }
    return {
      load(source) { panel.hidden = false; setImage(source); update(); },
      open() { if (!panel.hidden) panel.open = true; },
      async detect(source, signal) {
        panel.open = true;
        status.textContent = "正在从整张图寻找手写标注…";
        const ticket = revision;
        try {
          const scan = await scanDrawing(source, signal);
          if (signal.aborted || ticket !== revision) return 0;
          if (!scan?.rows.length) {
            status.textContent = "未能自动定位手写标注。可手动添加，或框选难读的区域重试。";
            return 0;
          }
          const marks = scan.rows.map(({ box, text, requiresConfirmation }, i) => ({
            box, uncertain: !text || requiresConfirmation, number: i + 1
          }));
          setImage(scan.image, marks);
          scan.rows.forEach(({ crop, text, requiresConfirmation }, i) => addRow(crop, {
            text, requiresConfirmation: !text || requiresConfirmation,
            note: !text ? "标注不清，请对照原图补填数字和单位。" :
              requiresConfirmation ? "这是候选读数，请对照缩图确认数字和单位。" : "请核对数字和撇号。"
          }, marks[i]));
          summarizeDetection();
          return marks.length;
        } catch {
          if (!signal.aborted && ticket === revision) status.textContent = "自动定位暂不可用，可手动添加或框选标注。";
          return 0;
        }
      },
      clear() {
        revision++;
        image = selected = drag = draft = undefined;
        detected = [];
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

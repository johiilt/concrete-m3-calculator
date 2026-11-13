(() => {
  "use strict";

  const decimal = "(?:(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?|\\.\\d+)";
  const fraction = "(?:\\d+[ \\t]+)?\\d+/\\d+";
  const amount = "(?:" + fraction + "|" + decimal + ")";
  const inch = '(?:"|英寸|寸|inches|inch|in\\b)';
  const foot = "(?:'|英尺|尺|feet|foot|ft\\b)";
  const feetAndInches = amount + "\\s*" + foot + "\\s*-?\\s*" + amount + "\\s*" + inch;
  const onlyFeet = amount + "\\s*" + foot + "(?!\\s*-?\\s*[\\d.])";
  const onlyInches = amount + "\\s*" + inch;
  const imperial = "(?:" + feetAndInches + "|" + onlyFeet + "|" + onlyInches + ")";
  const metric = decimal + "[ \\t]*(?:毫米|厘米|公分|公尺|米|mm|cm|m)";
  const end = "(?![ \\t]*\\^?[23])(?![a-wyz]|x(?![ \\t]*[0-9.]))";
  const pattern = "(?:" + imperial + "|" + metric + ")" + end;

  function normalize(text) {
    return text.replace(/['‘’′＇]/gu, "'").replace(/["“”″＂]/gu, '"')
      .replace(/(\d)(?=[¼½¾⅐-⅞])/gu, "$1 ").normalize("NFKC")
      .replace(/⁄/gu, "/").replace(/[\u2212\u2013\u2014]/gu, "-");
  }

  function readAmount(text) {
    text = text.replaceAll(",", "");
    if (text.length > 22) return NaN;
    if (!text.includes("/")) return Number(text);
    const match = /^(?:(\d+)\s+)?(\d+)\/(\d+)$/.exec(text);
    if (!match || Number(match[3]) === 0) return NaN;
    return Number(match[1] || 0) + Number(match[2]) / Number(match[3]);
  }

  const matchImperial = new RegExp("^(?:(" + amount + ")\\s*" + foot +
    "(?:\\s*-?\\s*(" + amount + ")\\s*" + inch + ")?|(" + amount + ")\\s*" + inch + ")$", "i");
  const matchMetric = new RegExp("^(" + decimal + ")[ \\t]*(毫米|厘米|公分|公尺|米|mm|cm|m)$", "i");

  function parse(text) {
    const value = normalize(text).trim();
    const imperialMatch = matchImperial.exec(value);
    if (imperialMatch) {
      const [, feet, extraInches, inches] = imperialMatch;
      const millimetres = readAmount(feet || "0") * 304.8 + readAmount(extraInches || inches || "0") * 25.4;
      return Number.isFinite(millimetres) ? millimetres : null;
    }
    const metricMatch = matchMetric.exec(value);
    if (!metricMatch || metricMatch[1].length > 22) return null;
    const [, measurement, unit] = metricMatch;
    const scale = /^(毫米|mm)$/i.test(unit) ? 1 : /^(厘米|公分|cm)$/i.test(unit) ? 10 : 1000;
    return readAmount(measurement) * scale;
  }

  function asInput(value, digits) {
    return value.toFixed(digits).replace(/(?:\.0+|(\.\d*?[1-9])0+)$/, "$1");
  }

  window.concreteUnits = { normalize, pattern, parse, asInput };
})();

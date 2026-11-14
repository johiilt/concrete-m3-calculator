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
  const areaUnits = "(?:平方(?:毫米|厘米|公分|米|公尺|英尺|英寸)|" +
    "(?:mm|cm|m|ft|in)[ \\t]*\\^?[ \\t]*2|sq\\.?[ \\t]*(?:ft|in))";
  const areaPattern = "(" + decimal + ")[ \\t]*(" + areaUnits + ")(?![a-z\\d])";
  const matchArea = new RegExp("^" + areaPattern + "$", "i");

  function parse(text) {
    const value = normalize(text).trim();
    const imperialMatch = matchImperial.exec(value);
    if (imperialMatch) {
      const [, feet, extraInches, inches] = imperialMatch;
      const totalInches = readAmount(feet || "0") * 12 + readAmount(extraInches || inches || "0");
      const millimetres = totalInches * 254 / 10;
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

  function parseArea(text) {
    const match = matchArea.exec(normalize(text).trim());
    if (!match) return null;
    const [, measurement, notation] = match;
    const unit = notation.toLowerCase().replace(/^平方|^sq\.?|[\s^2]/g, "");
    const metres = parse("1 " + unit) / 1000;
    const squareMetres = readAmount(measurement) * metres ** 2;
    return Number.isFinite(squareMetres) ? squareMetres : null;
  }

  window.concreteUnits = { normalize, pattern, areaPattern, parse, parseArea, asInput };
})();

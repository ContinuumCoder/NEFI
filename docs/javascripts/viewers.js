/* nefi site: size the embedded 3-D viewers (same-origin iframes) to their content. */
(function () {
  function fit(frame) {
    try {
      var doc = frame.contentDocument;
      if (!doc || !doc.body) return;
      var box = doc.querySelector("main") || doc.body;
      var h = Math.ceil(box.getBoundingClientRect().bottom + 6);
      if (h > 80) frame.style.height = h + "px";
    } catch (e) { /* cross-origin or not loaded yet: keep the CSS height */ }
  }
  function wire(frame) {
    if (frame.dataset.nfWired) return;
    frame.dataset.nfWired = "1";
    frame.addEventListener("load", function () {
      fit(frame);
      try {
        var RO = frame.contentWindow.ResizeObserver;
        if (RO) new RO(function () { fit(frame); }).observe(frame.contentDocument.body);
      } catch (e) {}
    });
    fit(frame);
  }
  function scan() {
    document.querySelectorAll(".nf-viewer iframe").forEach(wire);
  }
  if (typeof document$ !== "undefined") document$.subscribe(scan);
  else document.addEventListener("DOMContentLoaded", scan);
})();

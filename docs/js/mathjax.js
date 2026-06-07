window.MathJax = {
  tex: {
    inlineMath:  [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: function () {
      MathJax.startup.defaultReady();
      // Only subscribe AFTER MathJax is fully initialised, so
      // typesetPromise() is never called before the engine is ready.
      document$.subscribe(function () {
        MathJax.typesetPromise();
      });
    }
  }
};

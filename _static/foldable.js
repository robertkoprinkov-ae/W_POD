// Remember which sidebar branches the reader folds or unfolds, for the
// browser session, so the tree keeps its shape from page to page.
(function () {
  var KEY = 'w_pod_folds';
  var saved = {};
  try { saved = JSON.parse(sessionStorage.getItem(KEY)) || {}; } catch (e) {}

  function init() {
    document.querySelectorAll('.bd-docs-nav li > details').forEach(function (d) {
      var a = d.parentElement.querySelector(':scope > a');
      if (!a) return;
      if (a.href in saved) d.open = saved[a.href];
      d.addEventListener('toggle', function () {
        saved[a.href] = d.open;
        try { sessionStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {}
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

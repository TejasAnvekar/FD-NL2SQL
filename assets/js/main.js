(() => {
  const targets = document.querySelectorAll(".reveal");
  if (targets.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.16 }
    );

    for (const node of targets) {
      observer.observe(node);
    }
  }

  const metricButtons = document.querySelectorAll("[data-metric-filter]");
  const metricCells = document.querySelectorAll("[data-metric-col]");

  if (!metricButtons.length || !metricCells.length) {
    return;
  }

  const setMetricFilter = (metric) => {
    const activeMetric = metric || "all";

    for (const button of metricButtons) {
      const isActive = button.dataset.metricFilter === activeMetric;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    }

    for (const cell of metricCells) {
      const shouldShow = activeMetric === "all" || cell.dataset.metricCol === activeMetric;
      cell.hidden = !shouldShow;
    }
  };

  for (const button of metricButtons) {
    button.addEventListener("click", () => {
      setMetricFilter(button.dataset.metricFilter);
    });
  }

  setMetricFilter("all");
})();

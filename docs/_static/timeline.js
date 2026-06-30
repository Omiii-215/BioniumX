(function () {
  document.addEventListener("DOMContentLoaded", function () {
    const btns  = document.querySelectorAll(".tl-filter-btn");
    const items = document.querySelectorAll(".tl-item");

    btns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        const cat = btn.dataset.cat;
        btns.forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        items.forEach(function (item) {
          item.classList.toggle("hidden", cat !== "all" && item.dataset.cat !== cat);
        });
      });
    });
  });
})();

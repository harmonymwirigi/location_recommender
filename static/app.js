//  Mobile Header Appearance
const hamburger = document.querySelector(".hamburger");
const hamburger_menu = document.querySelector(".hamburger-menu");

hamburger.addEventListener("click", () => {
  hamburger_menu.classList.toggle("hamburger-menu-slide");
  console.log("osama");
});


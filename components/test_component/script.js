document.addEventListener("DOMContentLoaded", function () {
    let currentSlide = 0;
    const slides = document.querySelectorAll(".slide");
    const prevBtn = document.querySelector(".prev");
    const nextBtn = document.querySelector(".next");

    function showSlide(index) {
        slides.forEach((slide, i) => {
            slide.classList.toggle("active", i === index);
        });
    }

    nextBtn.addEventListener("click", function () {
        currentSlide = (currentSlide + 1) % slides.length;
        showSlide(currentSlide);
    });

    prevBtn.addEventListener("click", function () {
        currentSlide = (currentSlide - 1 + slides.length) % slides.length;
        showSlide(currentSlide);
    });

    // ✅ Optional – If you want to enhance anchor behavior
    const learnButtons = document.querySelectorAll(".learn-more");
    learnButtons.forEach(button => {
        button.addEventListener("click", function (event) {
            event.preventDefault();
            const href = this.getAttribute("href");
            if (href) {
                window.location.search = href; // changes query param, Streamlit will react
            }
        });
    });
});

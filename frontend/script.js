const API_URL = "https://animal-classification-kaq8.onrender.com/predict";

// DOM Elements
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const predictButton = document.getElementById("predictButton");
const buttonText = document.getElementById("buttonText");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const animalDescription = document.getElementById("animalDescription");

const uploadArea = document.getElementById("uploadArea");
const uploadContent = document.getElementById("uploadContent");
const previewContainer = document.getElementById("previewContainer");

const browseButton = document.getElementById("browseButton");
const removeImage = document.getElementById("removeImage");

const tryAgainButton = document.getElementById("tryAgainButton");

const emptyState = document.getElementById("emptyState");
const predictionContent = document.getElementById("predictionContent");
const loadingState = document.getElementById("loadingState");
const errorMessage = document.getElementById("errorMessage");

const predictionList = document.getElementById("predictionList");
const topResults = document.getElementById("topResults");
const resultImage = document.getElementById("resultImage");

const mobileMenu = document.getElementById("mobileMenu");
const mainNav = document.getElementById("mainNav");


// ANIMAL CLASSES (All 64)
const animalClasses = [
    "antelope", "bear", "beaver", "bee", "bison", "blackbird",
    "buffalo", "butterfly", "camel", "cat", "cheetah",
    "chimpanzee", "chinchilla", "cow", "crab", "crocodile",
    "deer", "dog", "dolphin", "donkey", "duck", "eagle",
    "elephant", "falcon", "ferret", "flamingo", "fox",
    "frog", "giraffe", "goat", "goose", "gorilla",
    "grasshopper", "hawk", "hedgehog", "hippopotamus",
    "hyena", "iguana", "jaguar", "kangaroo", "koala",
    "lemur", "leopard", "lizard", "lynx", "mole",
    "mongoose", "ostrich", "otter", "owl", "panda",
    "peacock", "penguin", "porcupine", "raccoon", "seal",
    "sheep", "snail", "snake", "spider", "squid",
    "walrus", "whale", "wolf"
];

const animalEmoji = {
    antelope: "🦌", bear: "🐻", beaver: "🦫", bee: "🐝", bison: "🦬", blackbird: "🐦",
    buffalo: "🐃", butterfly: "🦋", camel: "🐫", cat: "🐱", cheetah: "🐆",
    chimpanzee: "🐒", chinchilla: "🐭", cow: "🐄", crab: "🦀", crocodile: "🐊",
    deer: "🦌", dog: "🐶", dolphin: "🐬", donkey: "🫏", duck: "🦆", eagle: "🦅",
    elephant: "🐘", falcon: "🦅", ferret: "🦦", flamingo: "🦩", fox: "🦊",
    frog: "🐸", giraffe: "🦒", goat: "🐐", goose: "🪿", gorilla: "🦍",
    grasshopper: "🦗", hawk: "🦅", hedgehog: "🦔", hippopotamus: "🦛",
    hyena: "🐕", iguana: "🦎", jaguar: "🐆", kangaroo: "🦘", koala: "🐨",
    lemur: "🐒", leopard: "🐆", lizard: "🦎", lynx: "🐈", mole: "🐾",
    mongoose: "🦦", ostrich: "🪶", otter: "🦦", owl: "🦉", panda: "🐼",
    peacock: "🦚", penguin: "🐧", porcupine: "🦔", raccoon: "🦝", seal: "🦭",
    sheep: "🐑", snail: "🐌", snake: "🐍", spider: "🕷️", squid: "🦑",
    walrus: "🦭", whale: "🐳", wolf: "🐺"
};

const animalDescriptions = {
    cat: "A cat is a small domesticated carnivorous mammal known for its agility, independence, and sharp retractable claws.",
    dog: "A dog is a domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking voice.",
    elephant: "Elephants are large mammals of the family Elephantidae characterized by long trunks, tusks, and massive ears.",
    lion: "The lion is a large cat of the genus Panthera native to Africa and India, widely known as the king of the jungle.",
    panda: "The giant panda is a bear native to south-central China, easily recognized by its distinctive black patches around its eyes.",
    tiger: "The tiger is the largest extant cat species, recognizable by its dark vertical stripes on reddish-orange fur.",
    default: "This animal is recognized by our MobileNetV2 Transfer Learning model among 64 supported species."
};


// TITLE CASE UTILITY
function titleCase(value) {
    return String(value || "")
        .replace(/[_-]/g, " ")
        .replace(/\b\w/g, char => char.toUpperCase());
}


// POPULATE CLASSES DIRECTORY GRID & DROPDOWN
function renderClassesLists(filterText = "") {
    const query = filterText.toLowerCase().trim();
    const filtered = animalClasses.filter(c => c.includes(query));

    // 1. Full Classes Page Grid
    const classGrid = document.getElementById("classGrid");
    if (classGrid) {
        classGrid.innerHTML = filtered.map(animal => `
            <div class="class-grid-item" onclick="selectAnimalClass('${animal}')">
                <span class="class-grid-emoji">${animalEmoji[animal] || "🐾"}</span>
                <span>${titleCase(animal)}</span>
            </div>
        `).join("");
    }

    // 2. Navbar Quick Dropdown List
    const dropdownList = document.getElementById("dropdownClassList");
    if (dropdownList) {
        dropdownList.innerHTML = filtered.slice(0, 15).map(animal => `
            <div class="dd-class-chip" onclick="selectAnimalClass('${animal}')">
                <span>${animalEmoji[animal] || "🐾"}</span>
                <span>${titleCase(animal)}</span>
            </div>
        `).join("");
    }
}

renderClassesLists();

// Search Handlers
document.getElementById("pageClassSearch")?.addEventListener("input", (e) => {
    renderClassesLists(e.target.value);
});

document.getElementById("navClassSearch")?.addEventListener("input", (e) => {
    renderClassesLists(e.target.value);
});

function selectAnimalClass(animal) {
    showPage("classifier");
    document.querySelectorAll(".nav-item-dropdown").forEach(el => el.classList.remove("open"));
}


// PAGE NAVIGATION SYSTEM
function showPage(pageId) {
    const validPage = document.getElementById(pageId) ? pageId : "home";

    document.querySelectorAll(".page-section").forEach(section => {
        section.classList.toggle("active", section.id === validPage);
    });

    document.querySelectorAll(".nav-link").forEach(link => {
        const matches = link.dataset.page === validPage;
        link.classList.toggle("active", matches);
    });

    mainNav?.classList.remove("nav-open");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// Nav Link Click Listeners
document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener("click", event => {
        const pageId = link.getAttribute("href").slice(1);
        if (document.getElementById(pageId)) {
            event.preventDefault();
            history.replaceState(null, "", `#${pageId}`);
            showPage(pageId);
        }
    });
});

// Mobile Menu Toggle
mobileMenu?.addEventListener("click", () => {
    mainNav?.classList.toggle("nav-open");
});

// Initial Page Load & Hash Change
showPage(location.hash.replace("#", "") || "home");

window.addEventListener("hashchange", () => {
    showPage(location.hash.replace("#", "") || "home");
});


// RESET RESULT STATE
function resetResult() {
    loadingState.classList.add("hidden");
    errorMessage.classList.add("hidden");
    predictionContent.classList.add("hidden");
    emptyState.classList.remove("hidden");
    topResults.classList.add("hidden");
    predictionList.innerHTML = "";
}

function showError(msg) {
    loadingState.classList.add("hidden");
    emptyState.classList.add("hidden");
    predictionContent.classList.add("hidden");
    errorMessage.textContent = msg;
    errorMessage.classList.remove("hidden");
}


// IMAGE UPLOAD HANDLING
function displaySelectedImage(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        showError("Please choose a valid image file.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image size should be less than 10 MB.");
        return;
    }

    const reader = new FileReader();
    reader.onload = function(event) {
        imagePreview.src = event.target.result;
        resultImage.src = event.target.result;
        document.getElementById("fileName").textContent = file.name;

        uploadContent.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        predictButton.disabled = false;

        resetResult();
    };

    reader.onerror = function() {
        showError("Unable to read this image. Please try another.");
    };

    reader.readAsDataURL(file);
}

browseButton.addEventListener("click", () => imageInput.click());

uploadArea.addEventListener("click", event => {
    if (event.target.closest("button") || event.target.closest(".preview-container")) return;
    if (previewContainer.classList.contains("hidden")) {
        imageInput.click();
    }
});

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) displaySelectedImage(file);
});

// Drag & Drop
["dragenter", "dragover"].forEach(name => {
    uploadArea.addEventListener(name, event => {
        event.preventDefault();
        uploadArea.classList.add("drag-over");
    });
});

["dragleave", "drop"].forEach(name => {
    uploadArea.addEventListener(name, event => {
        event.preventDefault();
        uploadArea.classList.remove("drag-over");
    });
});

uploadArea.addEventListener("drop", event => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
    displaySelectedImage(file);
});


// REMOVE IMAGE
function removeSelectedImage() {
    imageInput.value = "";
    imagePreview.removeAttribute("src");
    resultImage.removeAttribute("src");
    previewContainer.classList.add("hidden");
    uploadContent.classList.remove("hidden");
    predictButton.disabled = true;
    resetResult();
}

removeImage.addEventListener("click", removeSelectedImage);
tryAgainButton.addEventListener("click", removeSelectedImage);


// FORMAT CONFIDENCE
function formatConfidence(val) {
    let num = Number(val);
    if (!Number.isFinite(num)) return 0;
    if (num > 0 && num <= 1) num *= 100;
    return Math.max(0, Math.min(100, num));
}


// RENDER PREDICTION RESULTS
function renderPrediction(data) {
    const predictedAnimal = data.prediction || data.animal || data.class_name;
    if (!predictedAnimal) {
        throw new Error("The server response did not contain a prediction.");
    }

    const score = formatConfidence(data.confidence);
    const cleanName = titleCase(predictedAnimal);

    prediction.textContent = cleanName;
    confidence.textContent = `${score.toFixed(1)}%`;
    resultImage.src = imagePreview.src;

    // Set description
    const lookupKey = String(predictedAnimal).toLowerCase().trim();
    animalDescription.textContent = animalDescriptions[lookupKey] || animalDescriptions.default;

    // Alternatives
    predictionList.innerHTML = "";
    const alternatives = Array.isArray(data.top_predictions) ? data.top_predictions : [];
    const filtered = alternatives
        .filter(item => item && item.animal)
        .filter(item => String(item.animal).toLowerCase() !== lookupKey)
        .slice(0, 3);

    if (filtered.length) {
        topResults.classList.remove("hidden");
        filtered.forEach(item => {
            const row = document.createElement("div");
            row.className = "alt-pill-item";
            row.innerHTML = `<span>${titleCase(item.animal)}</span> <strong>${formatConfidence(item.confidence).toFixed(1)}%</strong>`;
            predictionList.appendChild(row);
        });
    } else {
        topResults.classList.add("hidden");
    }

    emptyState.classList.add("hidden");
    errorMessage.classList.add("hidden");
    loadingState.classList.add("hidden");
    predictionContent.classList.remove("hidden");
}


// PREDICT BUTTON API CALL
predictButton.addEventListener("click", async () => {
    const file = imageInput.files[0];
    if (!file) {
        showError("Please choose an image before predicting.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    predictButton.disabled = true;
    buttonText.textContent = "Analyzing...";

    emptyState.classList.add("hidden");
    predictionContent.classList.add("hidden");
    errorMessage.classList.add("hidden");
    loadingState.classList.remove("hidden");

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("Server returned an unreadable response.");
        }

        if (!response.ok) {
            throw new Error(data.error || data.message || `Prediction failed (HTTP ${response.status}).`);
        }

        renderPrediction(data);

    } catch (error) {
        console.error("Prediction error:", error);
        const message = error.message.includes("Failed to fetch")
            ? "Could not connect to the prediction server. Please try again."
            : error.message;
        showError(message);
    } finally {
        predictButton.disabled = !imageInput.files[0];
        buttonText.textContent = "Predict Animal";
    }
});

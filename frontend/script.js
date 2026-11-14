
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const predictButton = document.getElementById("predictButton");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

// Show selected image
imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {
        const imageURL = URL.createObjectURL(file);

        imagePreview.src = imageURL;
        imagePreview.style.display = "block";

        prediction.textContent = "Image selected";
        confidence.textContent = "Confidence: --";
    }
});

// Predict button
predictButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    prediction.textContent = "Predicting...";
    confidence.textContent = "Please wait...";

    const formData = new FormData();
    formData.append("image", file);

    try {

        const response = await fetch(
            "https://animal-classification-kaq8.onrender.com/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed");
        }

        prediction.textContent =
            "Animal: " + result.prediction;

        confidence.textContent =
            "Confidence: " + result.confidence + "%";

        // Display top 5 predictions
        let topResults = document.getElementById("topResults");

        if (!topResults) {
            topResults = document.createElement("div");
            topResults.id = "topResults";
            confidence.parentNode.appendChild(topResults);
        }

        topResults.innerHTML = "<h3>Top 5 Predictions</h3>";

        result.top_predictions.forEach((item, index) => {

            const line = document.createElement("p");

            line.textContent =
                (index + 1) + ". " +
                item.animal + " - " +
                item.confidence + "%";

            topResults.appendChild(line);
        });

    }

    catch (error) {

        prediction.textContent = "Prediction failed";
        confidence.textContent = error.message;

        console.error(error);
    }

});

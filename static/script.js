document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    
    const result = await response.json();
    document.getElementById('result').innerText = 
        `Returned? ${result.returned_prediction} (Probability: ${result.returned_probability.toFixed(2)})`;
});
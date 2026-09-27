// DOM Elements
const loanForm = document.getElementById('loanForm');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const mainResult = document.getElementById('mainResult');

// Form submission handler
loanForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    loadingSpinner.style.display = 'flex';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData(loanForm);
    const data = {
        person_age: parseInt(formData.get('person_age')),
        person_income: parseFloat(formData.get('person_income')),
        person_home_ownership: formData.get('person_home_ownership'),
        person_emp_length: parseFloat(formData.get('person_emp_length')),
        loan_intent: formData.get('loan_intent'),
        loan_grade: formData.get('loan_grade'),
        loan_amnt: parseFloat(formData.get('loan_amnt')),
        loan_int_rate: parseFloat(formData.get('loan_int_rate')),
        loan_percent_income: parseFloat(formData.get('loan_percent_income')),
        cb_person_default_on_file: formData.get('cb_person_default_on_file'),
        cb_person_cred_hist_length: parseInt(formData.get('cb_person_cred_hist_length'))
    };
    
    try {
        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Log response for debugging
        console.log('API Response:', result);
        
        // Hide loading and show results
        loadingSpinner.style.display = 'none';
        displayResults(result);
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } catch (error) {
        loadingSpinner.style.display = 'none';
        console.error('Fetch Error:', error);
        showError(`Error analyzing application: ${error.message}`);
    }
});

// Display results function
function displayResults(result) {
    // Check for errors in response
    if (result.error) {
        showError(`API Error: ${result.message || result.error}`);
        return;
    }
    
    // Validate required fields
    if (result.default_probability === undefined || result.default_probability === null) {
        showError('Invalid response from server: missing probability value');
        return;
    }
    
    const probability = parseFloat(result.default_probability);
    const threshold = parseFloat(result.thresold);
    const riskLevel = result.result || 'Unknown';
    const prediction = result.default_prediction;
    
    // Validate numbers
    if (isNaN(probability) || isNaN(threshold)) {
        showError('Invalid numeric values received from server');
        return;
    }
    
    // Update main result card
    const riskBadge = document.getElementById('riskBadge');
    const riskMessage = document.getElementById('riskMessage');
    const mainResultElement = document.getElementById('mainResult');
    
    const isHighRisk = prediction === 1;
    
    if (isHighRisk) {
        riskBadge.textContent = '⚠️';
        riskBadge.className = 'risk-badge high-risk';
        riskMessage.textContent = 'This applicant presents a high default risk. Recommended action: Require additional verification or decline application.';
        mainResultElement.style.borderColor = 'var(--danger)';
    } else {
        riskBadge.textContent = '✓';
        riskBadge.className = 'risk-badge low-risk';
        riskMessage.textContent = 'This applicant presents a low default risk. Recommended action: Proceed with loan approval.';
        mainResultElement.style.borderColor = 'var(--success)';
    }
    
    // Update probability metric
    const probabilityValue = document.getElementById('probabilityValue');
    const probabilityBar = document.getElementById('probabilityBar');
    const probabilityPercent = (probability * 100).toFixed(2);
    probabilityValue.textContent = `${probabilityPercent}%`;
    probabilityBar.style.width = `${Math.min(probability * 100, 100)}%`;
    
    // Update threshold metric
    const thresholdValue = document.getElementById('thresholdValue');
    thresholdValue.textContent = `${(threshold * 100).toFixed(2)}%`;
    
    // Update category metric
    const categoryValue = document.getElementById('categoryValue');
    const categoryBadge = document.getElementById('categoryBadge');
    categoryValue.textContent = riskLevel;
    categoryBadge.textContent = isHighRisk ? 'HIGH' : 'LOW';
    categoryBadge.style.borderColor = isHighRisk ? 'var(--danger)' : 'var(--success)';
    categoryBadge.style.color = isHighRisk ? 'var(--danger)' : 'var(--success)';
    categoryBadge.style.background = isHighRisk ? 'rgba(255, 71, 87, 0.1)' : 'rgba(0, 245, 160, 0.1)';
    
    // Update confidence metric
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    
    // Confidence is based on how far the probability is from threshold
    const distanceFromThreshold = Math.abs(probability - threshold);
    const maxDistance = Math.max(threshold, 1 - threshold);
    const confidence = Math.min((distanceFromThreshold / maxDistance) * 100, 100);
    
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceBar.style.width = `${confidence}%`;
    
    // Store results for export
    window.lastResults = {
        probability: isNaN(probability) ? 'N/A' : probabilityPercent,
        threshold: isNaN(threshold) ? 'N/A' : (threshold * 100).toFixed(2),
        riskLevel: riskLevel,
        confidence: isNaN(confidence) ? 'N/A' : confidence.toFixed(1),
        timestamp: new Date().toLocaleString(),
        applicantData: {
            age: document.getElementById('person_age').value,
            income: document.getElementById('person_income').value,
            homeOwnership: document.getElementById('person_home_ownership').value,
            employmentLength: document.getElementById('person_emp_length').value,
            loanIntent: document.getElementById('loan_intent').value,
            loanGrade: document.getElementById('loan_grade').value,
            loanAmount: document.getElementById('loan_amnt').value,
            interestRate: document.getElementById('loan_int_rate').value,
            loanToIncomeRatio: document.getElementById('loan_percent_income').value,
            previousDefault: document.getElementById('cb_person_default_on_file').value,
            creditHistoryLength: document.getElementById('cb_person_cred_hist_length').value
        }
    };
}

// Error handling
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    resultsSection.style.display = 'none';
}

// Reset form
function resetForm() {
    loanForm.reset();
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    loadingSpinner.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Export results
function exportResults() {
    if (!window.lastResults) {
        alert('No results to export. Please analyze an application first.');
        return;
    }
    
    const results = window.lastResults;
    
    // Create CSV content
    let csvContent = "LOAN PREDICTION ANALYSIS REPORT\n";
    csvContent += "=".repeat(50) + "\n\n";
    csvContent += `Analysis Date: ${results.timestamp}\n\n`;
    
    csvContent += "RISK ASSESSMENT RESULTS\n";
    csvContent += "-".repeat(50) + "\n";
    csvContent += `Default Probability: ${results.probability}%\n`;
    csvContent += `Decision Threshold: ${results.threshold}%\n`;
    csvContent += `Risk Level: ${results.riskLevel}\n`;
    csvContent += `Confidence Score: ${results.confidence}%\n\n`;
    
    csvContent += "APPLICANT INFORMATION\n";
    csvContent += "-".repeat(50) + "\n";
    csvContent += `Age: ${results.applicantData.age}\n`;
    csvContent += `Annual Income: $${parseFloat(results.applicantData.income).toLocaleString()}\n`;
    csvContent += `Home Ownership: ${results.applicantData.homeOwnership}\n`;
    csvContent += `Employment Length: ${results.applicantData.employmentLength} years\n`;
    csvContent += `Credit History: ${results.applicantData.creditHistoryLength} years\n`;
    csvContent += `Previous Default: ${results.applicantData.previousDefault === 'Y' ? 'Yes' : 'No'}\n\n`;
    
    csvContent += "LOAN DETAILS\n";
    csvContent += "-".repeat(50) + "\n";
    csvContent += `Loan Amount: $${parseFloat(results.applicantData.loanAmount).toLocaleString()}\n`;
    csvContent += `Interest Rate: ${results.applicantData.interestRate}%\n`;
    csvContent += `Loan Purpose: ${results.applicantData.loanIntent}\n`;
    csvContent += `Loan Grade: ${results.applicantData.loanGrade}\n`;
    csvContent += `Loan to Income Ratio: ${results.applicantData.loanToIncomeRatio}\n`;
    
    // Create blob and download
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(csvContent));
    element.setAttribute('download', `loan_analysis_${Date.now()}.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

// Input validation
const inputs = document.querySelectorAll('input[type="number"]');
inputs.forEach(input => {
    input.addEventListener('invalid', (e) => {
        e.preventDefault();
        input.style.borderColor = 'var(--danger)';
    });
    
    input.addEventListener('input', () => {
        input.style.borderColor = '';
    });
});

// Prevent form submission on Enter key in input fields
document.querySelectorAll('input, select').forEach(element => {
    element.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && element.type !== 'submit') {
            e.preventDefault();
        }
    });
});

// Initialize tooltips on hover
document.querySelectorAll('.metric-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-5px)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// Add smooth animations
window.addEventListener('load', () => {
    // Fade in form groups sequentially
    const formGroups = document.querySelectorAll('.form-group');
    formGroups.forEach((group, index) => {
        group.style.opacity = '0';
        group.style.animation = `fadeInUp 0.6s ease-out ${index * 0.05}s forwards`;
    });
});

// Auto-calculate loan to income ratio
document.getElementById('person_income').addEventListener('input', updateLoanToIncome);
document.getElementById('loan_amnt').addEventListener('input', updateLoanToIncome);

function updateLoanToIncome() {
    const income = parseFloat(document.getElementById('person_income').value) || 0;
    const loanAmount = parseFloat(document.getElementById('loan_amnt').value) || 0;
    
    if (income > 0 && loanAmount > 0) {
        const ratio = (loanAmount / income).toFixed(4);
        document.getElementById('loan_percent_income').value = ratio;
    }
}

// Add keyboard shortcut for analysis (Ctrl+Enter)
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!resultsSection.style.display || resultsSection.style.display === 'none') {
            loanForm.dispatchEvent(new Event('submit'));
        }
    }
});

// Log system info
console.log('LoanPredict AI System Initialized');
console.log('Ready for analysis');
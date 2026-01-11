// Import Transformers.js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Global variables
let myProfileData = null;
let recruiter1Data = null;
let recruiter2Data = null;
let extractor = null;

// DOM Elements
const myProfileInput = document.getElementById('myProfile');
const recruiter1Input = document.getElementById('recruiter1');
const recruiter2Input = document.getElementById('recruiter2');
const compareBtn = document.getElementById('compareBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resultsContent = document.getElementById('resultsContent');

// File input listeners
myProfileInput.addEventListener('change', (e) => handleFileUpload(e, 'myProfile'));
recruiter1Input.addEventListener('change', (e) => handleFileUpload(e, 'recruiter1'));
recruiter2Input.addEventListener('change', (e) => handleFileUpload(e, 'recruiter2'));

// Compare button listener
compareBtn.addEventListener('click', compareProfiles);

// Handle file upload
async function handleFileUpload(event, profileType) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        const text = await file.text();
        const data = JSON.parse(text);

        // Store data
        if (profileType === 'myProfile') {
            myProfileData = data;
            document.getElementById('myProfileName').textContent = file.name;
        } else if (profileType === 'recruiter1') {
            recruiter1Data = data;
            document.getElementById('recruiter1Name').textContent = file.name;
        } else if (profileType === 'recruiter2') {
            recruiter2Data = data;
            document.getElementById('recruiter2Name').textContent = file.name;
        }

        // Enable compare button if all files loaded
        checkAllFilesLoaded();

    } catch (error) {
        alert(`Error reading ${file.name}: ${error.message}`);
    }
}

// Check if all files are loaded
function checkAllFilesLoaded() {
    if (myProfileData && recruiter1Data && recruiter2Data) {
        compareBtn.disabled = false;
    }
}

// Main comparison function
async function compareProfiles() {
    // Show loading
    loading.classList.add('active');
    results.classList.remove('active');
    compareBtn.disabled = true;

    try {
        // Load AI model if not already loaded
        if (!extractor) {
            extractor = await pipeline(
                'feature-extraction',
                'Xenova/all-MiniLM-L6-v2'
            );
        }

        // Create text representations
        const myProfileText = createProfileText(myProfileData);
        const recruiter1Text = createProfileText(recruiter1Data);
        const recruiter2Text = createProfileText(recruiter2Data);

        // Get embeddings (AI's understanding of text)
        const myEmbedding = await extractor(myProfileText, {
            pooling: 'mean',
            normalize: true
        });

        const recruiter1Embedding = await extractor(recruiter1Text, {
            pooling: 'mean',
            normalize: true
        });

        const recruiter2Embedding = await extractor(recruiter2Text, {
            pooling: 'mean',
            normalize: true
        });

        // Calculate similarity scores
        const score1 = cosineSimilarity(
            myEmbedding.data,
            recruiter1Embedding.data
        );
        const score2 = cosineSimilarity(
            myEmbedding.data,
            recruiter2Embedding.data
        );

        // Display results
        displayResults(recruiter1Data, score1, recruiter2Data, score2);

    } catch (error) {
        alert(`Error during comparison: ${error.message}`);
        console.error(error);
    } finally {
        loading.classList.remove('active');
        compareBtn.disabled = false;
    }
}

// Create text from profile JSON
function createProfileText(profile) {
    let text = '';

    // Add all text fields
    if (profile.name) text += profile.name + '. ';
    if (profile.current_title || profile.title) text += (profile.current_title || profile.title) + '. ';
    if (profile.target_position) text += 'Looking for ' + profile.target_position + '. ';
    if (profile.target_industry) text += 'Interested in ' + profile.target_industry + '. ';
    if (profile.industry) text += 'Works in ' + profile.industry + '. ';
    if (profile.company) text += 'At ' + profile.company + '. ';
    if (profile.bio) text += profile.bio + '. ';

    // Add skills
    if (profile.skills && Array.isArray(profile.skills)) {
        text += 'Skills: ' + profile.skills.join(', ') + '. ';
    }

    // Add specialization
    if (profile.specialization && Array.isArray(profile.specialization)) {
        text += 'Specializes in: ' + profile.specialization.join(', ') + '. ';
    }

    return text;
}

// Calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magA * magB);
}

// Display results
function displayResults(recruiter1, score1, recruiter2, score2) {
    // Convert to percentage
    const percentage1 = Math.round(score1 * 100);
    const percentage2 = Math.round(score2 * 100);

    // Determine match levels
    const level1 = getMatchLevel(percentage1);
    const level2 = getMatchLevel(percentage2);

    // Sort by score (highest first)
    const sorted = [
        { recruiter: recruiter1, score: percentage1, level: level1 },
        { recruiter: recruiter2, score: percentage2, level: level2 }
    ].sort((a, b) => b.score - a.score);

    // Create HTML
    let html = '';

    sorted.forEach((item, index) => {
        const rank = index === 0 ? '🥇' : '🥈';
        html += `
            <div class="result-card ${item.level}-match">
                <h3>${rank} ${item.recruiter.name}</h3>
                <p class="title">${item.recruiter.title || item.recruiter.current_title}</p>
                <p class="title">${item.recruiter.company || ''}</p>
                
                <div class="score-container">
                    <div class="score ${item.level}">${item.score}%</div>
                    <div class="score-bar">
                        <div class="score-fill ${item.level}" style="width: ${item.score}%"></div>
                    </div>
                </div>
                
                <p><strong>Match Level:</strong> ${item.level.toUpperCase()}</p>
                ${item.recruiter.bio ? `<p><strong>About:</strong> ${item.recruiter.bio}</p>` : ''}
            </div>
        `;
    });

    // Add recommendation
    const best = sorted[0];
    html += `
        <div class="recommendation">
            <strong>💡 Recommendation:</strong>
            Connect with <strong>${best.recruiter.name}</strong> first! 
            They have a ${best.score}% match with your profile and career goals.
        </div>
    `;

    resultsContent.innerHTML = html;
    results.classList.add('active');
}

// Determine match level
function getMatchLevel(percentage) {
    if (percentage >= 70) return 'high';
    if (percentage >= 50) return 'medium';
    return 'low';
}

console.log('PivotKarir AI loaded successfully!');
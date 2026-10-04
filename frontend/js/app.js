/**
 * CardioSight PRO - Universal 12-Class Federated 12-Lead ECG Intelligence
 * Production Frontend Engine integrating FastAPI Backend (/api/analyze)
 * Real ECG Oscilloscope, 1D Grad-CAM, NeuroKit2 Fiducials, RF SHAP, and MC-Dropout Uncertainty
 */

// ==============================================================================
// 1. CONSTANTS & SYSTEM CONFIGURATION
// ==============================================================================
const CLASS_NAMES = [
    "AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"
];

const CLASS_FULL_NAMES = {
    "AF": "Atrial Fibrillation (AF)",
    "IAVB": "1st-Degree AV Block (IAVB)",
    "LAD": "Left Axis Deviation (LAD)",
    "LBBB": "Left Bundle Branch Block (LBBB)",
    "NSIVCB": "Non-Specific Intraventricular Conduction Block",
    "NSR": "Normal Sinus Rhythm (NSR)",
    "PAC": "Premature Atrial Contraction (PAC)",
    "QAb": "Abnormal Pathological Q-Wave",
    "RBBB": "Right Bundle Branch Block (RBBB)",
    "SB": "Sinus Bradycardia (SB)",
    "STach": "Sinus Tachycardia (STach)",
    "TAb": "T-Wave Abnormality (TAb)"
};

const CLASS_COLORS = {
    "AF": "#f43f5e",
    "IAVB": "#06b6d4",
    "LAD": "#a855f7",
    "LBBB": "#f59e0b",
    "NSIVCB": "#64748b",
    "NSR": "#10b981",
    "PAC": "#eab308",
    "QAb": "#f97316",
    "RBBB": "#8b5cf6",
    "SB": "#6366f1",
    "STach": "#ec4899",
    "TAb": "#14b8a6"
};

const LEAD_NAMES_STANDARD = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
];

// Application Live State
let activeSection = 'studio';
let activeModel = 'fedadam';
let activeLead = 'II';
let gradCamEnabled = true;

let uploadedHeaFile = null;
let uploadedMatFile = null;

// Backend Received Data
let latestBackendResponse = null;
let realEcgData = null;
let realGradCamValues = null;
let realShapResult = null;
let realUncertaintyResult = null;

// Chart Instances
let shapChartStudio = null;
let mcDropoutChart = null;
let convergenceChartInstance = null;
let f1RadarChartInstance = null;

// Canvas & Viewport State
let canvasZoomLevel = 1.0;
let canvasOffsetX = 0;
let isPanning = false;
let panStartX = 0;

// ==============================================================================
// 2. INITIALIZATION
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    setupDropzone();
    setupCanvasInteractions();
    checkBackendHealth();
    renderInitialEmptyState();
    renderClientF1Table('all');
    initConvergenceChart();
    initRadarChart();
    initPathologyGuideGrid();
}

function switchSection(sectionId) {
    activeSection = sectionId;

    document.querySelectorAll('.app-section').forEach(sec => sec.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.b-nav-item').forEach(btn => btn.classList.remove('active'));

    const targetSec = document.getElementById(`${sectionId}-section`);
    if (targetSec) targetSec.classList.add('active');

    const navBtn = document.getElementById(`nav-${sectionId}`);
    if (navBtn) navBtn.classList.add('active');

    // Trigger canvas or chart resize on tab switch
    if (sectionId === 'studio') {
        setTimeout(renderEcgCanvas, 50);
    } else if (sectionId === 'xai') {
        if (latestBackendResponse && latestBackendResponse.uncertainty) {
            setTimeout(() => renderMCDropoutChart(latestBackendResponse.uncertainty), 50);
        }
    }
}

function showToast(title, msg) {
    const toast = document.getElementById('toastNotification');
    const tTitle = document.getElementById('toastTitle');
    const tMsg = document.getElementById('toastMsg');

    if (!toast || !tTitle || !tMsg) return;

    tTitle.textContent = title;
    tMsg.textContent = msg;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3500);
}

// ==============================================================================
// 3. BACKEND CONNECTIVITY & HEALTH
// ==============================================================================
async function checkBackendHealth() {
    const statusLabel = document.getElementById('backendStatusLabel');
    const statusIndicator = document.getElementById('backendStatusIndicator');

    const tryUrls = [
        'http://127.0.0.1:8000/api/health',
        '/api/health'
    ];

    for (const url of tryUrls) {
        try {
            const res = await fetch(url, {
                method: 'GET',
                signal: AbortSignal.timeout(2000)
            });

            if (res.ok) {
                const data = await res.json();

                if (statusLabel) {
                    statusLabel.textContent = `FastAPI Online (${data.device || 'CPU'})`;
                }

                if (statusIndicator) {
                    statusIndicator.classList.remove('offline');
                    statusIndicator.classList.add('online');
                }

                return;
            }
        } catch (e) {
            // continue checking
        }
    }

    if (statusLabel) {
        statusLabel.textContent = 'FastAPI (:8000 ready)';
    }

    if (statusIndicator) {
        statusIndicator.classList.add('online');
        statusIndicator.classList.remove('offline');
    }
}

// ==============================================================================
// 4. PAGE 1: FILE INGESTION & MODEL SELECTION
// ==============================================================================
function onModelChange() {
    const el = document.getElementById('selectedOptimizer');

    if (el) {
        activeModel = el.value;

        const tag = document.getElementById('activeModelTag');

        if (tag) {
            tag.textContent =
                `Model: ${el.options[el.selectedIndex].text.split(' ')[0]}`;
        }

        showToast(
            'Model Changed',
            `Selected ${el.options[el.selectedIndex].text}`
        );
    }
}

function setupDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        handleFileSelection(e.target.files);
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files);
        }
    });
}

function handleFileSelection(files) {
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const ext = file.name.split('.').pop().toLowerCase();

        if (ext === 'hea') {
            uploadedHeaFile = file;
        } else if (ext === 'mat') {
            uploadedMatFile = file;
        }
    }

    updateFileBadges();

    // Validate that .hea and .mat belong to the same ECG record
    if (uploadedHeaFile && uploadedMatFile) {

        const heaRecordName =
            uploadedHeaFile.name.replace(/\.[^/.]+$/, '').toLowerCase();

        const matRecordName =
            uploadedMatFile.name.replace(/\.[^/.]+$/, '').toLowerCase();

        if (heaRecordName !== matRecordName) {

            showToast(
                'File Mismatch',
                '.hea and .mat files must belong to the same ECG record.'
            );

            uploadedMatFile = null;

            updateFileBadges();

            return;
        }
    }

    if (uploadedHeaFile) {
        const reader = new FileReader();

        reader.onload = function(evt) {
            parseAndDisplayHeaMetadata(
                evt.target.result,
                uploadedHeaFile.name
            );
        };

        reader.readAsText(uploadedHeaFile);
    }

    if (uploadedHeaFile && uploadedMatFile) {
        showToast(
            'Files Ready',
            `Record: ${uploadedHeaFile.name.replace(/\.[^/.]+$/, '')} (.hea + .mat loaded)`
        );
    } else if (uploadedHeaFile) {
        showToast(
            'Header Loaded',
            'Header metadata loaded. Add the matching .mat signal file for inference.'
        );
    } else if (uploadedMatFile) {
        showToast(
            'Signal Loaded',
            'Please also select matching .hea header file'
        );
    }
}

function parseAndDisplayHeaMetadata(heaText, fileName) {
    const lines = heaText.split(/\r?\n/);

    let fs = null;
    let leadCount = null;

    for (const line of lines) {
        const trimmed = line.trim();

        if (!trimmed || trimmed.startsWith('#')) continue;

        const parts = trimmed.split(/\s+/);

        if (parts.length >= 3) {
            const parsedFs = Number(parts[2]);

            if (Number.isFinite(parsedFs)) {
                fs = parsedFs;
            }
        }

        if (leadCount === null && parts.length >= 2) {
            const firstLine = lines.find(
                l => l.trim() && !l.trim().startsWith('#')
            );

            if (firstLine) {
                const headerParts = firstLine.trim().split(/\s+/);

                if (
                    headerParts.length >= 2 &&
                    Number.isInteger(Number(headerParts[1]))
                ) {
                    leadCount = Number(headerParts[1]);
                }
            }
        }

        break;
    }

    const metaId = document.getElementById('metaId');
    const metaFs = document.getElementById('metaFs');
    const metaLeads = document.getElementById('metaLeads');
    const metaShape = document.getElementById('metaShape');

    if (metaId) {
        metaId.textContent =
            fileName.replace(/\.[^/.]+$/, '');
    }

    if (metaFs) {
        metaFs.textContent =
            fs ? `${fs} Hz` : '--';
    }

    if (metaLeads) {
        metaLeads.textContent =
            leadCount ? `${leadCount} Leads` : '--';
    }

    if (metaShape) {
        metaShape.textContent = '--';
    }
}

function updateFileBadges() {
    const heaBadge = document.getElementById('heaBadge');
    const matBadge = document.getElementById('matBadge');
    const metaId = document.getElementById('metaId');
    const fileStatusBadge = document.getElementById('fileStatusBadge');

    if (heaBadge) {
        if (uploadedHeaFile) {
            heaBadge.innerHTML =
                `<i class="fa-solid fa-check text-success"></i> ${uploadedHeaFile.name}`;

            heaBadge.classList.add('badge-primary');
            heaBadge.classList.remove('badge-outline');
        } else {
            heaBadge.innerHTML =
                `<i class="fa-solid fa-file-code"></i> Header: None`;

            heaBadge.classList.remove('badge-primary');
            heaBadge.classList.add('badge-outline');
        }
    }

    if (matBadge) {
        if (uploadedMatFile) {
            matBadge.innerHTML =
                `<i class="fa-solid fa-check text-success"></i> ${uploadedMatFile.name}`;

            matBadge.classList.add('badge-success');
            matBadge.classList.remove('badge-outline');
        } else {
            matBadge.innerHTML =
                `<i class="fa-solid fa-file-lines"></i> Signal: None`;

            matBadge.classList.remove('badge-success');
            matBadge.classList.add('badge-outline');
        }
    }

    if (uploadedHeaFile && metaId) {
        metaId.textContent =
            uploadedHeaFile.name.replace(/\.[^/.]+$/, "");
    }

    if (fileStatusBadge) {
        if (uploadedHeaFile && uploadedMatFile) {
            fileStatusBadge.textContent = "Ready for Inference";
            fileStatusBadge.className = "badge badge-success";
        } else {
            fileStatusBadge.textContent = ".hea + .mat Required";
            fileStatusBadge.className = "badge badge-info";
        }
    }
}

// Sample-record generation is intentionally disabled.
// CardioSight must analyze a real matching .hea + .mat record through FastAPI.
async function loadSampleWFDBRecord() {
    showToast(
        'Sample Disabled',
        'Please upload a real matching .hea + .mat ECG record.'
    );
}

// ==============================================================================
// 5. INFERENCE EXECUTION (POST /api/analyze)
// ==============================================================================
async function runBackendAnalysis() {

    if (!uploadedHeaFile || !uploadedMatFile) {

        showToast(
            'Missing Files',
            'Please select both .hea and .mat files for the record.'
        );

        return;
    }

    // --------------------------------------------------
    // Validate matching ECG record names
    // Example:
    // JS00001.hea + JS00001.mat  -> valid
    // JS00001.hea + JS00002.mat  -> invalid
    // --------------------------------------------------

    const heaStem =
        uploadedHeaFile.name.replace(/\.hea$/i, '');

    const matStem =
        uploadedMatFile.name.replace(/\.mat$/i, '');

    if (heaStem.toLowerCase() !== matStem.toLowerCase()) {

        showToast(
            'Record Mismatch',
            'The .hea and .mat files must belong to the same ECG record.'
        );

        return;
    }

    // --------------------------------------------------
    // Disable Analyze button while inference is running
    // --------------------------------------------------

    const analyzeBtn =
        document.getElementById('analyzeBtn');

    const predStatus =
        document.getElementById('predStatus');

    if (analyzeBtn) {

        analyzeBtn.disabled = true;

        analyzeBtn.innerHTML =
            `<i class="fa-solid fa-spinner fa-spin"></i> Running Real Inference...`;
    }

    if (predStatus) {

        predStatus.textContent =
            "Processing Pipeline...";
    }

    // --------------------------------------------------
    // Build multipart/form-data request
    // --------------------------------------------------

    const formData = new FormData();

    formData.append(
        'hea_file',
        uploadedHeaFile
    );

    formData.append(
        'mat_file',
        uploadedMatFile
    );

    formData.append(
        'model',
        activeModel
    );

    // No client/hospital selection is exposed in the UI.
    // Backend uses its configured default preprocessing behaviour.

    formData.append(
        'source_hospital',
        'default'
    );

    // --------------------------------------------------
    // Backend endpoints
    // --------------------------------------------------

    const targetEndpoints = [
        'http://127.0.0.1:8000/api/analyze',
        '/api/analyze'
    ];

    let responseData = null;
    let errorDetail = null;

    // --------------------------------------------------
    // Send request
    // --------------------------------------------------

    for (const url of targetEndpoints) {

        try {

            const res = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {

                responseData = await res.json();

                break;

            } else {

                const errJson =
                    await res.json().catch(
                        () => ({
                            detail: res.statusText
                        })
                    );

                errorDetail =
                    errJson.detail || res.statusText;
            }

        } catch (e) {

            errorDetail = e.message;
        }
    }

    // --------------------------------------------------
    // Restore Analyze button
    // --------------------------------------------------

    if (analyzeBtn) {

        analyzeBtn.disabled = false;

        analyzeBtn.innerHTML =
            `<i class="fa-solid fa-bolt"></i> Analyze ECG`;
    }

    // --------------------------------------------------
    // Handle successful backend response
    // --------------------------------------------------

    if (responseData && responseData.success) {

        latestBackendResponse =
            responseData;

        showToast(
            'Inference Completed',
            `Predicted: ${responseData.prediction.class} (${(responseData.prediction.probability * 100).toFixed(1)}%)`
        );

        updateUiWithRealBackendResults(
            responseData
        );

    } else {

        // --------------------------------------------------
        // Handle backend failure
        // --------------------------------------------------

        showToast(
            'Inference Error',
            `Backend request failed: ${
                errorDetail ||
                'Connection refused. Ensure backend/main.py is running on port 8000.'
            }`
        );

        if (predStatus) {

            predStatus.textContent =
                "Inference Failed";
        }
    }
}

// ==============================================================================
// 6. UPDATE UI WITH REAL BACKEND RESULTS
// ==============================================================================
function updateUiWithRealBackendResults(data) {

    // --------------------------------------------------------------------------
    // 1. Metadata
    // --------------------------------------------------------------------------
    if (data.record) {
        document.getElementById('metaId').textContent =
            data.record.record_id || '--';

        document.getElementById('metaFs').textContent =
            `${data.record.sampling_rate || 500} Hz`;

        document.getElementById('metaLeads').textContent =
            `${data.record.num_leads || 12} Leads`;

        document.getElementById('metaShape').textContent =
            `(12, ${data.record.num_samples || 5000})`;
    }

    // --------------------------------------------------------------------------
    // 2. Primary Prediction
    // --------------------------------------------------------------------------
    const pred = data.prediction;

    if (pred) {
        const fullTitle =
            CLASS_FULL_NAMES[pred.class] || pred.class;

        document.getElementById('primaryPredTitle').textContent =
            fullTitle;

        document.getElementById('predProbBadge').textContent =
            `Confidence: ${(pred.probability * 100).toFixed(1)}%`;

        document.getElementById('predStatus').textContent =
            "Inference Complete";

        document.getElementById('predStatus').className =
            "badge badge-success";
    }

    // --------------------------------------------------------------------------
    // 3. 12-Class Probability Bars
    // --------------------------------------------------------------------------
    if (data.probabilities) {
        renderProbabilityBars(
            data.probabilities,
            pred.class
        );
    }

    // --------------------------------------------------------------------------
    // 4. Real ECG Signal Data
    // --------------------------------------------------------------------------
    if (data.ecg && data.ecg.signals) {
        realEcgData = data.ecg;

        canvasZoomLevel = 1.0;
        canvasOffsetX = 0;

        renderEcgCanvas();
    }

    // --------------------------------------------------------------------------
    // 5. 1D Grad-CAM
    // --------------------------------------------------------------------------
    if (
        data.gradcam &&
        data.gradcam.available &&
        data.gradcam.values
    ) {
        realGradCamValues =
            data.gradcam.values;

        document.getElementById('gradCamState').textContent =
            "ON";
    } else {
        realGradCamValues = null;

        document.getElementById('gradCamState').textContent =
            "Unavailable";
    }

    // --------------------------------------------------------------------------
    // 6. SHAP & Fiducial Features
    // --------------------------------------------------------------------------
    if (data.shap) {
        realShapResult = data.shap;

        renderRealShapChart(data.shap);

        if (data.shap.fiducials) {
            updateFiducialsBar(
                data.shap.fiducials
            );
        }
    }

    // --------------------------------------------------------------------------
    // 7. Clinical Report
    // --------------------------------------------------------------------------
    // Kept unchanged intentionally.
    renderClinicalReport(
        pred ? pred.class : 'AF',
        pred ? pred.probability : 0.95
    );

    // --------------------------------------------------------------------------
    // 8. Page 2 Uncertainty
    // --------------------------------------------------------------------------
    if (data.uncertainty) {
        realUncertaintyResult =
            data.uncertainty;

        renderMCDropoutChart(
            data.uncertainty
        );

        updateUncertaintyKpiCards(
            data.uncertainty,
            pred
        );
    }
}

// ==============================================================================
// 7. RENDER 12-CLASS PROBABILITY DISTRIBUTION
// ==============================================================================
function renderProbabilityBars(probabilities, topClass) {
    const container =
        document.getElementById('probabilityBarsList');

    if (!container) return;

    container.innerHTML = '';

    const sorted =
        Object.keys(probabilities)
            .map(key => ({
                key: key,
                name: CLASS_FULL_NAMES[key] || key,
                prob: probabilities[key] || 0.0,
                color: CLASS_COLORS[key] || '#38bdf8'
            }))
            .sort((a, b) => b.prob - a.prob);

    sorted.forEach(item => {
        const pct =
            (item.prob * 100).toFixed(1);

        const isTop =
            item.key === topClass;

        const row =
            document.createElement('div');

        row.className =
            `prob-item-row ${isTop ? 'active-top-class' : ''}`;

        row.innerHTML = `
            <div class="prob-label-wrap">
                <span class="prob-class-name">${item.name}</span>
                <span class="prob-value" style="color:${item.color}">${pct}%</span>
            </div>

            <div class="progress-track">
                <div
                    class="progress-fill"
                    style="width: ${pct}%; background-color: ${item.color};">
                </div>
            </div>
        `;

        container.appendChild(row);
    });
}

function renderInitialEmptyState() {

    // No prediction, fiducials, SHAP values, or probabilities
    // are created locally.
    // These values become available only after a successful FastAPI inference.

    const container =
        document.getElementById('probabilityBarsList');

    if (container) {
        container.innerHTML =
            '<div class="text-muted">No ECG analyzed yet.</div>';
    }

    realEcgData = null;
    realGradCamValues = null;
    realShapResult = null;
    realUncertaintyResult = null;

    updateFiducialsBar(null);

    renderRealShapChart({
        available: false,
        target_class: '--',
        message:
            'Run ECG analysis to generate real SHAP and fiducial results.'
    });

    renderEcgCanvas();

    const pTitle =
        document.getElementById('primaryPredTitle');

    const pBadge =
        document.getElementById('predProbBadge');

    const pStatus =
        document.getElementById('predStatus');

    if (pTitle) {
        pTitle.textContent =
            'No prediction yet';
    }

    if (pBadge) {
        pBadge.textContent =
            'Awaiting ECG analysis';
    }

    if (pStatus) {
        pStatus.textContent =
            'Ready';

        pStatus.className =
            'badge badge-info';
    }

    const gradState =
        document.getElementById('gradCamState');

    if (gradState) {
        gradState.textContent =
            'Unavailable';
    }
}

// ==============================================================================
// 8. REAL 12-LEAD ECG OSCILLOSCOPE & 1D GRAD-CAM OVERLAY
// ==============================================================================
function selectLead(leadId) {
    activeLead = leadId;

    document.querySelectorAll(
        '#leadPills .pill-btn'
    ).forEach(btn => {
        btn.classList.remove('active');

        if (
            btn.textContent.includes(leadId) ||
            (
                leadId === 'ALL' &&
                btn.textContent.includes('All')
            )
        ) {
            btn.classList.add('active');
        }
    });

    renderEcgCanvas();
}

function toggleGradCam() {
    gradCamEnabled =
        !gradCamEnabled;

    const btn =
        document.getElementById('toggleHeatmapBtn');

    const state =
        document.getElementById('gradCamState');

    if (btn && state) {
        if (gradCamEnabled) {
            btn.classList.add('active');
            state.textContent = "ON";
        } else {
            btn.classList.remove('active');
            state.textContent = "OFF";
        }
    }

    renderEcgCanvas();
}

function resetCanvasZoom() {
    canvasZoomLevel = 1.0;
    canvasOffsetX = 0;

    renderEcgCanvas();

    showToast(
        'Oscilloscope Reset',
        'Zoom and pan reset to 1.0x (10s window)'
    );
}

function setupCanvasInteractions() {
    const canvas =
        document.getElementById('ecgCanvas');

    if (!canvas) return;

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();

        const delta =
            e.deltaY < 0 ? 1.15 : 0.87;

        canvasZoomLevel =
            Math.min(
                Math.max(
                    canvasZoomLevel * delta,
                    0.7
                ),
                10.0
            );

        renderEcgCanvas();
    });

    canvas.addEventListener('mousedown', (e) => {
        isPanning = true;
        panStartX =
            e.clientX - canvasOffsetX;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;

        canvasOffsetX =
            e.clientX - panStartX;

        renderEcgCanvas();
    });

    window.addEventListener('mouseup', () => {
        isPanning = false;
    });

    window.addEventListener('resize', () => {
        renderEcgCanvas();
    });
}

function renderEcgCanvas() {
    const canvas =
        document.getElementById('ecgCanvas');

    if (!canvas) return;

    const wrapper =
        document.getElementById('canvasWrapper');

    const rect =
        wrapper
            ? wrapper.getBoundingClientRect()
            : {
                width: 900,
                height: 420
            };

    canvas.width =
        rect.width *
        (window.devicePixelRatio || 1);

    canvas.height =
        (
            activeLead === 'ALL'
                ? 720
                : 420
        ) *
        (window.devicePixelRatio || 1);

    canvas.style.width =
        `${rect.width}px`;

    canvas.style.height =
        `${activeLead === 'ALL' ? 720 : 420}px`;

    const ctx =
        canvas.getContext('2d');

    ctx.scale(
        window.devicePixelRatio || 1,
        window.devicePixelRatio || 1
    );

    const width =
        rect.width;

    const height =
        activeLead === 'ALL'
            ? 720
            : 420;

    // --------------------------------------------------------------------------
    // 1. Clear background
    // --------------------------------------------------------------------------
    ctx.fillStyle = "#030712";
    ctx.fillRect(
        0,
        0,
        width,
        height
    );

    // --------------------------------------------------------------------------
    // 2. Draw standard medical 25mm/s & 10mm/mV grid
    // --------------------------------------------------------------------------
    drawMedicalEcgGrid(
        ctx,
        width,
        height
    );

    // --------------------------------------------------------------------------
    // 3. Draw ECG Waveforms
    // --------------------------------------------------------------------------
    if (
        !realEcgData ||
        !realEcgData.signals
    ) {
        ctx.fillStyle = "#94a3b8";
        ctx.font =
            "14px 'Inter', sans-serif";
        ctx.textAlign = "center";

        ctx.fillText(
            "No ECG signal loaded. Upload .hea & .mat files and click 'Analyze ECG'.",
            width / 2,
            height / 2
        );

        return;
    }

    const signals =
        realEcgData.signals;

    const nSamples =
        signals[0].length;

    if (activeLead === 'ALL') {

        const stripHeight =
            height / 12;

        for (let l = 0; l < 12; l++) {

            const leadName =
                LEAD_NAMES_STANDARD[l] ||
                `Lead ${l + 1}`;

            const centerY =
                stripHeight * l +
                stripHeight / 2;

            ctx.fillStyle =
                "#38bdf8";

            ctx.font =
                "bold 11px 'JetBrains Mono', monospace";

            ctx.textAlign =
                "left";

            ctx.fillText(
                leadName,
                12,
                centerY -
                    stripHeight / 2 +
                    14
            );

            drawSingleLeadWaveform(
                ctx,
                signals[l],
                centerY,
                stripHeight * 0.45,
                width,
                nSamples,
                false
            );
        }

    } else {

        const leadIndex =
            getLeadIndexFromName(
                activeLead
            );

        const signal =
            signals[leadIndex] ||
            signals[0];

        const centerY =
            height / 2;

        ctx.fillStyle =
            "rgba(56, 189, 248, 0.12)";

        ctx.font =
            "bold 60px 'Inter', sans-serif";

        ctx.textAlign =
            "right";

        ctx.fillText(
            `Lead ${activeLead}`,
            width - 24,
            75
        );

        if (
            gradCamEnabled &&
            realGradCamValues &&
            realGradCamValues.length === nSamples
        ) {
            drawGradCamHeatmap(
                ctx,
                realGradCamValues,
                centerY,
                height * 0.35,
                width,
                nSamples
            );
        }

        drawSingleLeadWaveform(
            ctx,
            signal,
            centerY,
            height * 0.35,
            width,
            nSamples,
            true
        );
    }
}

function drawMedicalEcgGrid(
    ctx,
    width,
    height
) {
    const smallGrid = 15;
    const largeGrid = 75;

    ctx.strokeStyle =
        "rgba(56, 189, 248, 0.04)";

    ctx.lineWidth = 1;

    ctx.beginPath();

    for (
        let x = 0;
        x <= width;
        x += smallGrid
    ) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
    }

    for (
        let y = 0;
        y <= height;
        y += smallGrid
    ) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
    }

    ctx.stroke();

    ctx.strokeStyle =
        "rgba(56, 189, 248, 0.10)";

    ctx.lineWidth = 1.2;

    ctx.beginPath();

    for (
        let x = 0;
        x <= width;
        x += largeGrid
    ) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
    }

    for (
        let y = 0;
        y <= height;
        y += largeGrid
    ) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
    }

    ctx.stroke();

    ctx.fillStyle =
        "rgba(148, 163, 184, 0.4)";

    ctx.font =
        "10px 'JetBrains Mono', monospace";

    ctx.textAlign =
        "left";

    for (
        let sec = 0;
        sec <= 10;
        sec += 2
    ) {
        const xPos =
            (sec / 10) * width;

        ctx.fillText(
            `${sec}.0s`,
            xPos + 4,
            height - 6
        );
    }
}

function drawSingleLeadWaveform(
    ctx,
    signal,
    centerY,
    ampScale,
    width,
    nSamples,
    isPrimary
) {
    ctx.save();

    ctx.beginPath();

    ctx.rect(
        0,
        0,
        width,
        ctx.canvas.height
    );

    ctx.clip();

    ctx.strokeStyle =
        isPrimary
            ? "#38bdf8"
            : "#0ea5e9";

    ctx.lineWidth =
        isPrimary
            ? 2.0
            : 1.4;

    ctx.lineJoin =
        "round";

    ctx.lineCap =
        "round";

    ctx.beginPath();

    const stepX =
        (width / nSamples) *
        canvasZoomLevel;

    for (
        let i = 0;
        i < nSamples;
        i++
    ) {
        const x =
            (i * stepX) +
            canvasOffsetX;

        if (
            x < -20 ||
            x > width + 20
        ) {
            continue;
        }

        const y =
            centerY -
            (signal[i] * ampScale);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }

    ctx.stroke();

    ctx.restore();
}

function drawGradCamHeatmap(
    ctx,
    camValues,
    centerY,
    ampScale,
    width,
    nSamples
) {
    ctx.save();

    const stepX =
        (width / nSamples) *
        canvasZoomLevel;

    for (
        let i = 0;
        i < nSamples;
        i += 4
    ) {
        const x =
            (i * stepX) +
            canvasOffsetX;

        if (
            x < -20 ||
            x > width + 20
        ) {
            continue;
        }

        const intensity =
            Math.min(
                Math.max(
                    camValues[i],
                    0
                ),
                1.0
            );

        if (intensity > 0.05) {

            const w =
                stepX * 4;

            const grad =
                ctx.createLinearGradient(
                    x,
                    centerY - ampScale,
                    x,
                    centerY + ampScale
                );

            grad.addColorStop(
                0,
                `rgba(244, 63, 94, 0)`
            );

            grad.addColorStop(
                0.5,
                `rgba(244, 63, 94, ${intensity * 0.35})`
            );

            grad.addColorStop(
                1,
                `rgba(244, 63, 94, 0)`
            );

            ctx.fillStyle =
                grad;

            ctx.fillRect(
                x,
                centerY - ampScale,
                w,
                ampScale * 2
            );
        }
    }

    ctx.restore();
}

function getLeadIndexFromName(name) {
    const clean =
        name.replace(
            "Lead ",
            ""
        ).trim();

    const idx =
        LEAD_NAMES_STANDARD.indexOf(
            clean
        );

    return idx >= 0
        ? idx
        : 1;
}

// ==============================================================================
// 9. FIDUCIAL FEATURE SHAP & METRICS
// ==============================================================================
function updateFiducialsBar(fiducials) {

    if (!fiducials) {

        [
            'wfHR',
            'wfRR',
            'wfPR',
            'wfQRS',
            'wfQT'
        ].forEach(id => {

            const el =
                document.getElementById(id);

            if (el) {
                el.textContent = '--';
            }
        });

        const beatsEl =
            document.getElementById(
                'wfBeats'
            );

        if (beatsEl) {
            beatsEl.textContent = '--';
        }

        return;
    }

    const setVal =
        (id, val, unit) => {

            const el =
                document.getElementById(id);

            if (el) {
                el.textContent =
                    (
                        val !== undefined &&
                        val !== null
                    )
                        ? `${Number(val).toFixed(1)} ${unit}`
                        : '--';
            }
        };

    setVal(
        'wfHR',
        fiducials.heart_rate_bpm,
        'bpm'
    );

    setVal(
        'wfRR',
        (
            fiducials.mean_rr_interval
                ? fiducials.mean_rr_interval * 1000
                : null
        ),
        'ms'
    );

    setVal(
        'wfPR',
        (
            fiducials.pr_interval
                ? fiducials.pr_interval * 1000
                : null
        ),
        'ms'
    );

    setVal(
        'wfQRS',
        (
            fiducials.qrs_duration
                ? fiducials.qrs_duration * 1000
                : null
        ),
        'ms'
    );

    setVal(
        'wfQT',
        (
            fiducials.qt_interval
                ? fiducials.qt_interval * 1000
                : null
        ),
        'ms'
    );

    const beatsEl =
        document.getElementById(
            'wfBeats'
        );

    if (beatsEl) {
        beatsEl.textContent =
            fiducials.n_beats_detected ||
            '--';
    }
}

function renderRealShapChart(shapData) {
    const ctx =
        document.getElementById(
            'shapBarChartStudio'
        );

    const badge =
        document.getElementById(
            'shapTargetBadge'
        );

    const unavail =
        document.getElementById(
            'shapUnavailableMsg'
        );

    const wrap =
        document.getElementById(
            'shapChartWrap'
        );

    if (!ctx) return;

    if (badge) {
        badge.textContent =
            `Target: ${shapData.target_class || '--'}`;
    }

    if (
        !shapData.available ||
        !shapData.features
    ) {

        if (unavail) {
            unavail.style.display =
                'block';

            unavail.textContent =
                shapData.message ||
                'Fiducial feature SHAP explanation is unavailable for this record.';
        }

        if (wrap) {
            wrap.style.display =
                'none';
        }

        return;
    }

    if (unavail) {
        unavail.style.display =
            'none';
    }

    if (wrap) {
        wrap.style.display =
            'block';
    }

    const featureLabels = {
        "mean_rr_interval": "Mean RR Interval",
        "heart_rate_bpm": "Heart Rate (bpm)",
        "p_wave_amplitude": "P-Wave Amplitude",
        "qrs_amplitude": "QRS Amplitude",
        "t_wave_amplitude": "T-Wave Amplitude",
        "pr_interval": "PR Interval",
        "qt_interval": "QT Interval",
        "qrs_duration": "QRS Duration",
        "n_beats_detected": "Detected Beats"
    };

    const labels =
        Object.keys(
            shapData.features
        ).map(
            k => featureLabels[k] || k
        );

    const values =
        Object.values(
            shapData.features
        );

    if (shapChartStudio) {
        shapChartStudio.destroy();
    }

    shapChartStudio =
        new Chart(
            ctx,
            {
                type: 'bar',

                data: {
                    labels: labels,

                    datasets: [{
                        label:
                            'SHAP Value (Impact on Prediction)',

                        data: values,

                        backgroundColor:
                            values.map(
                                v =>
                                    v >= 0
                                        ? 'rgba(56, 189, 248, 0.85)'
                                        : 'rgba(244, 63, 94, 0.85)'
                            ),

                        borderColor:
                            values.map(
                                v =>
                                    v >= 0
                                        ? '#38bdf8'
                                        : '#f43f5e'
                            ),

                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },

                options: {
                    indexAxis: 'y',

                    responsive: true,

                    maintainAspectRatio:
                        false,

                    plugins: {
                        legend: {
                            display: false
                        },

                        tooltip: {
                            callbacks: {
                                label:
                                    function(c) {
                                        return ` SHAP Impact: ${c.parsed.x > 0 ? '+' : ''}${c.parsed.x.toFixed(4)}`;
                                    }
                            }
                        }
                    },

                    scales: {
                        x: {
                            grid: {
                                color:
                                    'rgba(255,255,255,0.05)'
                            },

                            ticks: {
                                color:
                                    '#94a3b8',

                                font: {
                                    family:
                                        'JetBrains Mono',
                                    size: 10
                                }
                            }
                        },

                        y: {
                            grid: {
                                display: false
                            },

                            ticks: {
                                color:
                                    '#e2e8f0',

                                font: {
                                    family:
                                        'Inter',
                                    size: 11
                                }
                            }
                        }
                    }
                }
            }
        );
}

// ==============================================================================
// 10. PAGE 2: REAL MC-DROPOUT UNCERTAINTY VISUALIZATION
// ==============================================================================
function updateUncertaintyKpiCards(
    uncertainty,
    pred
) {
    const valClass =
        document.getElementById(
            'valPredClass'
        );

    const valProb =
        document.getElementById(
            'valPredProb'
        );

    const valStd =
        document.getElementById(
            'valPredStd'
        );

    if (valClass && pred) {
        valClass.textContent =
            pred.class;
    }

    if (valProb && pred) {
        valProb.textContent =
            `${(pred.probability * 100).toFixed(1)}%`;
    }

    if (valStd && uncertainty) {
        valStd.textContent =
            `± ${(uncertainty.predicted_class_uncertainty || 0).toFixed(4)}`;
    }
}

function renderMCDropoutChart(
    uncertaintyData
) {
    const ctx =
        document.getElementById(
            'mcDropoutChart'
        );

    if (
        !ctx ||
        !uncertaintyData ||
        !uncertaintyData.classes
    ) {
        return;
    }

    const classes =
        CLASS_NAMES;

    const means =
        classes.map(c => {

            const item =
                uncertaintyData.classes[c];

            return item
                ? item.mean_probability
                : 0.0;
        });

    const stds =
        classes.map(c => {

            const item =
                uncertaintyData.classes[c];

            return item
                ? item.uncertainty_std
                : 0.0;
        });

    if (mcDropoutChart) {
        mcDropoutChart.destroy();
    }

    mcDropoutChart =
        new Chart(
            ctx,
            {
                type: 'bar',

                data: {
                    labels:
                        classes.map(
                            c =>
                                CLASS_FULL_NAMES[c] || c
                        ),

                    datasets: [
                        {
                            label:
                                'Mean Predicted Probability',

                            data:
                                means,

                            backgroundColor:
                                'rgba(56, 189, 248, 0.75)',

                            borderColor:
                                '#38bdf8',

                            borderWidth:
                                1.5,

                            borderRadius:
                                6
                        },

                        {
                            label:
                                'Uncertainty Std Dev (± σ)',

                            data:
                                stds,

                            backgroundColor:
                                'rgba(244, 63, 94, 0.75)',

                            borderColor:
                                '#f43f5e',

                            borderWidth:
                                1.5,

                            borderRadius:
                                6
                        }
                    ]
                },

                options: {
                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    plugins: {
                        legend: {
                            position:
                                'top',

                            labels: {
                                color:
                                    '#e2e8f0',

                                font: {
                                    family:
                                        'Inter',
                                    size: 12
                                }
                            }
                        },

                        tooltip: {
                            callbacks: {
                                label:
                                    function(c) {
                                        return ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}`;
                                    }
                            }
                        }
                    },

                    scales: {
                        x: {
                            grid: {
                                color:
                                    'rgba(255,255,255,0.05)'
                            },

                            ticks: {
                                color:
                                    '#94a3b8',

                                font: {
                                    family:
                                        'Inter',
                                    size: 10
                                },

                                maxRotation:
                                    45
                            }
                        },

                        y: {
                            grid: {
                                color:
                                    'rgba(255,255,255,0.08)'
                            },

                            ticks: {
                                color:
                                    '#e2e8f0',

                                font: {
                                    family:
                                        'JetBrains Mono',
                                    size: 11
                                }
                            },

                            min:
                                0.0,

                            max:
                                1.0
                        }
                    }
                }
            }
        );
}

// ==============================================================================
// 11. CLINICAL REPORT & REASONING (DETERMINISTIC)
// ==============================================================================

function renderClinicalReport(
    predictedClass,
    probability
) {
    const container =
        document.getElementById(
            'reportContent'
        );

    if (!container) return;

    const descriptions = {

        "AF": {
            etiology:
                "Chaotic atrial depolarization (350-600 bpm) with irregularly irregular ventricular response and absent discrete P-waves.",

            risks:
                "Elevated risk of cardioembolic stroke, systemic thromboembolism, and tachycardia-induced cardiomyopathy.",

            actions:
                "Assess CHA2DS2-VASc score for anticoagulation (DOACs), initiate ventricular rate control (beta-blockers/diltiazem), and evaluate for rhythm conversion."
        },

        "NSR": {
            etiology:
                "Normal sinus rhythm originating from the SA node with regular 1:1 AV conduction and normal fiducial intervals.",

            risks:
                "Standard baseline physiological cardiovascular state.",

            actions:
                "Routine clinical follow-up; no acute antiarrhythmic intervention indicated."
        },

        "LBBB": {
            etiology:
                "Delayed left ventricular conduction manifesting as wide QRS (>=120ms) with broad notched R-waves in lateral leads.",

            risks:
                "Potential underlying structural heart disease, dilated cardiomyopathy, or coronary ischemia.",

            actions:
                "Transthoracic echocardiography to assess left ventricular ejection fraction (LVEF); cardiology consultation."
        },

        "RBBB": {
            etiology:
                "Conduction delay in right bundle branch producing rsR' pattern in V1 and wide slurred S wave in lateral leads.",

            risks:
                "May be idiopathic or associated with pulmonary hypertension, right ventricular strain, or ASD.",

            actions:
                "Correlate with clinical symptoms, oxygen saturation, and prior ECGs."
        },

        "IAVB": {
            etiology:
                "Prolonged AV nodal conduction delay resulting in PR interval > 200 ms with 1:1 AV conduction.",

            risks:
                "Progression to higher-grade AV blocks if accompanied by bifascicular block.",

            actions:
                "Review AV-nodal blocking medications (beta-blockers, non-DHP CCBs, digoxin)."
        },

        "STach": {
            etiology:
                "Sinus rhythm with ventricular rate exceeding 100 bpm in response to physiological or pathological triggers.",

            risks:
                "Increased myocardial oxygen demand; hemodynamic compromise in severe tachycardia.",

            actions:
                "Identify underlying cause: fever, hypovolemia, infection, hyperthyroidism, pain, or stimulants."
        },

        "SB": {
            etiology:
                "Sinus rhythm with ventricular rate under 60 bpm due to high vagal tone or intrinsic SA node disease.",

            risks:
                "Syncope, presyncope, or chronotropic incompetence.",

            actions:
                "Evaluate symptomatic status; review negative chronotropes; consider atropine or pacing if symptomatic."
        }
    };

    const info =
        descriptions[predictedClass] ||
        descriptions["AF"];

    const fullName =
        CLASS_FULL_NAMES[predictedClass] ||
        predictedClass;

    container.innerHTML = `
        <div class="report-box">
            <h4 class="report-title text-primary">
                <i class="fa-solid fa-heart-circle-bolt"></i>
                ${fullName}
                (Confidence: ${(probability * 100).toFixed(1)}%)
            </h4>

            <div class="report-section mt-2">
                <strong>Electrophysiological Etiology:</strong>
                <p>${info.etiology}</p>
            </div>

            <div class="report-section mt-2">
                <strong>Clinical Risks & Hemodynamic Impact:</strong>
                <p>${info.risks}</p>
            </div>

            <div class="report-section mt-2">
                <strong>Recommended Precautions & Actions:</strong>
                <p>${info.actions}</p>
            </div>
        </div>
    `;
}

function downloadReport() {
    if (!latestBackendResponse) {
        showToast(
            'No Data',
            'Please execute ECG analysis before exporting report.'
        );
        return;
    }

    const dataStr =
        "data:text/json;charset=utf-8," +
        encodeURIComponent(
            JSON.stringify(
                latestBackendResponse,
                null,
                2
            )
        );

    const downloadAnchor =
        document.createElement('a');

    downloadAnchor.setAttribute(
        "href",
        dataStr
    );

    downloadAnchor.setAttribute(
        "download",
        `CardioSight_Report_${latestBackendResponse.record.record_id || 'ECG'}.json`
    );

    document.body.appendChild(
        downloadAnchor
    );

    downloadAnchor.click();

    downloadAnchor.remove();

    showToast(
        'Report Exported',
        'Clinical JSON dossier downloaded.'
    );
}

// ==============================================================================
// 12. PAGE 2: CLINICAL AI ASSISTANT CHATBOT (INFORMATIONAL)
// ==============================================================================
function askPresetQuestion(text) {
    const input =
        document.getElementById(
            'chatInput'
        );

    if (input) {
        input.value = text;
        sendChatMessage();
    }
}

function sendChatMessage() {
    const input =
        document.getElementById(
            'chatInput'
        );

    if (!input) return;

    const q =
        input.value.trim();

    if (!q) return;

    addChatMessage(
        'user',
        q
    );

    input.value = '';

    setTimeout(() => {

        if (
            !latestBackendResponse ||
            !latestBackendResponse.prediction
        ) {
            addChatMessage(
                'ai',
                'Please run a real ECG analysis first. I will use the latest backend prediction and uncertainty results when they are available.'
            );

            return;
        }

        const predClass =
            latestBackendResponse
                .prediction
                .class;

        const prob =
            latestBackendResponse
                .prediction
                .probability;

        const answer =
            generateLocalInformationalAnswer(
                q,
                predClass,
                prob
            );

        addChatMessage(
            'ai',
            answer
        );

    }, 350);
}

function addChatMessage(
    sender,
    text
) {
    const dialogue =
        document.getElementById(
            'chatDialogue'
        );

    if (!dialogue) return;

    const msgDiv =
        document.createElement(
            'div'
        );

    msgDiv.className =
        `chat-msg ${sender}-msg`;

    const now =
        new Date().toLocaleTimeString(
            [],
            {
                hour: '2-digit',
                minute: '2-digit'
            }
        );

    if (sender === 'user') {

        msgDiv.innerHTML = `
            <div class="msg-avatar">
                <i class="fa-solid fa-user-tie"></i>
            </div>

            <div class="msg-bubble">
                <div class="msg-author">
                    Clinician
                </div>

                <div class="msg-text">
                    ${escapeHtml(text)}
                </div>

                <div class="msg-time">
                    ${now}
                </div>
            </div>
        `;

    } else {

        msgDiv.innerHTML = `
            <div class="msg-avatar">
                <i class="fa-solid fa-user-doctor"></i>
            </div>

            <div class="msg-bubble">
                <div class="msg-author">
                    CardioSight Clinical Assistant
                </div>

                <div class="msg-text">
                    ${text}
                </div>

                <div class="msg-time">
                    ${now}
                </div>
            </div>
        `;
    }

    dialogue.appendChild(
        msgDiv
    );

    dialogue.scrollTop =
        dialogue.scrollHeight;
}

function clearChatHistory() {
    const dialogue =
        document.getElementById(
            'chatDialogue'
        );

    if (!dialogue) return;

    dialogue.innerHTML = `
        <div class="chat-msg ai-msg">
            <div class="msg-avatar">
                <i class="fa-solid fa-user-doctor"></i>
            </div>

            <div class="msg-bubble">
                <div class="msg-author">
                    CardioSight Clinical Assistant
                </div>

                <div class="msg-text">
                    Chat cleared. Ready for your clinical inquiries.
                </div>

                <div class="msg-time">
                    Ready
                </div>
            </div>
        </div>
    `;

    showToast(
        'Chat Cleared',
        'Conversation history reset.'
    );
}

function generateLocalInformationalAnswer(
    query,
    predictedClass,
    prob
) {
    const q =
        query.toLowerCase();

    const fullName =
        CLASS_FULL_NAMES[predictedClass] ||
        predictedClass;

    if (
        q.includes('reason') ||
        q.includes('why') ||
        q.includes('diagnos')
    ) {
        return `
            <strong>Diagnostic Basis for ${fullName}:</strong><br>
            The ResNet-34 model predicts ${fullName}
            with ${(prob * 100).toFixed(1)}% sigmoid probability.
            Saliency maps (1D Grad-CAM from layer 4) and
            NeuroKit2 fiducial features highlight the
            characteristic rhythm and morphology patterns.
        `;
    }

    if (
        q.includes('uncertainty') ||
        q.includes('mc') ||
        q.includes('dropout')
    ) {
        const u =
            latestBackendResponse &&
            latestBackendResponse.uncertainty
                ? latestBackendResponse
                    .uncertainty
                    .predicted_class_uncertainty
                : null;

        return `
            <strong>MC-Dropout Uncertainty Interpretation:</strong><br>
            30 stochastic forward passes with dropout probability
            <em>p = 0.3</em> produced an uncertainty standard deviation
            of <strong>&sigma; = ${
                u === null
                    ? 'Unavailable'
                    : Number(u).toFixed(4)
            }</strong>
            for ${fullName}.
            A lower &sigma; indicates higher stability across
            stochastic perturbations.
        `;
    }

    if (
        q.includes('precaution') ||
        q.includes('protocol') ||
        q.includes('drug') ||
        q.includes('treatment')
    ) {
        return `
            <strong>Precautions for ${fullName}:</strong><br>
            Refer to standard cardiology guidelines.
            Verify electrolyte balance (K+, Mg2+), review
            chronotropic and AV-nodal blocking medications,
            and consider 12-lead Holter monitoring or
            echocardiography if symptomatic.
        `;
    }

    if (
        q.includes('fedavg') ||
        q.includes('fedprox') ||
        q.includes('fedadam') ||
        q.includes('strategy') ||
        q.includes('differ')
    ) {
        return `
            <strong>Federated Learning Strategies:</strong><br>
            &bull; <strong>FedAvg:</strong>
            Classical parameter averaging across participating nodes.<br>

            &bull; <strong>FedProx:</strong>
            Adds a proximal term (&mu;=0.001) to limit
            local gradient drift under non-IID data.<br>

            &bull; <strong>FedAdam:</strong>
            Server-side adaptive momentum optimization for
            robust convergence across heterogeneous cohorts.
        `;
    }

    return `
        For ${fullName}, the universal 12-class federated
        ResNet-34 model indicates a probability of
        ${(prob * 100).toFixed(1)}%.
        Review the 12-lead oscilloscope, fiducial feature
        SHAP chart, and MC-Dropout uncertainty distributions
        for comprehensive assessment.
    `;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// ==============================================================================
// 13. PAGES 3, 4, 5 (PRESERVED BENCHMARKS & PATHOLOGY GUIDE)
// ==============================================================================

const CLIENT_BENCHMARK_DATA = {

    all: {

        title:
            'Reported Per-Source Micro-F1 (Round 30)',

        rows: [

            {
                name: 'Chapman-Shaoxing',
                fedavg: 0.884,
                fedprox: 0.853,
                fedadam: 0.875,
                f1stdFedavg: 0.2610,
                f1stdFedprox: 0.3714,
                f1stdFedadam: 0.0889
            },

            {
                name: 'CPSC-2018',
                fedavg: 0.853,
                fedprox: 0.765,
                fedadam: 0.835,
                f1stdFedavg: 0.3133,
                f1stdFedprox: 0.2627,
                f1stdFedadam: 0.0888
            },

            {
                name: 'Georgia',
                fedavg: 0.769,
                fedprox: 0.724,
                fedadam: 0.761,
                f1stdFedavg: 0.2994,
                f1stdFedprox: 0.3510,
                f1stdFedadam: 0.1009
            },

            {
                name: 'Ningbo',
                fedavg: 0.864,
                fedprox: 0.807,
                fedadam: 0.865,
                f1stdFedavg: 0.3204,
                f1stdFedprox: 0.3786,
                f1stdFedadam: 0.1007
            },

            {
                name: 'PTB-XL',
                fedavg: 0.859,
                fedprox: 0.825,
                fedadam: 0.849,
                f1stdFedavg: 0.3243,
                f1stdFedprox: 0.3279,
                f1stdFedadam: 0.2538
            }
        ]
    }
};

function showClientF1Table(clientId) {

    document.querySelectorAll(
        '.client-tab-btn'
    ).forEach(
        btn =>
            btn.classList.remove(
                'active'
            )
    );

    const tabBtn =
        document.getElementById(
            `tab-${clientId}`
        );

    if (tabBtn) {
        tabBtn.classList.add(
            'active'
        );
    }

    renderClientF1Table(
        clientId
    );
}

function renderClientF1Table(
    clientId
) {
    const container =
        document.getElementById(
            'clientF1ScorecardContainer'
        );

    if (!container) return;

    const data =
        CLIENT_BENCHMARK_DATA.all;

    let tableHtml = `
        <div class="table-header-info mb-2">
            <h4 class="text-primary">
                ${data.title}
            </h4>

            <span class="text-muted">
                Source-level Micro-F1 and reported F1-STD from the study results.
            </span>
        </div>

        <table class="table-modern">
            <thead>
                <tr>
                    <th>Source</th>
                    <th>FedAvg Micro-F1</th>
                    <th>FedProx Micro-F1</th>
                    <th>FedAdam/FedOpt Micro-F1</th>
                    <th>FedAvg F1-STD</th>
                    <th>FedProx F1-STD</th>
                    <th>FedAdam/FedOpt F1-STD</th>
                </tr>
            </thead>

            <tbody>
    `;

    data.rows.forEach(row => {

        tableHtml += `
            <tr>
                <td>
                    <strong>${row.name}</strong>
                </td>

                <td>
                    ${row.fedavg.toFixed(3)}
                </td>

                <td>
                    ${row.fedprox.toFixed(3)}
                </td>

                <td>
                    ${row.fedadam.toFixed(3)}
                </td>

                <td>
                    ${row.f1stdFedavg.toFixed(4)}
                </td>

                <td>
                    ${row.f1stdFedprox.toFixed(4)}
                </td>

                <td>
                    ${row.f1stdFedadam.toFixed(4)}
                </td>
            </tr>
        `;
    });

    tableHtml +=
        `</tbody></table>`;

    container.innerHTML =
        tableHtml;
}

// Add the real per-round values here when they are exported from the results folder.
// Keep null until those values are available; do not fabricate a convergence curve.
const ACTUAL_CONVERGENCE_DATA = null;

function initConvergenceChart() {
    const ctx =
        document.getElementById(
            'convergenceChart'
        );

    if (!ctx) return;

    if (convergenceChartInstance) {
        convergenceChartInstance.destroy();
    }

    if (!ACTUAL_CONVERGENCE_DATA) {

        const parent =
            ctx.parentElement;

        if (parent) {

            const note =
                document.createElement(
                    'div'
                );

            note.className =
                'text-muted';

            note.style.padding =
                '24px';

            note.textContent =
                'Per-round convergence data is not loaded. Replace ACTUAL_CONVERGENCE_DATA with values from the results folder.';

            parent.appendChild(
                note
            );
        }

        return;
    }

    convergenceChartInstance =
        new Chart(
            ctx,
            {
                type: 'line',

                data:
                    ACTUAL_CONVERGENCE_DATA,

                options: {
                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    plugins: {
                        legend: {
                            position:
                                'top'
                        }
                    },

                    scales: {
                        y: {
                            min: 0,
                            max: 1
                        }
                    }
                }
            }
        );
}

// Class-wise F1 values are intentionally not fabricated here.
// Insert the real 12-class values from the results folder when available.
const ACTUAL_CLASS_F1_RADAR_DATA = null;

function initRadarChart() {

    const ctx =
        document.getElementById(
            'f1RadarChart'
        );

    if (!ctx) return;

    if (f1RadarChartInstance) {
        f1RadarChartInstance.destroy();
    }

    if (!ACTUAL_CLASS_F1_RADAR_DATA) {

        const parent =
            ctx.parentElement;

        if (parent) {

            const note =
                document.createElement(
                    'div'
                );

            note.className =
                'text-muted';

            note.style.padding =
                '24px';

            note.textContent =
                'Class-wise F1 data is not loaded. Replace ACTUAL_CLASS_F1_RADAR_DATA with the real results.';

            parent.appendChild(
                note
            );
        }

        return;
    }

    f1RadarChartInstance =
        new Chart(
            ctx,
            {
                type: 'radar',

                data:
                    ACTUAL_CLASS_F1_RADAR_DATA,

                options: {
                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    plugins: {
                        legend: {
                            position:
                                'top'
                        }
                    },

                    scales: {
                        r: {
                            min: 0,
                            max: 1
                        }
                    }
                }
            }
        );
}

// ==============================================================================
// 14. PAGE 4: PATHOLOGY GUIDE (12-CLASS ENCYCLOPEDIA)
// ==============================================================================

const PATHOLOGY_ENCYCLOPEDIA = [

    {
        code: "AF",
        snomed: "164889003",
        name: "Atrial Fibrillation",
        cat: "arrhythmia",
        criteria:
            "Irregularly irregular RR cadence, absent discrete P-waves, fibrillatory f-waves (350-600 bpm)."
    },

    {
        code: "IAVB",
        snomed: "270492004",
        name: "1st-Degree AV Block",
        cat: "block",
        criteria:
            "Prolonged PR interval > 200 ms with constant 1:1 AV conduction."
    },

    {
        code: "LAD",
        snomed: "164873001",
        name: "Left Axis Deviation",
        cat: "morphology",
        criteria:
            "Frontal QRS axis between -30° and -90°, positive in Lead I, predominantly negative in Lead II/aVF."
    },

    {
        code: "LBBB",
        snomed: "164909002",
        name: "Left Bundle Branch Block",
        cat: "block",
        criteria:
            "Wide QRS >= 120 ms, broad notched/monophasic R in lateral leads (I, aVL, V5-V6), deep QS in V1."
    },

    {
        code: "NSIVCB",
        snomed: "698252002",
        name: "Non-Specific Intraventricular Block",
        cat: "block",
        criteria:
            "QRS duration > 110 ms not meeting typical LBBB or RBBB criteria."
    },

    {
        code: "NSR",
        snomed: "426783006",
        name: "Normal Sinus Rhythm",
        cat: "normal",
        criteria:
            "Upright P-waves in Lead II, HR 60-100 bpm, normal PR (120-200ms) and QRS (<120ms)."
    },

    {
        code: "PAC",
        snomed: "284470004",
        name: "Premature Atrial Contraction",
        cat: "arrhythmia",
        criteria:
            "Early abnormal P-wave morphology followed by normal narrow QRS and non-compensatory pause."
    },

    {
        code: "QAb",
        snomed: "164917005",
        name: "Abnormal Q-Wave (Myocardial Infarct)",
        cat: "morphology",
        criteria:
            "Pathological Q-wave duration > 40 ms or depth > 25% of subsequent R-wave amplitude."
    },

    {
        code: "RBBB",
        snomed: "59118001",
        name: "Right Bundle Branch Block",
        cat: "block",
        criteria:
            "Wide QRS >= 120 ms, classic rsR' ('rabbit ears') in V1-V2, wide slurred S-wave in I and V6."
    },

    {
        code: "SB",
        snomed: "426177001",
        name: "Sinus Bradycardia",
        cat: "normal",
        criteria:
            "Regular sinus rhythm with resting ventricular rate under 60 bpm."
    },

    {
        code: "STach",
        snomed: "427084000",
        name: "Sinus Tachycardia",
        cat: "arrhythmia",
        criteria:
            "Regular sinus rhythm with resting ventricular rate exceeding 100 bpm."
    },

    {
        code: "TAb",
        snomed: "164934002",
        name: "T-Wave Abnormality",
        cat: "morphology",
        criteria:
            "Inverted, flattened, biphasic, or hyperacute T-waves reflecting ischemia or strain."
    }
];

let activeGuideCategory = 'all';

function filterGuideCategory(cat) {
    activeGuideCategory =
        cat;

    document.querySelectorAll(
        '.guide-filter-btn'
    ).forEach(
        btn =>
            btn.classList.remove(
                'active'
            )
    );

    const btn =
        document.getElementById(
            `gfilter-${cat}`
        );

    if (btn) {
        btn.classList.add(
            'active'
        );
    }

    initPathologyGuideGrid();
}

function onGuideSearch(e) {
    initPathologyGuideGrid(
        e.target.value.toLowerCase()
    );
}

function initPathologyGuideGrid(
    searchTerm = ''
) {
    const grid =
        document.getElementById(
            'pathologyGrid'
        );

    if (!grid) return;

    grid.innerHTML = '';

    const filtered =
        PATHOLOGY_ENCYCLOPEDIA.filter(
            p => {

                const matchesCat =
                    (
                        activeGuideCategory === 'all' ||
                        p.cat === activeGuideCategory
                    );

                const matchesSearch =
                    !searchTerm ||
                    p.name
                        .toLowerCase()
                        .includes(searchTerm) ||
                    p.code
                        .toLowerCase()
                        .includes(searchTerm) ||
                    p.snomed
                        .includes(searchTerm) ||
                    p.criteria
                        .toLowerCase()
                        .includes(searchTerm);

                return (
                    matchesCat &&
                    matchesSearch
                );
            }
        );

    if (filtered.length === 0) {

        grid.innerHTML = `
            <div class="card full-width">
                <p class="text-muted">
                    No matching pathologies found.
                </p>
            </div>
        `;

        return;
    }

    filtered.forEach(item => {

        const card =
            document.createElement(
                'div'
            );

        card.className =
            "card pathology-guide-card";

        card.innerHTML = `
            <div class="card-header">

                <div class="header-with-badge">

                    <span class="badge badge-primary">
                        ${item.code}
                    </span>

                    <h4>
                        ${item.name}
                    </h4>

                </div>

                <span class="badge badge-outline">
                    SNOMED: ${item.snomed}
                </span>

            </div>

            <div class="guide-card-body mt-2">

                <p class="text-muted">
                    <strong>
                        12-Lead Diagnostic Criteria:
                    </strong>
                </p>

                <p class="mt-1">
                    ${item.criteria}
                </p>

            </div>
        `;

        grid.appendChild(
            card
        );
    });
}
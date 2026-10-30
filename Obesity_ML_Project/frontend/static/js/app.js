/**
 * Obesity Risk and Level Prediction System
 * Frontend Client Application (app.js)
 * Connects UI to Flask backend endpoints via JavaScript Fetch API.
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initBMICalculator();
  initSamplePatientLoader();
  initFormSubmission();
  initDatasetUpload();
  initTrainingTrigger();

  // Load initial data from APIs
  loadDatasetSummary();
  loadPreprocessingReport();
  loadEDAData();
  loadModelMetrics();
});

// ==========================================================================
// 1. Navigation & View Switching
// ==========================================================================
const VIEW_TITLES = {
  dashboard: { title: 'Dashboard Overview', subtitle: 'Project metrics, dataset provenance, and ML workflow' },
  dataset: { title: 'Dataset Explorer', subtitle: 'Upload, audit, inspect attributes, and preview records' },
  preprocessing: { title: 'Data Preprocessing', subtitle: 'Duplicate handling, encoding, scaling, and leakage prevention' },
  eda: { title: 'Exploratory Data Analysis', subtitle: 'Healthcare visualizations, distributions, and behavioral patterns' },
  training: { title: 'Model Training & Tuning', subtitle: 'Stratified cross-validation, hyperparameter tuning, and target leakage' },
  evaluation: { title: 'Model Evaluation', subtitle: 'Multiclass metrics, algorithm benchmark, confusion matrix, and ROC curves' },
  prediction: { title: 'Obesity Level Prediction', subtitle: 'Patient assessment form, automated BMI, probabilities, and explainability' },
  reports: { title: 'Academic Reports & Summary', subtitle: 'Downloadable benchmarks, quality audits, and viva summary' }
};

function initNavigation() {
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
  const pageViews = document.querySelectorAll('.page-view');
  const topTitle = document.getElementById('top-title');
  const topSubtitle = document.getElementById('top-subtitle');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetView = item.getAttribute('data-view');

      // Update active nav button
      navItems.forEach(nav => nav.classList.remove('active'));
      item.classList.add('active');

      // Update visible page view
      pageViews.forEach(view => {
        if (view.id === `view-${targetView}`) {
          view.classList.add('active');
        } else {
          view.classList.remove('active');
        }
      });

      // Update titles
      if (VIEW_TITLES[targetView]) {
        topTitle.textContent = VIEW_TITLES[targetView].title;
        topSubtitle.textContent = VIEW_TITLES[targetView].subtitle;
      }

      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

// ==========================================================================
// 2. Dataset Explorer & Summary API
// ==========================================================================
async function loadDatasetSummary() {
  try {
    const res = await fetch('/api/dataset-summary');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;

    // Dashboard stats
    const dashRecords = document.getElementById('dash-records');
    const dashFeatures = document.getElementById('dash-features');
    const dashClasses = document.getElementById('dash-classes');
    if (dashRecords) dashRecords.textContent = data.total_records.toLocaleString();
    if (dashFeatures) dashFeatures.textContent = data.total_features;
    if (dashClasses) dashClasses.textContent = data.target_classes_count;

    // Explorer stats
    const expRecords = document.getElementById('explorer-records');
    const expCols = document.getElementById('explorer-columns');
    const expMissing = document.getElementById('explorer-missing');
    const expDup = document.getElementById('explorer-duplicates');
    if (expRecords) expRecords.textContent = data.total_records.toLocaleString();
    if (expCols) expCols.textContent = data.total_columns;
    if (expMissing) expMissing.textContent = data.missing_values_count;
    if (expDup) expDup.textContent = data.duplicate_rows_count;

    // Preview table
    const previewTbody = document.getElementById('preview-tbody');
    if (previewTbody && data.preview) {
      previewTbody.innerHTML = data.preview.map((row, idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td>${row.Gender || ''}</td>
          <td>${row.Age || ''}</td>
          <td>${row.Height || ''}</td>
          <td>${row.Weight || ''}</td>
          <td>${row.family_history_with_overweight || ''}</td>
          <td>${row.FAVC || ''}</td>
          <td>${row.FCVC || ''}</td>
          <td>${row.NCP || ''}</td>
          <td>${row.CAEC || ''}</td>
          <td>${row.SMOKE || ''}</td>
          <td>${row.CH2O || ''}</td>
          <td>${row.SCC || ''}</td>
          <td>${row.FAF || ''}</td>
          <td>${row.TUE || ''}</td>
          <td>${row.CALC || ''}</td>
          <td>${row.MTRANS || ''}</td>
          <td><span class="status-badge model-ready">${(row.NObeyesdad || '').replace(/_/g, ' ')}</span></td>
        </tr>
      `).join('');
    }

    // Columns dictionary table
    const columnsTbody = document.getElementById('columns-tbody');
    if (columnsTbody && data.columns) {
      columnsTbody.innerHTML = data.columns.map(col => {
        let rangeStr = '';
        if (col.type === 'Numerical') {
          rangeStr = `Min: ${col.min} | Max: ${col.max} | &mu;: ${col.mean}`;
        } else if (col.unique_values) {
          rangeStr = col.unique_values.slice(0, 5).join(', ') + (col.unique_values.length > 5 ? '...' : '');
        }
        return `
          <tr>
            <td><strong>${col.name}</strong></td>
            <td><span class="status-badge ${col.type === 'Numerical' ? 'online' : 'model-ready'}">${col.type}</span></td>
            <td><code>${col.dtype}</code></td>
            <td>${col.missing}</td>
            <td style="font-size: 12px; color: var(--text-muted);">${rangeStr}</td>
            <td style="font-size: 12.5px;">${col.description}</td>
          </tr>
        `;
      }).join('');
    }
  } catch (err) {
    console.error('Failed to load dataset summary:', err);
  }
}

function initDatasetUpload() {
  const form = document.getElementById('upload-form');
  const input = document.getElementById('csv-file-input');
  const btn = document.getElementById('upload-btn');
  const feedback = document.getElementById('upload-feedback');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!input.files || input.files.length === 0) {
      alert('Please select a CSV file to upload.');
      return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    btn.disabled = true;
    feedback.textContent = 'Uploading and validating CSV...';
    feedback.style.color = 'var(--secondary-blue)';

    try {
      const res = await fetch('/api/upload-dataset', {
        method: 'POST',
        body: formData
      });
      const json = await res.json();

      if (json.success) {
        feedback.textContent = 'Dataset successfully updated and validated!';
        feedback.style.color = 'var(--success-color)';
        // Refresh views
        loadDatasetSummary();
        loadPreprocessingReport();
        loadEDAData();
      } else {
        feedback.textContent = `Error: ${json.error}`;
        feedback.style.color = 'var(--danger-color)';
      }
    } catch (err) {
      feedback.textContent = `Upload failed: ${err.message}`;
      feedback.style.color = 'var(--danger-color)';
    } finally {
      btn.disabled = false;
    }
  });
}

// ==========================================================================
// 3. Preprocessing Report API
// ==========================================================================
async function loadPreprocessingReport() {
  try {
    const res = await fetch('/api/preprocessing-report');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;

    // Populate before/after stats
    const beforeRecords = document.getElementById('prep-before-records');
    const beforeDup = document.getElementById('prep-before-duplicates');
    const afterRecords = document.getElementById('prep-after-records');
    const afterSplit = document.getElementById('prep-after-split');

    if (beforeRecords) beforeRecords.textContent = `${data.before.total_records} instances`;
    if (beforeDup) beforeDup.textContent = `${data.before.duplicate_rows} duplicate rows detected`;
    if (afterRecords) afterRecords.textContent = `${data.after.total_records} records retained (${data.after.duplicates_removed} removed)`;
    if (afterSplit) afterSplit.textContent = `${data.after.train_samples} Train (${data.after.split_ratio.split(':')[0]}%) / ${data.after.test_samples} Test (${data.after.split_ratio.split(':')[1]}%)`;

    // Numerical Range Table
    const rangeTbody = document.getElementById('range-validation-tbody');
    if (rangeTbody && data.before.range_validation && data.before.range_validation.details) {
      const details = data.before.range_validation.details;
      rangeTbody.innerHTML = Object.keys(details).map(feature => {
        const item = details[feature];
        return `
          <tr>
            <td><strong>${feature}</strong></td>
            <td>${item.expected_range[0]} - ${item.expected_range[1]}</td>
            <td>${item.actual_min}</td>
            <td>${item.actual_max}</td>
            <td>${item.out_of_range_count}</td>
            <td><span class="status-badge ${item.out_of_range_count === 0 ? 'online' : ''}">${item.status}</span></td>
          </tr>
        `;
      }).join('');
    }
  } catch (err) {
    console.error('Failed to load preprocessing report:', err);
  }
}

// ==========================================================================
// 4. Exploratory Data Analysis (EDA) API
// ==========================================================================
async function loadEDAData() {
  try {
    const res = await fetch('/api/eda-data');
    const json = await res.json();
    if (!json.success) return;

    const gallery = document.getElementById('eda-gallery');
    if (!gallery || !json.data.charts) return;

    gallery.innerHTML = json.data.charts.map(chart => `
      <div class="chart-card">
        <h4 style="font-size: 14px; font-weight: 700; color: var(--primary-blue); margin-bottom: 10px;">
          ${chart.title}
        </h4>
        <img src="${chart.image_url}" alt="${chart.title}" loading="lazy">
        <div class="chart-interpretation">
          <strong>Clinical Interpretation:</strong> ${chart.interpretation}
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error('Failed to load EDA data:', err);
  }
}

// ==========================================================================
// 5. Model Training Trigger & CV
// ==========================================================================
function initTrainingTrigger() {
  const trainBtn = document.getElementById('start-training-btn');
  const spinner = document.getElementById('training-spinner');
  const msg = document.getElementById('training-msg');

  if (!trainBtn) return;

  trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    spinner.style.display = 'inline-flex';
    msg.textContent = '';

    try {
      const res = await fetch('/api/train-model', { method: 'POST' });
      const json = await res.json();

      if (json.success) {
        msg.textContent = 'Models successfully trained and validated!';
        msg.style.color = 'var(--success-color)';
        // Refresh metrics
        loadModelMetrics();
      } else {
        msg.textContent = `Training error: ${json.error}`;
        msg.style.color = 'var(--danger-color)';
      }
    } catch (err) {
      msg.textContent = `Request failed: ${err.message}`;
      msg.style.color = 'var(--danger-color)';
    } finally {
      trainBtn.disabled = false;
      spinner.style.display = 'none';
    }
  });
}

// ==========================================================================
// 6. Model Evaluation API
// ==========================================================================
async function loadModelMetrics() {
  try {
    const res = await fetch('/api/model-metrics');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;

    // Top pill
    const topPill = document.getElementById('top-model-pill');
    if (topPill && data.selected_model) {
      const bestAcc = data.random_forest ? (data.random_forest.accuracy * 100).toFixed(1) : '97.9';
      topPill.innerHTML = `<span>Model: ${data.selected_model} (${bestAcc}%)</span>`;
    }

    // Evaluation stat cards
    const testAcc = document.getElementById('eval-test-acc');
    const macroF1 = document.getElementById('eval-macro-f1');
    const macroRec = document.getElementById('eval-macro-recall');
    const rocAuc = document.getElementById('eval-roc-auc');

    const rf = data.random_forest;
    if (rf) {
      if (testAcc) testAcc.textContent = `${(rf.accuracy * 100).toFixed(2)}%`;
      if (macroF1) macroF1.textContent = rf.f1_macro.toFixed(4);
      if (macroRec) macroRec.textContent = rf.recall_macro.toFixed(4);
      if (rocAuc) rocAuc.textContent = typeof rf.roc_auc_macro === 'number' ? rf.roc_auc_macro.toFixed(4) : rf.roc_auc_macro;
    }

    // Comparison table
    const compTbody = document.getElementById('comparison-tbody');
    if (compTbody && data.comparison_table) {
      compTbody.innerHTML = data.comparison_table.map(row => `
        <tr>
          <td><strong>${row.metric}</strong></td>
          <td><code>${row.logistic_regression}</code></td>
          <td><strong style="color: var(--accent-teal);">${row.random_forest}</strong></td>
          <td style="font-size: 12.5px; color: var(--text-muted);">${row.interpretation}</td>
        </tr>
      `).join('');
    }

    // Per-class table (RF)
    const perClassTbody = document.getElementById('per-class-tbody');
    if (perClassTbody && rf && rf.per_class) {
      const classes = Object.keys(rf.per_class).filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
      perClassTbody.innerHTML = classes.map(cls => {
        const item = rf.per_class[cls];
        return `
          <tr>
            <td><strong>${cls.replace(/_/g, ' ')}</strong></td>
            <td>${(item.precision * 100).toFixed(2)}%</td>
            <td>${(item.recall * 100).toFixed(2)}%</td>
            <td><strong>${(item['f1-score']).toFixed(4)}</strong></td>
            <td>${item.support} instances</td>
          </tr>
        `;
      }).join('');
    }

    // Evaluation images refresh
    const cmImg = document.getElementById('eval-cm-img');
    const rocImg = document.getElementById('eval-roc-img');
    const prImg = document.getElementById('eval-pr-img');
    const cacheBuster = `?t=${Date.now()}`;
    if (cmImg) cmImg.src = `/reports/evaluation/confusion_matrix.png${cacheBuster}`;
    if (rocImg) rocImg.src = `/reports/evaluation/roc_curves.png${cacheBuster}`;
    if (prImg) prImg.src = `/reports/evaluation/pr_curves.png${cacheBuster}`;
  } catch (err) {
    console.error('Failed to load model metrics:', err);
  }
}

// ==========================================================================
// 7. Live BMI Calculator & Form Helpers
// ==========================================================================
function getWHOBMICategory(bmi) {
  if (bmi < 18.5) return { label: 'Insufficient Weight (< 18.5)', class: 'online' };
  if (bmi < 25.0) return { label: 'Normal Weight (18.5 - 24.9)', class: 'online' };
  if (bmi < 30.0) return { label: 'Overweight (25.0 - 29.9)', class: 'model-ready' };
  if (bmi < 35.0) return { label: 'Obesity Class I (30.0 - 34.9)', class: 'model-ready' };
  if (bmi < 40.0) return { label: 'Obesity Class II (35.0 - 39.9)', class: '' };
  return { label: 'Obesity Class III (>= 40.0)', class: '' };
}

function initBMICalculator() {
  const hInput = document.getElementById('inp-height');
  const wInput = document.getElementById('inp-weight');
  const bmiDisplay = document.getElementById('live-bmi-display');
  const bmiBadge = document.getElementById('live-bmi-badge');

  function update() {
    const hCm = parseFloat(hInput.value);
    const wKg = parseFloat(wInput.value);

    if (hCm > 0 && wKg > 0) {
      const hM = hCm / 100.0;
      const bmi = wKg / (hM * hM);
      const cat = getWHOBMICategory(bmi);

      if (bmiDisplay) bmiDisplay.textContent = `${bmi.toFixed(2)} kg/m\u00B2`;
      if (bmiBadge) {
        bmiBadge.textContent = cat.label;
        bmiBadge.className = `status-badge ${cat.class}`;
      }
    }
  }

  if (hInput && wInput) {
    hInput.addEventListener('input', update);
    wInput.addEventListener('input', update);
    update();
  }
}

function initSamplePatientLoader() {
  const btn = document.getElementById('load-sample-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    document.getElementById('inp-age').value = '24';
    document.getElementById('inp-gender').value = 'Female';
    document.getElementById('inp-height').value = '165';
    document.getElementById('inp-weight').value = '80';
    document.getElementById('inp-family-history').value = 'yes';
    document.getElementById('inp-favc').value = 'yes';
    document.getElementById('inp-fcvc').value = '2.0';
    document.getElementById('inp-ncp').value = '3.0';
    document.getElementById('inp-caec').value = 'Sometimes';
    document.getElementById('inp-smoke').value = 'no';
    document.getElementById('inp-ch2o').value = '2.0';
    document.getElementById('inp-scc').value = 'no';
    document.getElementById('inp-faf').value = '1.0';
    document.getElementById('inp-tue').value = '1.0';
    document.getElementById('inp-calc').value = 'Sometimes';
    document.getElementById('inp-mtrans').value = 'Public_Transportation';

    // Trigger live BMI recalculation
    document.getElementById('inp-height').dispatchEvent(new Event('input'));
  });
}

// ==========================================================================
// 8. Patient Prediction Submission & Output Panel
// ==========================================================================
function initFormSubmission() {
  const form = document.getElementById('prediction-form');
  const container = document.getElementById('prediction-result-container');
  const submitBtn = document.getElementById('predict-submit-btn');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const payload = {
      Age: parseFloat(document.getElementById('inp-age').value),
      Gender: document.getElementById('inp-gender').value,
      Height: parseFloat(document.getElementById('inp-height').value),
      Weight: parseFloat(document.getElementById('inp-weight').value),
      family_history_with_overweight: document.getElementById('inp-family-history').value,
      FAVC: document.getElementById('inp-favc').value,
      FCVC: parseFloat(document.getElementById('inp-fcvc').value),
      NCP: parseFloat(document.getElementById('inp-ncp').value),
      CAEC: document.getElementById('inp-caec').value,
      SMOKE: document.getElementById('inp-smoke').value,
      CH2O: parseFloat(document.getElementById('inp-ch2o').value),
      SCC: document.getElementById('inp-scc').value,
      FAF: parseFloat(document.getElementById('inp-faf').value),
      TUE: parseFloat(document.getElementById('inp-tue').value),
      CALC: document.getElementById('inp-calc').value,
      MTRANS: document.getElementById('inp-mtrans').value
    };

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Running inference...</span>';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const json = await res.json();

      if (!json.success) {
        container.innerHTML = `
          <div class="card" style="border-left: 4px solid var(--danger-color);">
            <h3 style="color: var(--danger-color); font-size: 16px; margin-bottom: 8px;">Prediction Error</h3>
            <p style="font-size: 13px; color: var(--text-dark);">${json.error}</p>
          </div>
        `;
        return;
      }

      renderPredictionResult(json, container);
    } catch (err) {
      container.innerHTML = `
        <div class="card" style="border-left: 4px solid var(--danger-color);">
          <h3 style="color: var(--danger-color); font-size: 16px; margin-bottom: 8px;">Network Error</h3>
          <p style="font-size: 13px; color: var(--text-dark);">${err.message}</p>
        </div>
      `;
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>Generate Obesity Level Prediction</span>';
    }
  });
}

function renderPredictionResult(data, container) {
  const probs = data.class_probabilities || {};
  const probBars = Object.keys(probs).map(cls => {
    const val = probs[cls];
    const isWinner = cls === data.predicted_class;
    return `
      <div class="prob-bar-container">
        <div class="prob-header">
          <span style="${isWinner ? 'font-weight: 700; color: var(--primary-blue);' : 'color: var(--text-muted);'}">
            ${isWinner ? '&#10004; ' : ''}${cls.replace(/_/g, ' ')}
          </span>
          <span style="font-weight: 600;">${val.toFixed(1)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${Math.max(val, 2)}%; ${isWinner ? 'background: linear-gradient(90deg, #159A9C, #2878B5);' : 'background: #CBD5E1;'}"></div>
        </div>
      </div>
    `;
  }).join('');

  const featureRows = (data.influential_features || []).map(f => `
    <tr>
      <td><strong>${f.feature}</strong></td>
      <td><code>${f.patient_value}</code></td>
      <td><strong style="color: var(--accent-teal);">${f.importance_percentage}%</strong></td>
      <td style="font-size: 12px; color: var(--text-muted);">${f.impact_note}</td>
    </tr>
  `).join('');

  container.innerHTML = `
    <!-- Top Result Banner -->
    <div class="prediction-panel">
      <h3>Predicted Obesity Category</h3>
      <div class="prediction-category">${data.predicted_class_clean}</div>

      <div class="prediction-metrics-row">
        <div class="pred-metric">
          <div class="label">Calculated BMI</div>
          <div class="val">${data.bmi.toFixed(2)} kg/m&sup2;</div>
        </div>
        <div class="pred-metric">
          <div class="label">WHO Reference Tier</div>
          <div class="val" style="font-size: 16px;">${data.who_category}</div>
        </div>
        <div class="pred-metric">
          <div class="label">Model Class Probability</div>
          <div class="val">${data.model_probability.toFixed(1)}%</div>
        </div>
      </div>

      <p style="font-size: 13px; line-height: 1.5; color: rgba(255, 255, 255, 0.9);">
        ${data.explanation}
      </p>
    </div>

    <!-- Class Probability Distribution -->
    <div class="card">
      <div class="card-title">
        <span>Full Multiclass Probability Distribution</span>
      </div>
      <div class="card-subtitle">Estimated likelihood across all seven categorical tiers</div>
      ${probBars}
    </div>

    <!-- Influential Features Interpretation -->
    <div class="card">
      <div class="card-title">
        <span>Most Influential Predictors for this Assessment</span>
      </div>
      <div class="card-subtitle">Derived from model ensemble split importance and patient biometric deviation</div>
      <div class="table-responsive">
        <table class="data-table">
          <thead>
            <tr>
              <th>Feature Name</th>
              <th>Patient Value</th>
              <th>Relative Weight</th>
              <th>Clinical / Lifestyle Note</th>
            </tr>
          </thead>
          <tbody>
            ${featureRows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Mandatory Clinical Disclaimer -->
    <div class="alert alert-warning" style="font-size: 12px; line-height: 1.5;">
      <div>
        <strong>Academic Governance Notice:</strong> ${data.disclaimer}
      </div>
    </div>
  `;
}


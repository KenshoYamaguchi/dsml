// JavaScript for ML Analysis System

document.addEventListener('DOMContentLoaded', function() {
    // Chart type change handler
    const chartTypeSelect = document.getElementById('chartType');
    const scatterOptions = document.getElementById('scatterOptions');
    
    if (chartTypeSelect && scatterOptions) {
        chartTypeSelect.addEventListener('change', function() {
            if (this.value === 'scatter') {
                scatterOptions.style.display = 'block';
            } else {
                scatterOptions.style.display = 'none';
            }
        });
    }
    
    // Visualization button handler
    const visualizeBtn = document.getElementById('visualizeBtn');
    const columnSelect = document.getElementById('columnSelect');
    const plotDiv = document.getElementById('plotDiv');
    
    if (visualizeBtn && columnSelect && plotDiv) {
        visualizeBtn.addEventListener('click', function() {
            const column = columnSelect.value;
            const chartType = chartTypeSelect.value;
            
            if (!column) {
                alert('列を選択してください');
                return;
            }
            
            // Show loading
            plotDiv.innerHTML = '<div class="text-center"><div class="loading"></div><p>作成中...</p></div>';
            
            // Prepare request data
            let requestData = {
                column: column,
                chart_type: chartType
            };
            
            // Add scatter plot specific data
            if (chartType === 'scatter') {
                const xColumn = document.getElementById('xColumn').value;
                const yColumn = document.getElementById('yColumn').value;
                
                if (!xColumn || !yColumn) {
                    alert('散布図にはX軸とY軸の列を選択してください');
                    plotDiv.innerHTML = '';
                    return;
                }
                
                requestData.x_column = xColumn;
                requestData.y_column = yColumn;
            }
            
            // Make AJAX request
            fetch('/visualize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    plotDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    try {
                        // Clear the div and display the base64 image
                        plotDiv.innerHTML = '';
                        
                        // Create image element with base64 data
                        const imgElement = document.createElement('img');
                        imgElement.src = `data:image/png;base64,${data.plot}`;
                        imgElement.className = 'img-fluid';
                        imgElement.alt = 'Data Visualization';
                        imgElement.style.maxWidth = '100%';
                        imgElement.style.height = 'auto';
                        
                        plotDiv.appendChild(imgElement);
                    } catch (plotError) {
                        console.error('Plot rendering error:', plotError);
                        console.log('Plot data:', data.plot);
                        plotDiv.innerHTML = `<div class="alert alert-danger">グラフの描画に失敗しました: ${plotError.message}</div>`;
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                plotDiv.innerHTML = '<div class="alert alert-danger">可視化中にエラーが発生しました</div>';
            });
        });
    }
    
    // Form validation for model training
    const trainForm = document.querySelector('form[action*="train"]');
    if (trainForm) {
        trainForm.addEventListener('submit', function(e) {
            const targetColumn = document.getElementById('target_column').value;
            const featureColumns = Array.from(document.getElementById('feature_columns').selectedOptions).map(option => option.value);
            
            if (!targetColumn) {
                e.preventDefault();
                alert('目的変数を選択してください');
                return;
            }
            
            if (featureColumns.length === 0) {
                e.preventDefault();
                alert('説明変数を1つ以上選択してください');
                return;
            }
            
            if (featureColumns.includes(targetColumn)) {
                e.preventDefault();
                alert('目的変数と説明変数に同じ列を選択することはできません');
                return;
            }
            
            // Show loading state
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<div class="loading"></div> 学習中...';
                submitBtn.disabled = true;
            }
        });
    }
    
    // File upload validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            if (this.files.length > 0) {
                const file = this.files[0];
                const maxSize = 16 * 1024 * 1024; // 16MB
                
                if (file.size > maxSize) {
                    alert('ファイルサイズが大きすぎます（最大16MB）');
                    this.value = '';
                    return;
                }
                
                const allowedTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
                if (!allowedTypes.includes(file.type) && !file.name.match(/\.(csv|xls|xlsx)$/i)) {
                    alert('許可されていないファイル形式です。CSV、XLS、XLSXファイルのみ対応しています。');
                    this.value = '';
                    return;
                }
            }
        });
    });
    
    // Multi-select helper text
    const multiSelects = document.querySelectorAll('select[multiple]');
    multiSelects.forEach(select => {
        if (!select.parentNode.querySelector('.multi-select-help')) {
            const helpText = document.createElement('div');
            helpText.className = 'form-text multi-select-help';
            helpText.textContent = 'Ctrl/Cmdキーを押しながらクリックで複数選択できます';
            select.parentNode.appendChild(helpText);
        }
    });
    
    // Auto-resize textareas
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    });
    
    // Tooltip initialization (if Bootstrap tooltips are used)
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Progress indication for form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitButtons = this.querySelectorAll('button[type="submit"], input[type="submit"]');
            submitButtons.forEach(btn => {
                if (!btn.classList.contains('no-loading')) {
                    const originalText = btn.textContent || btn.value;
                    btn.innerHTML = '<div class="loading"></div> 処理中...';
                    btn.disabled = true;
                    
                    // Re-enable button after 30 seconds (failsafe)
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.disabled = false;
                    }, 30000);
                }
            });
        });
    });
});

// Utility functions

// Format numbers for display
function formatNumber(num, decimals = 2) {
    if (typeof num !== 'number') return num;
    return num.toFixed(decimals);
}

// Show/hide loading overlay
function showLoading(message = '処理中...') {
    const overlay = document.createElement('div');
    overlay.id = 'loadingOverlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading"></div>
            <p>${message}</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

// Display success/error messages
function showMessage(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Download function for results
function downloadData(data, filename, type = 'text/csv') {
    const blob = new Blob([data], { type: type });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}
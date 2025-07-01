/**
 * Hardware Monitor Plugin for ComfyUI
 * Draggable floating window displaying hardware monitoring information with progress bars
 * Supports custom transparency, colors, background and configuration saving
 */

// Start checking DOM directly
(function() {
    // Add necessary CSS styles
    const styleSheet = document.createElement('style');
    styleSheet.id = 'hardware-monitor-styles';
    styleSheet.textContent = `
        .hardware-monitor-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 8px 10px;
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 6px;
            font-size: 12px;
            color: #fff;
            box-sizing: border-box;
            border: 1px solid rgba(100, 100, 100, 0.4);
            user-select: none;
            width: 100%;
            cursor: move;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        
        /* Floating window styles */
        .hardware-monitor-float {
            position: fixed;
            top: 50px;
            right: 50px;
            z-index: 1000;
            cursor: move;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            user-select: none;
            pointer-events: auto;
            width: 250px;
            resize: none !important;
        }
        
        /* Floating window header styles */
        .monitor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background-color: rgba(50, 50, 50, 0.9);
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: bold;
            cursor: move;
            user-select: none;
            touch-action: none;
        }
        
        .monitor-header-title {
            font-size: 13px;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
        }
        
        .monitor-header-controls {
            display: flex;
            gap: 8px;
        }
        
        .monitor-control-btn {
            cursor: pointer;
            width: 18px;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
            font-size: 12px;
            background-color: rgba(80, 80, 80, 0.7);
            transition: background-color 0.2s;
        }
        
        .monitor-control-btn:hover {
            background-color: rgba(120, 120, 120, 0.9);
        }
        
        /* Settings button styles */
        .monitor-settings-btn {
            cursor: pointer;
            width: 18px;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
            font-size: 12px;
            background-color: rgba(80, 80, 80, 0.7);
            transition: background-color 0.2s;
        }
        
        .monitor-settings-btn:hover {
            background-color: rgba(120, 120, 120, 0.9);
        }
        
        /* Settings panel styles */
        .monitor-settings-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 450px;
            max-height: 600px;
            background-color: rgba(40, 40, 40, 0.95);
            border: 2px solid rgba(100, 100, 100, 0.6);
            border-radius: 8px;
            z-index: 2000;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            color: #fff;
            font-size: 13px;
            overflow-y: auto;
        }
        
        .settings-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .settings-panel-title {
            font-size: 16px;
            font-weight: bold;
        }
        
        .settings-close-btn {
            cursor: pointer;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            background-color: rgba(80, 80, 80, 0.7);
            transition: background-color 0.2s;
        }
        
        .settings-close-btn:hover {
            background-color: rgba(120, 120, 120, 0.9);
        }
        
        .settings-group {
            margin-bottom: 20px;
        }
        
        .settings-group-title {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #ccc;
        }
        
        .settings-item {
            margin-bottom: 15px;
        }
        
        .settings-item-label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .settings-slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.2);
            outline: none;
            -webkit-appearance: none;
        }
        
        .settings-slider::-webkit-slider-thumb {
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }
        
        .settings-slider::-moz-range-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            border: none;
        }
        
        .settings-color-input {
            width: 100%;
            height: 40px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .individual-color {
            width: 60px !important;
            height: 30px !important;
            margin-left: 10px;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .individual-color:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
        }
        
        .settings-file-input {
            width: 100%;
            padding: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            background-color: rgba(60, 60, 60, 0.7);
            color: #fff;
        }
        
        .settings-button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            font-size: 12px;
            margin-right: 10px;
            margin-bottom: 5px;
            transition: background-color 0.2s;
        }
        
        .settings-button:hover {
            background-color: #45a049;
        }
        
        .settings-button.secondary {
            background-color: #666;
        }
        
        .settings-button.secondary:hover {
            background-color: #777;
        }
        
        .color-preset {
            display: inline-block;
            width: 30px;
            height: 30px;
            border-radius: 4px;
            margin: 2px;
            cursor: pointer;
            border: 2px solid transparent;
            transition: border-color 0.2s;
        }
        
        .color-preset:hover {
            border-color: #fff;
        }
        
        .color-preset.selected {
            border-color: #4CAF50;
        }
        
        .history-colors {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        
        .settings-value-display {
            display: inline-block;
            margin-left: 10px;
            font-weight: normal;
            color: #ccc;
        }
        
        /* Scrollbar styles */
        .monitor-settings-panel::-webkit-scrollbar {
            width: 8px;
        }
        
        .monitor-settings-panel::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        .monitor-settings-panel::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
        }
        
        .monitor-settings-panel::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        /* File input optimization */
        .settings-file-input::-webkit-file-upload-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        
        .settings-file-input::-webkit-file-upload-button:hover {
            background-color: #45a049;
        }
        
        /* Animation effects */
        .monitor-settings-panel {
            animation: fadeIn 0.3s ease-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.9);
            }
            to {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
        }
        
        /* Responsive design */
        @media (max-width: 500px) {
            .monitor-settings-panel {
                width: 90%;
                max-width: 400px;
            }
        }
        
        /* Floating window content area */
        .monitor-content {
            padding: 5px;
        }
        
        /* Opacity control */
        .monitor-opacity-high {
            opacity: 1;
            transition: opacity 0.2s;
        }
        
        .monitor-opacity-low {
            opacity: 0.5;
            transition: opacity 0.2s;
        }
        
        /* Divider */
        .monitor-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.2);
            margin: 4px 0;
            width: 100%;
        }
        
        /* Cleanup button */
        .monitor-cleanup-button {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: rgba(60, 60, 60, 0.7);
            border-radius: 4px;
            padding: 6px 10px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 2px;
        }
        
        .monitor-cleanup-button:hover {
            background-color: rgba(80, 80, 80, 0.9);
        }
        
        .cleanup-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #888;
            margin-left: 5px;
        }
        
        .cleanup-text {
            font-size: 12px;
            font-weight: bold;
        }
        
        .monitor-item {
            position: relative;
            height: 26px;
            margin: 4px 0;
            box-sizing: border-box;
            display: flex;
            align-items: stretch;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid rgba(80, 80, 80, 0.4);
        }
        
        .monitor-group {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        
        .monitor-group-title {
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 4px;
            padding-left: 4px;
            color: rgba(255, 255, 255, 0.9);
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
        }
        
        /* Add different background colors for different types of monitor items */
        .system-group .monitor-item {
            background-color: rgba(0, 0, 0, 0.3);
        }
        
        .gpu-group .monitor-item {
            background-color: rgba(50, 50, 50, 0.3);
        }
        
        .monitor-label {
            position: absolute;
            left: 8px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 12px;
            font-weight: bold;
            z-index: 5;
            text-shadow: 0 0 2px rgba(0, 0, 0, 0.8);
        }
        
        .monitor-bar {
            position: absolute;
            width: 100%;
            height: 100%;
            background-color: rgba(20, 20, 20, 0.7);
            overflow: hidden;
            box-sizing: border-box;
        }
        
        .monitor-progress {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
            transition: width 0.3s ease, background 0.3s ease;
            box-sizing: border-box;
            box-shadow: inset 0 0 5px rgba(255, 255, 255, 0.2);
        }
        
        .monitor-value {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 12px;
            font-weight: bold;
            z-index: 5;
            text-shadow: 0 0 2px rgba(0, 0, 0, 0.8);
        }
        
        .cpu-progress { background-color: rgba(76, 175, 80, 0.8); }
        .ram-progress { background-color: rgba(33, 150, 243, 0.8); }
        .gpu-progress { background-color: rgba(255, 152, 0, 0.8); }
        .vram-progress { background-color: rgba(156, 39, 176, 0.8); }
        .temp-progress { background-color: rgba(244, 67, 54, 0.8); }
        
        /* Resize handle styles */
        .resize-handle {
            position: absolute;
            width: 15px;
            height: 15px;
            background-color: transparent;
            border-radius: 0;
            z-index: 1002;
            cursor: nwse-resize;
            border: none;
            box-shadow: none;
        }
        
        .resize-handle-br {
            right: 0;
            bottom: 0;
            width: 15px;
            height: 15px;
        }
        
        .resize-handle-bl {
            left: 0;
            bottom: 0;
            width: 15px;
            height: 15px;
        }
    `;
    document.head.appendChild(styleSheet);
    
    // Create floating window
    setupHardwareMonitorFloating();
})();

// Format bytes to readable format
function formatBytes(bytes) {
    if (bytes === 0) return '0B';
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)}${sizes[i]}`;
}

// Create floating window monitor
function setupHardwareMonitorFloating() {
    
    // Create floating window container
    const floatContainer = document.createElement('div');
    floatContainer.className = 'hardware-monitor-float monitor-opacity-high';
    floatContainer.id = 'hardware-monitor-float';
    
    // Create floating window header
    const header = document.createElement('div');
    header.className = 'monitor-header';
    
    // Header title
    const title = document.createElement('div');
    title.className = 'monitor-header-title';
    title.textContent = 'System Monitor';
    header.appendChild(title);
    
    // Header control buttons
    const controls = document.createElement('div');
    controls.className = 'monitor-header-controls';
    
    // Settings button
    const settingsBtn = document.createElement('div');
    settingsBtn.className = 'monitor-settings-btn';
    settingsBtn.innerHTML = '⚙';
    settingsBtn.title = 'Settings';
    controls.appendChild(settingsBtn);
    
    // Opacity toggle button
    const opacityBtn = document.createElement('div');
    opacityBtn.className = 'monitor-control-btn';
    opacityBtn.innerHTML = '⊙';
    opacityBtn.title = 'Toggle Opacity';
    controls.appendChild(opacityBtn);
    
    // Add collapse/expand button
    const collapseBtn = document.createElement('div');
    collapseBtn.className = 'monitor-control-btn';
    collapseBtn.innerHTML = '−';
    collapseBtn.title = 'Collapse/Expand';
    controls.appendChild(collapseBtn);
    
    header.appendChild(controls);
    floatContainer.appendChild(header);
    
    // Create content area
    const content = document.createElement('div');
    content.className = 'monitor-content';
    floatContainer.appendChild(content);
    
    // Create hardware monitor container
    const monitorContainer = document.createElement('div');
    monitorContainer.className = 'hardware-monitor-container';
    content.appendChild(monitorContainer);
    
    // Create system resources group (CPU/RAM)
    const systemGroup = document.createElement('div');
    systemGroup.className = 'monitor-group system-group';
    
    // Add system group title
    const systemTitle = document.createElement('div');
    systemTitle.className = 'monitor-group-title';
    systemTitle.textContent = 'System Resources';
    systemGroup.appendChild(systemTitle);
    
    monitorContainer.appendChild(systemGroup);
    
    // Initialize displayed information
    const cpuElement = createMonitorElement('CPU', '0%', 'cpu-progress');
    const ramElement = createMonitorElement('Memory', '0%', 'ram-progress');
    
    // Add to system resources group
    systemGroup.appendChild(cpuElement.container);
    systemGroup.appendChild(ramElement.container);
    
    // Create GPU divider placeholder (create only once)
    const gpuDivider = document.createElement('div');
    gpuDivider.className = 'monitor-divider';
    gpuDivider.id = 'gpu-divider';
    gpuDivider.style.display = 'none'; // Initially hidden
    monitorContainer.appendChild(gpuDivider);
    
    // Add auto cleanup button
    const cleanupButton = document.createElement('div');
    cleanupButton.className = 'monitor-cleanup-button';
    
    const cleanupText = document.createElement('span');
    cleanupText.className = 'cleanup-text';
    cleanupText.textContent = 'Auto Clear VRAM';
    cleanupButton.appendChild(cleanupText);
    
    const indicator = document.createElement('div');
    indicator.className = 'cleanup-indicator';
    cleanupButton.appendChild(indicator);
    
    monitorContainer.appendChild(cleanupButton);
    
    // Add resize handles
    const resizeHandleBR = document.createElement('div');
    resizeHandleBR.className = 'resize-handle resize-handle-br';
    floatContainer.appendChild(resizeHandleBR);
    
    const resizeHandleBL = document.createElement('div');
    resizeHandleBL.className = 'resize-handle resize-handle-bl';
    floatContainer.appendChild(resizeHandleBL);
    
    // Add to page
    document.body.appendChild(floatContainer);
    
    // Create tooltip
    const tooltip = document.createElement('div');
    tooltip.style.position = 'absolute';
    tooltip.style.display = 'none';
    tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    tooltip.style.color = 'white';
    tooltip.style.padding = '5px 8px';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '12px';
    tooltip.style.zIndex = '1001';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.whiteSpace = 'nowrap';
    tooltip.textContent = 'Last Cleanup Time: None';
    document.body.appendChild(tooltip);
    
    // Show tooltip on mouse hover
    cleanupButton.addEventListener('mouseenter', (e) => {
        tooltip.style.display = 'block';
        updateTooltipPosition(e);
    });
    
    cleanupButton.addEventListener('mousemove', updateTooltipPosition);
    
    cleanupButton.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
    
    // Update tooltip position
    function updateTooltipPosition(e) {
        const rect = cleanupButton.getBoundingClientRect();
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.bottom + 5}px`;
    }
    
    // Initialize auto cleanup variables
    let isActive = false;
    let intervalId = null;
    let lastCleanupTime = null;
    
    // Configuration variables
    let config = {
        opacity: 0.8,
        progressColor: '#4CAF50', // Keep as default color
        backgroundColor: 'rgba(30, 30, 30, 0.7)',
        backgroundImage: null,
        backgroundImageOpacity: 0.8,
        colorHistory: ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'],
        individualColors: {
            cpu: '#4CAF50',
            ram: '#2196F3',
            gpu: '#FF9800',
            temp: '#E91E63',
            vram: '#9C27B0'
        },
        cleanupInterval: 1 // Default 1 second cleanup interval
    };
    
    // Create settings panel
    function createSettingsPanel() {
        const panel = document.createElement('div');
        panel.className = 'monitor-settings-panel';
        panel.id = 'monitor-settings-panel';
        panel.style.display = 'none';
        
        panel.innerHTML = `
            <div class="settings-panel-header">
                <div class="settings-panel-title">Hardware Monitor Settings</div>
                <div class="settings-close-btn">×</div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">Appearance Settings</div>
                
                <div class="settings-item">
                    <label class="settings-item-label">Opacity: <span class="settings-value-display" id="opacity-value">${Math.round(config.opacity * 100)}%</span></label>
                    <input type="range" class="settings-slider" id="opacity-slider" min="0.1" max="1" step="0.1" value="${config.opacity}">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">Background Image:</label>
                    <input type="file" class="settings-file-input" id="background-image" accept="image/*">
                    <button class="settings-button secondary" id="remove-background">Remove Background</button>
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">Background Image Opacity: <span class="settings-value-display" id="bg-opacity-value">${Math.round(config.backgroundImageOpacity * 100)}%</span></label>
                    <input type="range" class="settings-slider" id="bg-opacity-slider" min="0.1" max="1" step="0.1" value="${config.backgroundImageOpacity}">
                </div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">Progress Bar Colors</div>
                
                <div class="settings-item">
                    <label class="settings-item-label">CPU Usage:</label>
                    <input type="color" class="settings-color-input individual-color" id="cpu-color" value="${config.individualColors.cpu}" data-type="cpu">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">Memory Usage:</label>
                    <input type="color" class="settings-color-input individual-color" id="ram-color" value="${config.individualColors.ram}" data-type="ram">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">GPU Usage:</label>
                    <input type="color" class="settings-color-input individual-color" id="gpu-color" value="${config.individualColors.gpu}" data-type="gpu">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">GPU Temperature:</label>
                    <input type="color" class="settings-color-input individual-color" id="temp-color" value="${config.individualColors.temp}" data-type="temp">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">VRAM Usage:</label>
                    <input type="color" class="settings-color-input individual-color" id="vram-color" value="${config.individualColors.vram}" data-type="vram">
                </div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">Auto Cleanup Settings</div>
                
                <div class="settings-item">
                    <label class="settings-item-label">Cleanup Interval (seconds): <span class="settings-value-display" id="cleanup-interval-value">${config.cleanupInterval}</span></label>
                    <input type="range" class="settings-slider" id="cleanup-interval-slider" min="1" max="300" step="1" value="${config.cleanupInterval}">
                </div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">Quick Colors</div>
                <div style="font-size: 11px; color: #aaa; margin-bottom: 10px;">Click a color block to apply to the last selected progress bar</div>
                <div id="color-presets">
                    <div class="color-preset" style="background-color: #4CAF50" data-color="#4CAF50"></div>
                    <div class="color-preset" style="background-color: #2196F3" data-color="#2196F3"></div>
                    <div class="color-preset" style="background-color: #FF9800" data-color="#FF9800"></div>
                    <div class="color-preset" style="background-color: #E91E63" data-color="#E91E63"></div>
                    <div class="color-preset" style="background-color: #9C27B0" data-color="#9C27B0"></div>
                    <div class="color-preset" style="background-color: #00BCD4" data-color="#00BCD4"></div>
                    <div class="color-preset" style="background-color: #8BC34A" data-color="#8BC34A"></div>
                    <div class="color-preset" style="background-color: #FF5722" data-color="#FF5722"></div>
                </div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">Color History</div>
                <div class="history-colors" id="history-colors"></div>
                <button class="settings-button secondary" id="clear-history">Clear History</button>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">Configuration Management</div>
                <button class="settings-button" id="save-config">Save Configuration</button>
                <button class="settings-button secondary" id="reset-config">Reset to Default</button>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // Bind events
        bindSettingsEvents(panel);
        
        return panel;
    }
    
    // Bind settings panel events
    function bindSettingsEvents(panel) {
        const closeBtn = panel.querySelector('.settings-close-btn');
        const opacitySlider = panel.querySelector('#opacity-slider');
        const opacityValue = panel.querySelector('#opacity-value');
        const backgroundImageInput = panel.querySelector('#background-image');
        const removeBackgroundBtn = panel.querySelector('#remove-background');
        const bgOpacitySlider = panel.querySelector('#bg-opacity-slider');
        const bgOpacityValue = panel.querySelector('#bg-opacity-value');
        const individualColorInputs = panel.querySelectorAll('.individual-color');
        const colorPresets = panel.querySelectorAll('.color-preset');
        const saveConfigBtn = panel.querySelector('#save-config');
        const resetConfigBtn = panel.querySelector('#reset-config');
        const clearHistoryBtn = panel.querySelector('#clear-history');
        const cleanupIntervalSlider = panel.querySelector('#cleanup-interval-slider');
        const cleanupIntervalValue = panel.querySelector('#cleanup-interval-value');
        
        // Close panel
        closeBtn.addEventListener('click', () => {
            panel.style.display = 'none';
        });
        
        // Close panel when clicking outside
        panel.addEventListener('click', (e) => {
            if (e.target === panel) {
                panel.style.display = 'none';
            }
        });
        
        // ESC key to close panel
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && panel.style.display === 'block') {
                panel.style.display = 'none';
            }
        });
        
        // Opacity adjustment
        opacitySlider.addEventListener('input', (e) => {
            config.opacity = parseFloat(e.target.value);
            opacityValue.textContent = Math.round(config.opacity * 100) + '%';
            applyOpacity();
            saveConfig();
        });
        
        // Background image opacity adjustment
        bgOpacitySlider.addEventListener('input', (e) => {
            config.backgroundImageOpacity = parseFloat(e.target.value);
            bgOpacityValue.textContent = Math.round(config.backgroundImageOpacity * 100) + '%';
            applyBackgroundImage();
            saveConfig();
        });
        
        // Cleanup interval adjustment
        cleanupIntervalSlider.addEventListener('input', (e) => {
            config.cleanupInterval = parseInt(e.target.value);
            cleanupIntervalValue.textContent = config.cleanupInterval;
            
            // If auto cleanup is running, restart to apply new interval
            if (isActive) {
                if (intervalId) {
                    clearInterval(intervalId);
                }
                intervalId = setInterval(performCleanup, config.cleanupInterval * 1000);
            }
            
            saveConfig();
        });
        
        // Individual color adjustment
        individualColorInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                const type = e.target.dataset.type;
                const color = e.target.value;
                config.individualColors[type] = color;
                applyIndividualColor(type, color);
                addToColorHistory(color);
                saveConfig();
            });
        });
        
        // Background image upload
        backgroundImageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    config.backgroundImage = e.target.result;
                    applyBackgroundImage();
                    saveConfig();
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Remove background
        removeBackgroundBtn.addEventListener('click', () => {
            config.backgroundImage = null;
            applyBackgroundImage();
            backgroundImageInput.value = '';
            saveConfig();
        });
        
        // Preset color selection - apply to last clicked color input
        let lastClickedColorInput = null;
        
        individualColorInputs.forEach(input => {
            input.addEventListener('focus', () => {
                lastClickedColorInput = input;
            });
        });
        
        colorPresets.forEach(preset => {
            preset.addEventListener('click', () => {
                const color = preset.dataset.color;
                
                // If there's a last clicked color input, apply to that input
                if (lastClickedColorInput) {
                    const type = lastClickedColorInput.dataset.type;
                    config.individualColors[type] = color;
                    lastClickedColorInput.value = color;
                    applyIndividualColor(type, color);
                } else {
                    // Default apply to CPU color
                    config.individualColors.cpu = color;
                    panel.querySelector('#cpu-color').value = color;
                    applyIndividualColor('cpu', color);
                }
                
                addToColorHistory(color);
                saveConfig();
                
                // Update selected state
                colorPresets.forEach(p => p.classList.remove('selected'));
                preset.classList.add('selected');
            });
        });
        
        // Save configuration
        saveConfigBtn.addEventListener('click', () => {
            saveConfig();
            alert('Configuration saved!');
        });
        
        // Reset configuration
        resetConfigBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to reset to default configuration?')) {
                resetConfig();
                updateSettingsPanel();
            }
        });
        
        // Clear history
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear color history?')) {
                config.colorHistory = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'];
                updateHistoryColors();
                saveConfig();
            }
        });
    }
    
    // Apply opacity
    function applyOpacity() {
        floatContainer.style.opacity = config.opacity;
    }
    
    // Apply individual progress bar color
    function applyIndividualColor(type, color) {
        let selector = '';
        switch(type) {
            case 'cpu':
                selector = '.cpu-progress';
                break;
            case 'ram':
                selector = '.ram-progress';
                break;
            case 'gpu':
                selector = '.gpu-progress';
                break;
            case 'temp':
                selector = '.temp-progress';
                break;
            case 'vram':
                selector = '.vram-progress';
                break;
        }
        
        if (selector) {
            const progressBars = floatContainer.querySelectorAll(selector);
            progressBars.forEach(bar => {
                bar.style.background = `linear-gradient(90deg, ${color} 0%, ${adjustColorBrightness(color, -20)} 100%)`;
            });
        }
    }
    
    // Apply all progress bar colors
    function applyAllProgressColors() {
        Object.keys(config.individualColors).forEach(type => {
            applyIndividualColor(type, config.individualColors[type]);
        });
    }
    
    // Apply background image
    function applyBackgroundImage() {
        const container = floatContainer.querySelector('.hardware-monitor-container');
        if (config.backgroundImage) {
            container.style.backgroundImage = `url(${config.backgroundImage})`;
            // Set background color based on background image opacity
            const alpha = 1 - config.backgroundImageOpacity;
            container.style.backgroundColor = `rgba(30, 30, 30, ${alpha})`;
            
            // If there's a background image, add a semi-transparent overlay to control image opacity
            let overlay = container.querySelector('.bg-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'bg-overlay';
                overlay.style.position = 'absolute';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.right = '0';
                overlay.style.bottom = '0';
                overlay.style.pointerEvents = 'none';
                overlay.style.borderRadius = '6px';
                container.style.position = 'relative';
                container.appendChild(overlay);
            }
            const overlayAlpha = 1 - config.backgroundImageOpacity;
            overlay.style.backgroundColor = `rgba(30, 30, 30, ${overlayAlpha})`;
        } else {
            container.style.backgroundImage = 'none';
            container.style.backgroundColor = config.backgroundColor;
            // Remove overlay
            const overlay = container.querySelector('.bg-overlay');
            if (overlay) {
                overlay.remove();
            }
        }
    }
    
    // Color brightness adjustment function
    function adjustColorBrightness(color, amount) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * amount);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 + (B < 255 ? B < 1 ? 0 : B : 255))
            .toString(16).slice(1);
    }
    
    // Add to color history
    function addToColorHistory(color) {
        if (!config.colorHistory.includes(color)) {
            config.colorHistory.unshift(color);
            if (config.colorHistory.length > 10) {
                config.colorHistory = config.colorHistory.slice(0, 10);
            }
            updateHistoryColors();
        }
    }
    
    // Update history colors display
    function updateHistoryColors() {
        const historyContainer = document.querySelector('#history-colors');
        if (historyContainer) {
            historyContainer.innerHTML = '';
            config.colorHistory.forEach(color => {
                const colorDiv = document.createElement('div');
                colorDiv.className = 'color-preset';
                colorDiv.style.backgroundColor = color;
                colorDiv.dataset.color = color;
                colorDiv.addEventListener('click', () => {
                    // Apply to last clicked color input
                    if (lastClickedColorInput) {
                        const type = lastClickedColorInput.dataset.type;
                        config.individualColors[type] = color;
                        lastClickedColorInput.value = color;
                        applyIndividualColor(type, color);
                    } else {
                        // Default apply to CPU color
                        config.individualColors.cpu = color;
                        const cpuInput = document.querySelector('#cpu-color');
                        if (cpuInput) {
                            cpuInput.value = color;
                            applyIndividualColor('cpu', color);
                        }
                    }
                    saveConfig();
                });
                historyContainer.appendChild(colorDiv);
            });
        }
    }
    
    // Update settings panel
    function updateSettingsPanel() {
        const panel = document.querySelector('#monitor-settings-panel');
        if (panel) {
            panel.querySelector('#opacity-slider').value = config.opacity;
            panel.querySelector('#opacity-value').textContent = Math.round(config.opacity * 100) + '%';
            panel.querySelector('#bg-opacity-slider').value = config.backgroundImageOpacity;
            panel.querySelector('#bg-opacity-value').textContent = Math.round(config.backgroundImageOpacity * 100) + '%';
            panel.querySelector('#cleanup-interval-slider').value = config.cleanupInterval;
            panel.querySelector('#cleanup-interval-value').textContent = config.cleanupInterval;
            
            // Update individual color inputs
            Object.keys(config.individualColors).forEach(type => {
                const input = panel.querySelector(`#${type}-color`);
                if (input) {
                    input.value = config.individualColors[type];
                }
            });
            
            updateHistoryColors();
        }
    }
    
    // Save configuration
    function saveConfig() {
        try {
            localStorage.setItem('hardwareMonitorConfig', JSON.stringify(config));
        } catch (e) {
            // Silently handle error
        }
    }
    
    // Load configuration
    function loadConfig() {
        try {
            const savedConfig = localStorage.getItem('hardwareMonitorConfig');
            if (savedConfig) {
                const parsed = JSON.parse(savedConfig);
                // Merge configuration, ensure new config items have default values
                config = { 
                    ...config, 
                    ...parsed,
                    // Ensure individualColors exists
                    individualColors: {
                        ...config.individualColors,
                        ...parsed.individualColors
                    }
                };
                
                // Ensure backgroundImageOpacity exists
                if (typeof config.backgroundImageOpacity === 'undefined') {
                    config.backgroundImageOpacity = 0.8;
                }
                
                // Ensure cleanupInterval exists
                if (typeof config.cleanupInterval === 'undefined') {
                    config.cleanupInterval = 1;
                }
            }
        } catch (e) {
            // Silently handle error
        }
    }
    
    // Reset configuration
    function resetConfig() {
        config = {
            opacity: 0.8,
            progressColor: '#4CAF50',
            backgroundColor: 'rgba(30, 30, 30, 0.7)',
            backgroundImage: null,
            backgroundImageOpacity: 0.8,
            colorHistory: ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'],
            individualColors: {
                cpu: '#4CAF50',
                ram: '#2196F3',
                gpu: '#FF9800',
                temp: '#E91E63',
                vram: '#9C27B0'
            },
            cleanupInterval: 1
        };
        applyOpacity();
        applyAllProgressColors();
        applyBackgroundImage();
        saveConfig();
    }
    
    // Initialize settings
    loadConfig();
    const settingsPanel = createSettingsPanel();
    
    // Settings button click event
    settingsBtn.addEventListener('click', () => {
        settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
        updateSettingsPanel();
    });
    
    // Load state from localStorage
    function loadState() {
        try {
            const savedState = localStorage.getItem('autoCleanupState');
            if (savedState !== null) {
                isActive = JSON.parse(savedState);
                
                // If state is active, start cleanup
                if (isActive) {
                    updateButtonState();
                }
            }
        } catch (e) {
            // Silently handle error
        }
        
        // Load last cleanup time
        try {
            const savedTime = localStorage.getItem('autoCleanupLastTime');
            if (savedTime !== null) {
                lastCleanupTime = new Date(savedTime);
                updateTooltip();
            }
        } catch (e) {
            // Silently handle error
        }
    }
    
    // Save state to localStorage
    function saveState() {
        try {
            localStorage.setItem('autoCleanupState', JSON.stringify(isActive));
        } catch (e) {
            // Silently handle error
        }
    }
    
    // Save last cleanup time
    function saveLastCleanupTime() {
        if (lastCleanupTime) {
            try {
                localStorage.setItem('autoCleanupLastTime', lastCleanupTime.toISOString());
            } catch (e) {
                // Silently handle error
            }
        }
    }
    
    // Add cleanup button click event
    cleanupButton.addEventListener('click', () => {
        isActive = !isActive;
        updateButtonState();
        saveState();
    });
    
    // Update button state
    function updateButtonState() {
        if (isActive) {
            indicator.style.backgroundColor = '#4CAF50'; // Green indicates on
            
            // Start scheduled cleanup
            if (!intervalId) {
                performCleanup(); // Execute once immediately
                intervalId = setInterval(performCleanup, config.cleanupInterval * 1000); // Use configured interval
            }
        } else {
            indicator.style.backgroundColor = '#888'; // Gray indicates off
            
            // Stop scheduled cleanup
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
        }
    }
    
    // Perform cleanup operation
    async function performCleanup() {
        // Get current page host address
        const currentUrl = window.location.href;
        const url = new URL(currentUrl);
        const baseUrl = `${url.protocol}//${url.host}`;
        
        // Build API URL
        const apiUrl = `${baseUrl}/api/free`;
        
        const params = {
            "unload_models": true,
            "free_memory": true
        };

        try {
            const response = await fetch(apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(params)
            });

            if (response.ok) {
                lastCleanupTime = new Date();
                updateTooltip();
                saveLastCleanupTime();
            }
        } catch (error) {
            // Silently handle error
        }
    }
    
    // Update tooltip information
    function updateTooltip() {
        if (lastCleanupTime) {
            const timeString = lastCleanupTime.toLocaleTimeString();
            tooltip.textContent = `Last Cleanup Time: ${timeString}`;
        } else {
            tooltip.textContent = "Last Cleanup Time: None";
        }
    }
    
    // Load saved state
    loadState();
    
    // Set drag and resize related variables
    let isDragging = false;
    let startX, startY;
    let startPosX, startPosY;
    let isResizingBR = false;
    let isResizingBL = false;
    let startWidthBR, startFontScale;
    let startWidthBL, startRectLeft;
    
    // Load position and size from localStorage
    function loadPositionAndSize() {
        try {
            // Delay execution to ensure DOM is fully loaded and rendered
            setTimeout(() => {
                const savedPosition = localStorage.getItem('hardware-monitor-position');
                if (savedPosition) {
                    const position = JSON.parse(savedPosition);
                    floatContainer.style.left = position.left;
                    floatContainer.style.top = position.top;
                }
                
                const savedSize = localStorage.getItem('hardware-monitor-size');
                if (savedSize) {
                    const size = JSON.parse(savedSize);
                    // Ensure using pixel units
                    if (!size.width.endsWith('px')) {
                        size.width += 'px';
                    }
                    floatContainer.style.width = size.width;
                    
                    // Update font size for all child elements
                    const fontScale = size.fontScale || 1;
                    // Save current font scale to monitorElements for future GPU items
                    monitorElements.currentFontScale = fontScale;
                    updateFontSize(fontScale);
                    
                    // Force re-layout
                    void floatContainer.offsetWidth;
                }
            }, 100);
        } catch (e) {
            console.error("Error loading saved position or size:", e);
        }
    }
    
    // Save position and size to localStorage
    function savePosition() {
        const position = {
            left: floatContainer.style.left,
            top: floatContainer.style.top
        };
        localStorage.setItem('hardware-monitor-position', JSON.stringify(position));
    }
    
    function saveSize() {
        // Get computed actual width to ensure saving actual value
        const computedWidth = window.getComputedStyle(floatContainer).width;
        const size = {
            width: computedWidth,
            fontScale: parseFloat(floatContainer.getAttribute('data-font-scale') || '1')
        };
        localStorage.setItem('hardware-monitor-size', JSON.stringify(size));
    }
    
    // Update font size
    function updateFontSize(scale) {
        floatContainer.setAttribute('data-font-scale', scale);
        
        // Titles
        const titles = floatContainer.querySelectorAll('.monitor-group-title, .monitor-header-title');
        titles.forEach(el => {
            el.style.fontSize = `${13 * scale}px`;
        });
        
        // Labels and values
        const labels = floatContainer.querySelectorAll('.monitor-label, .monitor-value');
        labels.forEach(el => {
            el.style.fontSize = `${12 * scale}px`;
        });
        
        // Buttons and indicators
        const buttons = floatContainer.querySelectorAll('.monitor-control-btn, .cleanup-text');
        buttons.forEach(el => {
            el.style.fontSize = `${12 * scale}px`;
        });
        
        // Adjust monitor item height
        const items = floatContainer.querySelectorAll('.monitor-item');
        items.forEach(el => {
            el.style.height = `${26 * scale}px`;
        });
        
        // Save current scale to monitorElements
        if (monitorElements) {
            monitorElements.currentFontScale = scale;
        }
    }
    
    // Listen for drag start on header and content area
    const dragHandlers = [header, content];
    dragHandlers.forEach(element => {
        element.addEventListener('mousedown', function(e) {
            // Don't drag if clicking resize handle or button
            if (e.target.closest('.resize-handle') || e.target.closest('.monitor-control-btn')) {
                return;
            }
            
            // Only handle left click
            if (e.button !== 0) return;
            
            // Record mouse start coordinates
            startX = e.clientX;
            startY = e.clientY;
            
            // Record window start position
            const rect = floatContainer.getBoundingClientRect();
            startPosX = rect.left;
            startPosY = rect.top;
            
            // Start dragging
            isDragging = true;
            
            // Add temporary styles
            floatContainer.style.transition = 'none';
            element.style.cursor = 'grabbing';
            
            // Prevent event propagation and default behavior
            e.preventDefault();
            e.stopPropagation();
        });
    });
    
    // Resize functionality - bottom right corner
    resizeHandleBR.addEventListener('mousedown', function(e) {
        isResizingBR = true;
        startX = e.clientX;
        startWidthBR = parseInt(getComputedStyle(floatContainer).width, 10);
        startFontScale = parseFloat(floatContainer.getAttribute('data-font-scale') || '1');
        
        // Prevent event propagation and default behavior
        e.preventDefault();
        e.stopPropagation();
    });
    
    // Resize functionality - bottom left corner
    resizeHandleBL.addEventListener('mousedown', function(e) {
        isResizingBL = true;
        startX = e.clientX;
        startWidthBL = parseInt(getComputedStyle(floatContainer).width, 10);
        startRectLeft = floatContainer.getBoundingClientRect().left;
        startFontScale = parseFloat(floatContainer.getAttribute('data-font-scale') || '1');
        
        // Prevent event propagation and default behavior
        e.preventDefault();
        e.stopPropagation();
    });
    
    // Listen for mouse move on entire document
    document.addEventListener('mousemove', function(e) {
        // Handle dragging
        if (isDragging) {
            // Calculate mouse movement distance
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            // Calculate new position
            let newX = startPosX + deltaX;
            let newY = startPosY + deltaY;
            
            // Limit within viewport
            const maxX = window.innerWidth - floatContainer.offsetWidth;
            const maxY = window.innerHeight - floatContainer.offsetHeight;
            
            newX = Math.max(0, Math.min(newX, maxX));
            newY = Math.max(0, Math.min(newY, maxY));
            
            // Set new position
            floatContainer.style.left = newX + 'px';
            floatContainer.style.top = newY + 'px';
            
            // Prevent default behavior
            e.preventDefault();
        }
        
        // Handle bottom right resize
        if (isResizingBR) {
            const deltaX = e.clientX - startX;
            const newWidth = Math.max(200, Math.min(startWidthBR + deltaX, 500));
            floatContainer.style.width = newWidth + 'px';
            
            // Scale font size proportionally
            const fontScale = startFontScale * (newWidth / startWidthBR);
            updateFontSize(fontScale);
            
            // Prevent event propagation and default behavior
            e.preventDefault();
        }
        
        // Handle bottom left resize
        if (isResizingBL) {
            const deltaX = startX - e.clientX;
            const newWidth = Math.max(200, Math.min(startWidthBL + deltaX, 500));
            
            // Adjust width
            floatContainer.style.width = newWidth + 'px';
            
            // Move position to keep right side fixed
            const newLeft = startRectLeft - (newWidth - startWidthBL);
            if (newLeft >= 0) {
                floatContainer.style.left = newLeft + 'px';
            }
            
            // Scale font size proportionally
            const fontScale = startFontScale * (newWidth / startWidthBL);
            updateFontSize(fontScale);
            
            // Prevent event propagation and default behavior
            e.preventDefault();
        }
    });
    
    // Listen for mouse release on entire document
    document.addEventListener('mouseup', function(e) {
        // Handle drag end
        if (isDragging) {
            isDragging = false;
            
            // Restore styles
            floatContainer.style.transition = '';
            dragHandlers.forEach(element => {
                element.style.cursor = 'move';
            });
            
            // Save position
            savePosition();
        }
        
        // Handle resize end
        if (isResizingBR || isResizingBL) {
            isResizingBR = false;
            isResizingBL = false;
            
            // Save size
            saveSize();
        }
    });
    
    // Add protection to avoid dragging out of window
    document.addEventListener('mouseleave', function() {
        if (isDragging) {
            isDragging = false;
            dragHandlers.forEach(element => {
                element.style.cursor = 'move';
            });
            floatContainer.style.transition = '';
            savePosition();
        }
        
        if (isResizingBR || isResizingBL) {
            isResizingBR = false;
            isResizingBL = false;
            saveSize();
        }
    });
    
    // Listen for window resize to ensure floating window is visible
    window.addEventListener('resize', function() {
        // Get current position
        const rect = floatContainer.getBoundingClientRect();
        
        // Check if floating window partially exceeds viewport
        if (rect.right > window.innerWidth) {
            // If right side exceeds, reposition to right boundary
            floatContainer.style.left = (window.innerWidth - rect.width) + 'px';
        }
        
        if (rect.bottom > window.innerHeight) {
            // If bottom exceeds, reposition to bottom boundary
            floatContainer.style.top = (window.innerHeight - rect.height) + 'px';
        }
        
        // Save new position
        savePosition();
    });
    
    // Ensure floating window is always on top when clicked
    floatContainer.addEventListener('mousedown', function() {
        this.style.zIndex = '1001';
    });
    
    // Call load position and size function
    loadPositionAndSize();
    
    // Store active GPU indices and monitor elements
    const monitorElements = {
        container: monitorContainer,
        systemGroup: systemGroup,
        gpuGroups: {},
        cpu: cpuElement,
        ram: ramElement,
        gpus: [],
        temps: [],
        vrams: [],
        activeGpuIndices: [],
        cmdGpuIndices: [],
        gpuDevices: {},
        lastGpuIds: [],
        currentFontScale: 1,
        hasGpuDivider: false // Add flag to prevent duplicate dividers
    };
    
    // Get active GPU information
    fetchSystemStats(monitorElements);
    
    // Set up WebSocket listener
    setupWebSocketListener(monitorElements);
    
    // Opacity toggle
    opacityBtn.addEventListener('click', function() {
        floatContainer.classList.toggle('monitor-opacity-low');
        floatContainer.classList.toggle('monitor-opacity-high');
    });
    
    // Collapse/expand functionality
    let isCollapsed = false;
    collapseBtn.addEventListener('click', function() {
        isCollapsed = !isCollapsed;
        content.style.display = isCollapsed ? 'none' : 'block';
        collapseBtn.innerHTML = isCollapsed ? '+' : '−';
    });
    
    // Apply initial configuration
    setTimeout(() => {
        applyOpacity();
        applyAllProgressColors();
        applyBackgroundImage();
        updateHistoryColors();
    }, 100);
}

// Create monitor element
function createMonitorElement(label, value, progressClass) {
    const container = document.createElement('div');
    container.className = 'monitor-item';
    
    const labelEl = document.createElement('span');
    labelEl.className = 'monitor-label';
    labelEl.textContent = label;
    
    const barEl = document.createElement('div');
    barEl.className = 'monitor-bar';
    
    const progressEl = document.createElement('div');
    progressEl.className = `monitor-progress ${progressClass || ''}`;
    
    // Apply currently configured progress bar color
    const savedConfig = localStorage.getItem('hardwareMonitorConfig');
    if (savedConfig) {
        try {
            const config = JSON.parse(savedConfig);
            const adjustColorBrightness = (color, amount) => {
                const num = parseInt(color.replace("#", ""), 16);
                const amt = Math.round(2.55 * amount);
                const R = (num >> 16) + amt;
                const G = (num >> 8 & 0x00FF) + amt;
                const B = (num & 0x0000FF) + amt;
                return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
                    (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 + (B < 255 ? B < 1 ? 0 : B : 255))
                    .toString(16).slice(1);
            };
            
            // Determine which color to use based on progressClass
            let color = config.progressColor; // Default color
            if (config.individualColors) {
                if (progressClass.includes('cpu')) {
                    color = config.individualColors.cpu;
                } else if (progressClass.includes('ram')) {
                    color = config.individualColors.ram;
                } else if (progressClass.includes('gpu')) {
                    color = config.individualColors.gpu;
                } else if (progressClass.includes('temp')) {
                    color = config.individualColors.temp;
                } else if (progressClass.includes('vram')) {
                    color = config.individualColors.vram;
                }
            }
            
            progressEl.style.background = `linear-gradient(90deg, ${color} 0%, ${adjustColorBrightness(color, -20)} 100%)`;
        } catch (e) {
            // Use default color if parsing fails
        }
    }
    
    barEl.appendChild(progressEl);
    
    const valueEl = document.createElement('span');
    valueEl.className = 'monitor-value';
    valueEl.textContent = value;
    
    container.appendChild(barEl);
    container.appendChild(labelEl);
    container.appendChild(valueEl);
    
    // Get warning color and style based on percentage
    function getWarningStyle(percent, element) {
        if (percent >= 80) {
            // High load red warning
            element.style.color = '#ff4d4f';  // Bright red
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(255, 77, 79, 0.8)';
            element.style.fontWeight = 'bold';
        } else if (percent >= 60) {
            // Medium load yellow warning
            element.style.color = '#fadb14';  // Bright yellow, brighter than before
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(250, 219, 20, 0.8)';
            element.style.fontWeight = 'bold';
        } else {
            // Normal state
            element.style.color = '#fff';
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.8)';
            element.style.fontWeight = 'normal';
        }
    }
    
    // Temperature warning style
    function getTempWarningStyle(temp, element) {
        if (temp >= 85) {
            // High temperature red warning
            element.style.color = '#ff4d4f';  // Bright red
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(255, 77, 79, 0.8)';
            element.style.fontWeight = 'bold';
        } else if (temp >= 75) {
            // Medium temperature yellow warning
            element.style.color = '#fadb14';  // Bright yellow, brighter than before
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(250, 219, 20, 0.8)';
            element.style.fontWeight = 'bold';
        } else {
            // Normal temperature
            element.style.color = '#fff';
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.8)';
            element.style.fontWeight = 'normal';
        }
    }
    
    return {
        container,
        label: labelEl,
        bar: barEl,
        value: valueEl,
        progress: progressEl,
        update: function(newValue, percent, used, total) {
            this.value.textContent = newValue;
            
            if (percent !== undefined) {
                this.progress.style.width = `${percent}%`;
                
                // Add warning color
                if (!label.includes('Temp') && !label.includes('T')) {
                    // For CPU, memory and GPU load, set color based on percentage
                    getWarningStyle(percent, this.value);
                    getWarningStyle(percent, this.label);
                }
            }
            
            // Special handling for VRAM display
            if (label.includes('VRAM')) {
                if (used !== undefined && total !== undefined) {
                    // Display actual usage (e.g. "8.2GB/12.0GB")
                    const usedStr = formatBytes(used);
                    const totalStr = formatBytes(total);
                    this.value.textContent = `${usedStr}/${totalStr}`;

                    // Keep progress bar percentage
                    this.progress.style.width = `${percent}%`;

                    // Set tooltip to show percentage
                    this.container.title = `${label}: ${percent}% (${usedStr}/${totalStr})`;

                    // Add warning color
                    getWarningStyle(percent, this.value);
                    getWarningStyle(percent, this.label);
                }
            }
            // Keep other monitor items as is
            else {
                this.value.textContent = newValue;
                if (percent !== undefined) {
                    this.progress.style.width = `${percent}%`;
                    getWarningStyle(percent, this.value);
                    getWarningStyle(percent, this.label);
                }
            }
            // Temperature special handling - color changes with temperature
            if (label.includes('T') || label.includes('Temp')) {
                const tempValue = parseFloat(newValue);
                
                // Color gradient from green to red, green below 60°, red above 90°
                const hue = Math.max(0, 120 - (tempValue - 60) * 2);
                this.progress.style.backgroundColor = `hsla(${hue}, 80%, 50%, 0.7)`;
                
                // Temperature text color also changes with temperature
                getTempWarningStyle(tempValue, this.value);
                getTempWarningStyle(tempValue, this.label);
            }
        }
    };
}

// Set up WebSocket listener
function setupWebSocketListener(monitorElements) {
    // Listen to existing WebSocket connections
    const originalWebSocket = window.WebSocket;
    window.WebSocket = function(url, protocols) {
        const socket = new originalWebSocket(url, protocols);
        
        // Listen for messages
        socket.addEventListener('message', function(event) {
            try {
                const message = JSON.parse(event.data);
                
                // Monitor data - support multiple data types
                if (message.type === "crystools.monitor" || message.type === "hardware.monitor") {
                    updateMonitorUI(monitorElements, message.data);
                }
            } catch (error) {
                // Ignore non-JSON messages
            }
        });
        
        return socket;
    };
    
    // For existing WebSocket connections
    if (window.app && window.app.socketio) {
        const originalOnMessage = window.app.socketio.onmessage;
        window.app.socketio.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                
                // Monitor data - support multiple data types
                if (message.type === "crystools.monitor" || message.type === "hardware.monitor") {
                    updateMonitorUI(monitorElements, message.data);
                }
            } catch (error) {
                // Ignore non-JSON messages
            }
            
            // Call original handler
            if (originalOnMessage) {
                originalOnMessage.call(this, event);
            }
        };
    }
}

// Get system information to determine active GPUs
async function fetchSystemStats(monitorElements) {
    try {
        // Get current page host address
        const currentUrl = window.location.href;
        const url = new URL(currentUrl);
        const baseUrl = `${url.protocol}//${url.host}`;
        
        // Build API URL
        const apiUrl = `${baseUrl}/api/system_stats`;
        
        const response = await fetch(apiUrl);
        if (response.ok) {
            const data = await response.json();
            
            // Store active GPU indices and device information
            monitorElements.activeGpuIndices = [];
            monitorElements.cmdGpuIndices = [];
            monitorElements.gpuDevices = {};
            
            // Parse device information, store GPU model
            if (data.devices && data.devices.length > 0) {
                data.devices.forEach(device => {
                    if (device.type === 'cuda') {
                        // Extract GPU model name
                        let gpuName = device.name || '';
                        if (gpuName.includes('NVIDIA')) {
                            // Extract model part from full name
                            const match = gpuName.match(/NVIDIA\s+(GeForce\s+RTX\s+\w+|Tesla\s+\w+|Quadro\s+\w+|RTX\s+\w+|GTX\s+\w+)/i);
                            if (match && match[1]) {
                                gpuName = match[1];
                            }
                        }
                        
                        // Store device information
                        monitorElements.gpuDevices[device.index] = {
                            name: gpuName,
                            vram_total: device.vram_total,
                            index: device.index
                        };
                    }
                });
                console.log("Hardware Monitor Plugin - GPU devices info:", monitorElements.gpuDevices);
            }
            
            // Check CUDA devices in command line arguments
            if (data.system && data.system.argv) {
                const cudaDeviceArg = data.system.argv.indexOf('--cuda-device');
                if (cudaDeviceArg !== -1 && cudaDeviceArg + 1 < data.system.argv.length) {
                    const deviceArg = data.system.argv[cudaDeviceArg + 1];
                    // Parse device parameter (might be single number or comma-separated list)
                    const devices = deviceArg.split(',').map(d => parseInt(d.trim()));
                    
                    // Store command line specified GPU indices
                    monitorElements.cmdGpuIndices = devices;
                    
                    // Add hint information for command line specified devices
                    devices.forEach(cmdIndex => {
                        // Try to find corresponding actual device from device list
                        for (const deviceIndex in monitorElements.gpuDevices) {
                            if (parseInt(deviceIndex) === cmdIndex) {
                                return; // Already exists, no need to add
                            }
                        }
                        
                        // If corresponding actual device not found, add an assumed device
                        // This mainly handles cases where command line specifies device but API doesn't return it
                        monitorElements.gpuDevices[cmdIndex] = {
                            name: "Unknown GPU",
                            index: cmdIndex
                        };
                    });
                }
            }
        }
    } catch (error) {
        // Silently handle error
    }
}

// Update monitor UI
function updateMonitorUI(monitorElements, data) {
    if (!monitorElements || !monitorElements.container) return;

    // Ensure floating window is visible
    const floatContainer = document.getElementById('hardware-monitor-float');
    if (floatContainer && floatContainer.style.display === 'none') {
        floatContainer.style.display = 'block';
    }

    // Update CPU usage
    if (data.cpu_utilization !== undefined) {
        const cpuPercent = Math.round(data.cpu_utilization);
        monitorElements.cpu.update(`${cpuPercent}%`, cpuPercent);
    }

    // Update RAM usage
    if (data.ram_used_percent !== undefined) {
        const ramPercent = Math.round(data.ram_used_percent);
        monitorElements.ram.update(`${ramPercent}%`, ramPercent, data.ram_used, data.ram_total);
    }

    // Update GPU information
    // Ensure gpus field always exists
    const gpus = data.gpus || [];

    // Show divider (only once when there's GPU data)
    const divider = document.getElementById('gpu-divider');
    if (gpus.length > 0 && divider) {
        divider.style.display = 'block';
    }

    // Determine which GPUs to show
    let gpuIndicesToShow = [];
    const cmdLineSpecifiedGpus = gpus.filter(gpu => gpu.is_cmdline_specified);

    // Prioritize command line specified GPUs
    if (cmdLineSpecifiedGpus.length > 0) {
        gpuIndicesToShow = cmdLineSpecifiedGpus.map(gpu => gpu.index);
    } else {
        // Otherwise show all GPUs
        gpuIndicesToShow = gpus.map(gpu => gpu.index);
    }

    // Process each GPU to show
    gpuIndicesToShow.forEach((gpuIndex, idx) => {
        // Ensure gpuIndex is not undefined
        if (gpuIndex === undefined || gpuIndex === null) {
            return;
        }
        // Find corresponding GPU data from gpus
        const gpuData = gpus.find(gpu => gpu.index === gpuIndex);

        // Format labels
        const gpuLabel = `GPU ${gpuIndex}`;
        const vramLabel = `VRAM`;
        const tempLabel = `Temp ${gpuIndex}`;

        // Prepare tooltip information
        const gpuName = (gpuData && gpuData.name) || `GPU ${gpuIndex}`;
        const gpuTooltip = `GPU ${gpuIndex}: ${gpuName}`;

        // Ensure corresponding GPU group has been created
        if (!monitorElements.gpuGroups[gpuIndex]) {
            // Create new GPU group
            const gpuGroup = document.createElement('div');
            gpuGroup.className = 'monitor-group gpu-group';

            // Add GPU group title
            const gpuTitle = document.createElement('div');
            gpuTitle.className = 'monitor-group-title';
            gpuTitle.textContent = `GPU ${gpuIndex}: ${gpuName.split(' ').pop() || `GPU${gpuIndex}`}`;
            gpuGroup.appendChild(gpuTitle);

            // Insert GPU group before auto cleanup button
            const cleanupButton = monitorElements.container.querySelector('.monitor-cleanup-button');
            if (cleanupButton) {
                monitorElements.container.insertBefore(gpuGroup, cleanupButton);
            } else {
                monitorElements.container.appendChild(gpuGroup);
            }

            // Create GPU indicator
            const gpuEl = createMonitorElement(gpuLabel, '0%', 'gpu-progress');
            gpuEl.container.title = gpuTooltip;
            gpuGroup.appendChild(gpuEl.container);

            // Create VRAM indicator
            const vramEl = createMonitorElement(vramLabel, '0', 'vram-progress');
            vramEl.container.title = `${gpuTooltip} - VRAM`;
            gpuGroup.appendChild(vramEl.container);

            // Create temperature indicator
            const tempEl = createMonitorElement(tempLabel, '0°', 'temp-progress');
            tempEl.container.title = `${gpuTooltip} - Temperature`;
            gpuGroup.appendChild(tempEl.container);

            monitorElements.gpuGroups[gpuIndex] = {
                element: gpuGroup,
                index: gpuIndex,
                gpuEl: gpuEl,
                vramEl: vramEl,
                tempEl: tempEl
            };

            // Apply current font scaling
            if (monitorElements.currentFontScale && monitorElements.currentFontScale !== 1) {
                const scale = monitorElements.currentFontScale;
                gpuTitle.style.fontSize = `${13 * scale}px`;
                [gpuEl, vramEl, tempEl].forEach(item => {
                    if (item && item.container) {
                        item.container.style.height = `${26 * scale}px`;
                        if (item.label) item.label.style.fontSize = `${12 * scale}px`;
                        if (item.value) item.value.style.fontSize = `${12 * scale}px`;
                    }
                });
            }
        }

        // Get monitor elements for current GPU group
        const currentGroup = monitorElements.gpuGroups[gpuIndex];

        // Update GPU utilization
        const gpuPercent = gpuData ? Math.round(gpuData.gpu_utilization) : 0;
        if (currentGroup.gpuEl) {
            currentGroup.gpuEl.update(`${gpuPercent}%`, gpuPercent);
            currentGroup.gpuEl.container.title = `${gpuTooltip} - Usage: ${gpuPercent}%`;
        }

        // Update VRAM usage
        const vramPercent = gpuData ? Math.round(gpuData.vram_used_percent) : 0;
        const vramUsed = gpuData ? gpuData.vram_used : 0;
        const vramTotal = gpuData ? gpuData.vram_total : 0;
        if (currentGroup.vramEl) {
            currentGroup.vramEl.update(``, vramPercent, vramUsed, vramTotal);
        }

        // Update temperature
        const tempValue = gpuData ? Math.round(gpuData.gpu_temperature) : 0;
        if (currentGroup.tempEl) {
            currentGroup.tempEl.update(`${tempValue}°C`, tempValue);
            currentGroup.tempEl.container.title = `${gpuTooltip} - Temperature: ${tempValue}°C`;
        }
    });
}
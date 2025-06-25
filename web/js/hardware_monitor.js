/**
 * Hardware Monitor Plugin for ComfyUI
 * 可拖动悬浮窗显示硬件监控信息，带进度条显示
 * 支持自定义透明度、颜色、背景和配置保存
 */

console.log("Hardware Monitor Plugin - Script loaded");

// 直接开始检查DOM
(function() {
    // 添加必要的CSS样式
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
        
        /* 悬浮窗样式 */
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
        
        /* 悬浮窗头部样式 */
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
        
        /* 设置按钮样式 */
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
        
        /* 设置面板样式 */
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
        
        /* 滚动条样式 */
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
        
        /* 文件输入优化 */
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
        
        /* 动画效果 */
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
        
        /* 响应式设计 */
        @media (max-width: 500px) {
            .monitor-settings-panel {
                width: 90%;
                max-width: 400px;
            }
        }
        
        /* 悬浮窗内容区 */
        .monitor-content {
            padding: 5px;
        }
        
        /* 透明度控制 */
        .monitor-opacity-high {
            opacity: 1;
            transition: opacity 0.2s;
        }
        
        .monitor-opacity-low {
            opacity: 0.5;
            transition: opacity 0.2s;
        }
        
        /* 分隔线 */
        .monitor-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.2);
            margin: 4px 0;
            width: 100%;
        }
        
        /* 清理按钮 */
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
        
        /* 给不同类型的监控项添加不同的背景色 */
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
        
        /* 调整大小控制柄样式 */
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
    
    // 创建悬浮窗
    setupHardwareMonitorFloating();
})();

// 格式化字节数为可读格式
function formatBytes(bytes) {
    if (bytes === 0) return '0B';
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)}${sizes[i]}`;
}

// 创建悬浮窗监控器
function setupHardwareMonitorFloating() {
    console.log("Hardware Monitor Plugin - Creating floating window");
    
    // 创建悬浮窗容器
    const floatContainer = document.createElement('div');
    floatContainer.className = 'hardware-monitor-float monitor-opacity-high';
    floatContainer.id = 'hardware-monitor-float';
    
    // 创建悬浮窗头部
    const header = document.createElement('div');
    header.className = 'monitor-header';
    
    // 头部标题
    const title = document.createElement('div');
    title.className = 'monitor-header-title';
    title.textContent = '系统监控';
    header.appendChild(title);
    
    // 头部控制按钮
    const controls = document.createElement('div');
    controls.className = 'monitor-header-controls';
    
    // 设置按钮
    const settingsBtn = document.createElement('div');
    settingsBtn.className = 'monitor-settings-btn';
    settingsBtn.innerHTML = '⚙';
    settingsBtn.title = '设置';
    controls.appendChild(settingsBtn);
    
    // 透明度切换按钮
    const opacityBtn = document.createElement('div');
    opacityBtn.className = 'monitor-control-btn';
    opacityBtn.innerHTML = '⊙';
    opacityBtn.title = '切换透明度';
    controls.appendChild(opacityBtn);
    
    // 添加折叠/展开按钮
    const collapseBtn = document.createElement('div');
    collapseBtn.className = 'monitor-control-btn';
    collapseBtn.innerHTML = '−';
    collapseBtn.title = '折叠/展开';
    controls.appendChild(collapseBtn);
    
    header.appendChild(controls);
    floatContainer.appendChild(header);
    
    // 创建内容区域
    const content = document.createElement('div');
    content.className = 'monitor-content';
    floatContainer.appendChild(content);
    
    // 创建硬件监控容器
    const monitorContainer = document.createElement('div');
    monitorContainer.className = 'hardware-monitor-container';
    content.appendChild(monitorContainer);
    
    // 创建系统资源组（CPU/RAM）
    const systemGroup = document.createElement('div');
    systemGroup.className = 'monitor-group system-group';
    
    // 添加系统组标题
    const systemTitle = document.createElement('div');
    systemTitle.className = 'monitor-group-title';
    systemTitle.textContent = '系统资源';
    systemGroup.appendChild(systemTitle);
    
    monitorContainer.appendChild(systemGroup);
    
    // 初始化显示的信息
    const cpuElement = createMonitorElement('CPU', '0%', 'cpu-progress');
    const ramElement = createMonitorElement('内存', '0%', 'ram-progress');
    
    // 添加到系统资源组
    systemGroup.appendChild(cpuElement.container);
    systemGroup.appendChild(ramElement.container);
    
    // 添加分隔线
    const divider = document.createElement('div');
    divider.className = 'monitor-divider';
    monitorContainer.appendChild(divider);
    
    // 添加自动清理按钮
    const cleanupButton = document.createElement('div');
    cleanupButton.className = 'monitor-cleanup-button';
    
    const cleanupText = document.createElement('span');
    cleanupText.className = 'cleanup-text';
    cleanupText.textContent = '自动清理显存';
    cleanupButton.appendChild(cleanupText);
    
    const indicator = document.createElement('div');
    indicator.className = 'cleanup-indicator';
    cleanupButton.appendChild(indicator);
    
    monitorContainer.appendChild(cleanupButton);
    
    // 添加调整大小控制柄
    const resizeHandleBR = document.createElement('div');
    resizeHandleBR.className = 'resize-handle resize-handle-br';
    floatContainer.appendChild(resizeHandleBR);
    
    const resizeHandleBL = document.createElement('div');
    resizeHandleBL.className = 'resize-handle resize-handle-bl';
    floatContainer.appendChild(resizeHandleBL);
    
    // 添加到页面
    document.body.appendChild(floatContainer);
    
    // 创建提示工具提示
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
    tooltip.textContent = '最近清理时间: 无';
    document.body.appendChild(tooltip);
    
    // 鼠标悬停显示提示
    cleanupButton.addEventListener('mouseenter', (e) => {
        tooltip.style.display = 'block';
        updateTooltipPosition(e);
    });
    
    cleanupButton.addEventListener('mousemove', updateTooltipPosition);
    
    cleanupButton.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
    
    // 更新提示位置
    function updateTooltipPosition(e) {
        const rect = cleanupButton.getBoundingClientRect();
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.bottom + 5}px`;
    }
    
    // 初始化自动清理变量
    let isActive = false;
    let intervalId = null;
    let lastCleanupTime = null;
    
    // 配置变量
    let config = {
        opacity: 0.8,
        progressColor: '#4CAF50', // 保留作为默认颜色
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
        }
    };
    
    // 创建设置面板
    function createSettingsPanel() {
        const panel = document.createElement('div');
        panel.className = 'monitor-settings-panel';
        panel.id = 'monitor-settings-panel';
        panel.style.display = 'none';
        
        panel.innerHTML = `
            <div class="settings-panel-header">
                <div class="settings-panel-title">硬件监控设置</div>
                <div class="settings-close-btn">×</div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">外观设置</div>
                
                <div class="settings-item">
                    <label class="settings-item-label">透明度: <span class="settings-value-display" id="opacity-value">${Math.round(config.opacity * 100)}%</span></label>
                    <input type="range" class="settings-slider" id="opacity-slider" min="0.1" max="1" step="0.1" value="${config.opacity}">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">背景图片:</label>
                    <input type="file" class="settings-file-input" id="background-image" accept="image/*">
                    <button class="settings-button secondary" id="remove-background">移除背景</button>
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">背景图片透明度: <span class="settings-value-display" id="bg-opacity-value">${Math.round(config.backgroundImageOpacity * 100)}%</span></label>
                    <input type="range" class="settings-slider" id="bg-opacity-slider" min="0.1" max="1" step="0.1" value="${config.backgroundImageOpacity}">
                </div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">监控条颜色设置</div>
                
                <div class="settings-item">
                    <label class="settings-item-label">CPU 使用率:</label>
                    <input type="color" class="settings-color-input individual-color" id="cpu-color" value="${config.individualColors.cpu}" data-type="cpu">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">内存使用率:</label>
                    <input type="color" class="settings-color-input individual-color" id="ram-color" value="${config.individualColors.ram}" data-type="ram">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">GPU 使用率:</label>
                    <input type="color" class="settings-color-input individual-color" id="gpu-color" value="${config.individualColors.gpu}" data-type="gpu">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">GPU 温度:</label>
                    <input type="color" class="settings-color-input individual-color" id="temp-color" value="${config.individualColors.temp}" data-type="temp">
                </div>
                
                <div class="settings-item">
                    <label class="settings-item-label">显存使用率:</label>
                    <input type="color" class="settings-color-input individual-color" id="vram-color" value="${config.individualColors.vram}" data-type="vram">
                </div>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">快速配色</div>
                <div style="font-size: 11px; color: #aaa; margin-bottom: 10px;">点击颜色块应用到最后选择的监控条</div>
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
                <div class="settings-group-title">历史配色</div>
                <div class="history-colors" id="history-colors"></div>
                <button class="settings-button secondary" id="clear-history">清除历史</button>
            </div>
            
            <div class="settings-group">
                <div class="settings-group-title">配置管理</div>
                <button class="settings-button" id="save-config">保存配置</button>
                <button class="settings-button secondary" id="reset-config">重置为默认</button>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // 绑定事件
        bindSettingsEvents(panel);
        
        return panel;
    }
    
    // 绑定设置面板事件
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
        
        // 关闭面板
        closeBtn.addEventListener('click', () => {
            panel.style.display = 'none';
        });
        
        // 点击面板外部关闭面板
        panel.addEventListener('click', (e) => {
            if (e.target === panel) {
                panel.style.display = 'none';
            }
        });
        
        // ESC键关闭面板
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && panel.style.display === 'block') {
                panel.style.display = 'none';
            }
        });
        
        // 透明度调整
        opacitySlider.addEventListener('input', (e) => {
            config.opacity = parseFloat(e.target.value);
            opacityValue.textContent = Math.round(config.opacity * 100) + '%';
            applyOpacity();
            saveConfig();
        });
        
        // 背景图片透明度调整
        bgOpacitySlider.addEventListener('input', (e) => {
            config.backgroundImageOpacity = parseFloat(e.target.value);
            bgOpacityValue.textContent = Math.round(config.backgroundImageOpacity * 100) + '%';
            applyBackgroundImage();
            saveConfig();
        });
        
        // 单独颜色调整
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
        
        // 背景图片上传
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
        
        // 移除背景
        removeBackgroundBtn.addEventListener('click', () => {
            config.backgroundImage = null;
            applyBackgroundImage();
            backgroundImageInput.value = '';
            saveConfig();
        });
        
        // 预设颜色选择 - 应用到最后点击的颜色输入框
        let lastClickedColorInput = null;
        
        individualColorInputs.forEach(input => {
            input.addEventListener('focus', () => {
                lastClickedColorInput = input;
            });
        });
        
        colorPresets.forEach(preset => {
            preset.addEventListener('click', () => {
                const color = preset.dataset.color;
                
                // 如果有最后点击的颜色输入框，则应用到该输入框
                if (lastClickedColorInput) {
                    const type = lastClickedColorInput.dataset.type;
                    config.individualColors[type] = color;
                    lastClickedColorInput.value = color;
                    applyIndividualColor(type, color);
                } else {
                    // 默认应用到CPU颜色
                    config.individualColors.cpu = color;
                    panel.querySelector('#cpu-color').value = color;
                    applyIndividualColor('cpu', color);
                }
                
                addToColorHistory(color);
                saveConfig();
                
                // 更新选中状态
                colorPresets.forEach(p => p.classList.remove('selected'));
                preset.classList.add('selected');
            });
        });
        
        // 保存配置
        saveConfigBtn.addEventListener('click', () => {
            saveConfig();
            alert('配置已保存！');
        });
        
        // 重置配置
        resetConfigBtn.addEventListener('click', () => {
            if (confirm('确定要重置为默认配置吗？')) {
                resetConfig();
                updateSettingsPanel();
            }
        });
        
        // 清除历史
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('确定要清除颜色历史吗？')) {
                config.colorHistory = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'];
                updateHistoryColors();
                saveConfig();
            }
        });
    }
    
    // 应用透明度
    function applyOpacity() {
        floatContainer.style.opacity = config.opacity;
    }
    
    // 应用单独的进度条颜色
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
    
    // 应用所有进度条颜色
    function applyAllProgressColors() {
        Object.keys(config.individualColors).forEach(type => {
            applyIndividualColor(type, config.individualColors[type]);
        });
    }
    
    // 应用背景图片
    function applyBackgroundImage() {
        const container = floatContainer.querySelector('.hardware-monitor-container');
        if (config.backgroundImage) {
            container.style.backgroundImage = `url(${config.backgroundImage})`;
            // 根据背景图片透明度设置背景色
            const alpha = 1 - config.backgroundImageOpacity;
            container.style.backgroundColor = `rgba(30, 30, 30, ${alpha})`;
            
            // 如果有背景图片，添加一个半透明覆盖层来控制图片透明度
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
            // 移除覆盖层
            const overlay = container.querySelector('.bg-overlay');
            if (overlay) {
                overlay.remove();
            }
        }
    }
    
    // 颜色亮度调整函数
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
    
    // 添加到颜色历史
    function addToColorHistory(color) {
        if (!config.colorHistory.includes(color)) {
            config.colorHistory.unshift(color);
            if (config.colorHistory.length > 10) {
                config.colorHistory = config.colorHistory.slice(0, 10);
            }
            updateHistoryColors();
        }
    }
    
    // 更新历史颜色显示
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
                    // 应用到最后点击的颜色输入框
                    if (lastClickedColorInput) {
                        const type = lastClickedColorInput.dataset.type;
                        config.individualColors[type] = color;
                        lastClickedColorInput.value = color;
                        applyIndividualColor(type, color);
                    } else {
                        // 默认应用到CPU颜色
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
    
    // 更新设置面板
    function updateSettingsPanel() {
        const panel = document.querySelector('#monitor-settings-panel');
        if (panel) {
            panel.querySelector('#opacity-slider').value = config.opacity;
            panel.querySelector('#opacity-value').textContent = Math.round(config.opacity * 100) + '%';
            panel.querySelector('#bg-opacity-slider').value = config.backgroundImageOpacity;
            panel.querySelector('#bg-opacity-value').textContent = Math.round(config.backgroundImageOpacity * 100) + '%';
            
            // 更新单独颜色输入框
            Object.keys(config.individualColors).forEach(type => {
                const input = panel.querySelector(`#${type}-color`);
                if (input) {
                    input.value = config.individualColors[type];
                }
            });
            
            updateHistoryColors();
        }
    }
    
    // 保存配置
    function saveConfig() {
        try {
            localStorage.setItem('hardwareMonitorConfig', JSON.stringify(config));
        } catch (e) {
            console.error('Error saving config:', e);
        }
    }
    
    // 加载配置
    function loadConfig() {
        try {
            const savedConfig = localStorage.getItem('hardwareMonitorConfig');
            if (savedConfig) {
                const parsed = JSON.parse(savedConfig);
                // 合并配置，确保新的配置项有默认值
                config = { 
                    ...config, 
                    ...parsed,
                    // 确保individualColors存在
                    individualColors: {
                        ...config.individualColors,
                        ...parsed.individualColors
                    }
                };
                
                // 确保backgroundImageOpacity存在
                if (typeof config.backgroundImageOpacity === 'undefined') {
                    config.backgroundImageOpacity = 0.8;
                }
            }
        } catch (e) {
            console.error('Error loading config:', e);
        }
    }
    
    // 重置配置
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
            }
        };
        applyOpacity();
        applyAllProgressColors();
        applyBackgroundImage();
        saveConfig();
    }
    
    // 初始化设置
    loadConfig();
    const settingsPanel = createSettingsPanel();
    
    // 设置按钮点击事件
    settingsBtn.addEventListener('click', () => {
        settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
        updateSettingsPanel();
    });
    
    // 从localStorage加载状态
    function loadState() {
        try {
            const savedState = localStorage.getItem('autoCleanupState');
            if (savedState !== null) {
                isActive = JSON.parse(savedState);
                console.log(`Auto Cleanup Plugin - Loaded state: ${isActive}`);
                
                // 如果状态为活跃，启动清理
                if (isActive) {
                    updateButtonState();
                }
            }
        } catch (e) {
            console.error('Auto Cleanup Plugin - Error loading state:', e);
        }
        
        // 加载上次清理时间
        try {
            const savedTime = localStorage.getItem('autoCleanupLastTime');
            if (savedTime !== null) {
                lastCleanupTime = new Date(savedTime);
                updateTooltip();
                console.log(`Auto Cleanup Plugin - Loaded last cleanup time: ${lastCleanupTime}`);
            }
        } catch (e) {
            console.error('Auto Cleanup Plugin - Error loading last cleanup time:', e);
        }
    }
    
    // 保存状态到localStorage
    function saveState() {
        try {
            localStorage.setItem('autoCleanupState', JSON.stringify(isActive));
        } catch (e) {
            console.error('Auto Cleanup Plugin - Error saving state:', e);
        }
    }
    
    // 保存上次清理时间
    function saveLastCleanupTime() {
        if (lastCleanupTime) {
            try {
                localStorage.setItem('autoCleanupLastTime', lastCleanupTime.toISOString());
            } catch (e) {
                console.error('Auto Cleanup Plugin - Error saving last cleanup time:', e);
            }
        }
    }
    
    // 添加清理按钮点击事件
    cleanupButton.addEventListener('click', () => {
        isActive = !isActive;
        updateButtonState();
        saveState();
    });
    
    // 更新按钮状态
    function updateButtonState() {
        if (isActive) {
            indicator.style.backgroundColor = '#4CAF50'; // 绿色表示开启
            
            // 启动定时清理
            if (!intervalId) {
                performCleanup(); // 立即执行一次
                intervalId = setInterval(performCleanup, 5000); // 每5秒执行一次
            }
        } else {
            indicator.style.backgroundColor = '#888'; // 灰色表示关闭
            
            // 停止定时清理
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
        }
    }
    
    // 执行清理操作
    async function performCleanup() {
        // 获取当前页面的主机地址
        const currentUrl = window.location.href;
        const url = new URL(currentUrl);
        const baseUrl = `${url.protocol}//${url.host}`;
        
        // 构建API URL
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
            } else {
                console.error(`Auto Cleanup Plugin - Cleanup failed: ${response.status}`);
            }
        } catch (error) {
            console.error(`Auto Cleanup Plugin - Cleanup request error: ${error}`);
        }
    }
    
    // 更新提示信息
    function updateTooltip() {
        if (lastCleanupTime) {
            const timeString = lastCleanupTime.toLocaleTimeString();
            tooltip.textContent = `最近清理时间: ${timeString}`;
        } else {
            tooltip.textContent = "最近清理时间: 无";
        }
    }
    
    // 加载保存的状态
    loadState();
    
    // 设置拖动和调整大小相关变量
    let isDragging = false;
    let startX, startY;
    let startPosX, startPosY;
    let isResizingBR = false;
    let isResizingBL = false;
    let startWidthBR, startFontScale;
    let startWidthBL, startRectLeft;
    
    // 从localStorage加载位置和大小
    function loadPositionAndSize() {
        try {
            // 延迟执行确保DOM完全加载和渲染
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
                    // 确保使用像素单位
                    if (!size.width.endsWith('px')) {
                        size.width += 'px';
                    }
                    floatContainer.style.width = size.width;
                    
                    // 更新所有子元素的字体大小
                    const fontScale = size.fontScale || 1;
                    // 将当前的字体大小缩放比例保存到monitorElements中，供后续GPU项使用
                    monitorElements.currentFontScale = fontScale;
                    updateFontSize(fontScale);
                    
                    // 强制重新布局
                    void floatContainer.offsetWidth;
                }
            }, 100);
        } catch (e) {
            console.error("Error loading saved position or size:", e);
        }
    }
    
    // 保存位置和大小到localStorage
    function savePosition() {
        const position = {
            left: floatContainer.style.left,
            top: floatContainer.style.top
        };
        localStorage.setItem('hardware-monitor-position', JSON.stringify(position));
    }
    
    function saveSize() {
        // 获取计算后的实际宽度，确保保存的是实际值
        const computedWidth = window.getComputedStyle(floatContainer).width;
        const size = {
            width: computedWidth,
            fontScale: parseFloat(floatContainer.getAttribute('data-font-scale') || '1')
        };
        console.log("Saving size:", size);
        localStorage.setItem('hardware-monitor-size', JSON.stringify(size));
    }
    
    // 更新字体大小
    function updateFontSize(scale) {
        floatContainer.setAttribute('data-font-scale', scale);
        
        // 标题
        const titles = floatContainer.querySelectorAll('.monitor-group-title, .monitor-header-title');
        titles.forEach(el => {
            el.style.fontSize = `${13 * scale}px`;
        });
        
        // 标签和值
        const labels = floatContainer.querySelectorAll('.monitor-label, .monitor-value');
        labels.forEach(el => {
            el.style.fontSize = `${12 * scale}px`;
        });
        
        // 按钮和指示器
        const buttons = floatContainer.querySelectorAll('.monitor-control-btn, .cleanup-text');
        buttons.forEach(el => {
            el.style.fontSize = `${12 * scale}px`;
        });
        
        // 调整监控项高度
        const items = floatContainer.querySelectorAll('.monitor-item');
        items.forEach(el => {
            el.style.height = `${26 * scale}px`;
        });
        
        // 保存当前缩放比例到monitorElements
        if (monitorElements) {
            monitorElements.currentFontScale = scale;
        }
    }
    
    // 监听标题栏和内容区的拖动开始
    const dragHandlers = [header, content];
    dragHandlers.forEach(element => {
        element.addEventListener('mousedown', function(e) {
            // 如果点击的是调整大小的控制柄或按钮，不进行拖动
            if (e.target.closest('.resize-handle') || e.target.closest('.monitor-control-btn')) {
                return;
            }
            
            // 只处理左键点击
            if (e.button !== 0) return;
            
            // 记录鼠标起始坐标
            startX = e.clientX;
            startY = e.clientY;
            
            // 记录窗口起始位置
            const rect = floatContainer.getBoundingClientRect();
            startPosX = rect.left;
            startPosY = rect.top;
            
            // 开始拖动
            isDragging = true;
            
            // 添加临时样式
            floatContainer.style.transition = 'none';
            element.style.cursor = 'grabbing';
            
            // 阻止事件传播和默认行为
            e.preventDefault();
            e.stopPropagation();
        });
    });
    
    // 调整大小功能 - 右下角
    resizeHandleBR.addEventListener('mousedown', function(e) {
        isResizingBR = true;
        startX = e.clientX;
        startWidthBR = parseInt(getComputedStyle(floatContainer).width, 10);
        startFontScale = parseFloat(floatContainer.getAttribute('data-font-scale') || '1');
        
        // 阻止事件传播和默认行为
        e.preventDefault();
        e.stopPropagation();
    });
    
    // 调整大小功能 - 左下角
    resizeHandleBL.addEventListener('mousedown', function(e) {
        isResizingBL = true;
        startX = e.clientX;
        startWidthBL = parseInt(getComputedStyle(floatContainer).width, 10);
        startRectLeft = floatContainer.getBoundingClientRect().left;
        startFontScale = parseFloat(floatContainer.getAttribute('data-font-scale') || '1');
        
        // 阻止事件传播和默认行为
        e.preventDefault();
        e.stopPropagation();
    });
    
    // 在整个文档上监听鼠标移动
    document.addEventListener('mousemove', function(e) {
        // 处理拖动
        if (isDragging) {
            // 计算鼠标移动的距离
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            // 计算新位置
            let newX = startPosX + deltaX;
            let newY = startPosY + deltaY;
            
            // 限制在视窗内
            const maxX = window.innerWidth - floatContainer.offsetWidth;
            const maxY = window.innerHeight - floatContainer.offsetHeight;
            
            newX = Math.max(0, Math.min(newX, maxX));
            newY = Math.max(0, Math.min(newY, maxY));
            
            // 设置新位置
            floatContainer.style.left = newX + 'px';
            floatContainer.style.top = newY + 'px';
            
            // 阻止默认行为
            e.preventDefault();
        }
        
        // 处理右下角调整大小
        if (isResizingBR) {
            const deltaX = e.clientX - startX;
            const newWidth = Math.max(200, Math.min(startWidthBR + deltaX, 500));
            floatContainer.style.width = newWidth + 'px';
            
            // 等比例调整字体大小
            const fontScale = startFontScale * (newWidth / startWidthBR);
            updateFontSize(fontScale);
            
            // 阻止事件传播和默认行为
            e.preventDefault();
        }
        
        // 处理左下角调整大小
        if (isResizingBL) {
            const deltaX = startX - e.clientX;
            const newWidth = Math.max(200, Math.min(startWidthBL + deltaX, 500));
            
            // 调整宽度
            floatContainer.style.width = newWidth + 'px';
            
            // 移动位置以保持右边不变
            const newLeft = startRectLeft - (newWidth - startWidthBL);
            if (newLeft >= 0) {
                floatContainer.style.left = newLeft + 'px';
            }
            
            // 等比例调整字体大小
            const fontScale = startFontScale * (newWidth / startWidthBL);
            updateFontSize(fontScale);
            
            // 阻止事件传播和默认行为
            e.preventDefault();
        }
    });
    
    // 在整个文档上监听鼠标释放
    document.addEventListener('mouseup', function(e) {
        // 处理拖动结束
        if (isDragging) {
            isDragging = false;
            
            // 恢复样式
            floatContainer.style.transition = '';
            dragHandlers.forEach(element => {
                element.style.cursor = 'move';
            });
            
            // 保存位置
            savePosition();
        }
        
        // 处理调整大小结束
        if (isResizingBR || isResizingBL) {
            isResizingBR = false;
            isResizingBL = false;
            
            // 保存大小
            saveSize();
        }
    });
    
    // 添加避免拖出窗口的保护措施
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
    
    // 监听窗口大小变化，确保悬浮窗可见
    window.addEventListener('resize', function() {
        // 获取当前位置
        const rect = floatContainer.getBoundingClientRect();
        
        // 检查悬浮窗是否部分超出视窗
        if (rect.right > window.innerWidth) {
            // 如果右边超出，重新定位到右边界
            floatContainer.style.left = (window.innerWidth - rect.width) + 'px';
        }
        
        if (rect.bottom > window.innerHeight) {
            // 如果底部超出，重新定位到底部边界
            floatContainer.style.top = (window.innerHeight - rect.height) + 'px';
        }
        
        // 保存新位置
        savePosition();
    });
    
    // 确保悬浮窗被点击时始终在最前面
    floatContainer.addEventListener('mousedown', function() {
        this.style.zIndex = '1001';
    });
    
    // 调用加载位置和大小函数
    loadPositionAndSize();
    
    console.log("Hardware Monitor Plugin - Floating window created");
    
    // 存储活跃GPU索引和监控元素
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
        currentFontScale: 1
    };
    
    // 获取活跃GPU信息
    fetchSystemStats(monitorElements);
    
    // 设置WebSocket监听
    setupWebSocketListener(monitorElements);
    
    // 透明度切换
    opacityBtn.addEventListener('click', function() {
        floatContainer.classList.toggle('monitor-opacity-low');
        floatContainer.classList.toggle('monitor-opacity-high');
    });
    
    // 折叠/展开功能
    let isCollapsed = false;
    collapseBtn.addEventListener('click', function() {
        isCollapsed = !isCollapsed;
        content.style.display = isCollapsed ? 'none' : 'block';
        collapseBtn.innerHTML = isCollapsed ? '+' : '−';
    });
    
    // 应用初始配置
    setTimeout(() => {
        applyOpacity();
        applyAllProgressColors();
        applyBackgroundImage();
        updateHistoryColors();
    }, 100);
}

// 创建监控元素
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
    
    // 应用当前配置的进度条颜色
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
            
            // 根据progressClass确定使用哪个颜色
            let color = config.progressColor; // 默认颜色
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
            // 如果解析失败，使用默认颜色
        }
    }
    
    barEl.appendChild(progressEl);
    
    const valueEl = document.createElement('span');
    valueEl.className = 'monitor-value';
    valueEl.textContent = value;
    
    container.appendChild(barEl);
    container.appendChild(labelEl);
    container.appendChild(valueEl);
    
    // 根据百分比获取告警颜色和样式
    function getWarningStyle(percent, element) {
        if (percent >= 80) {
            // 高负载红色告警
            element.style.color = '#ff4d4f';  // 鲜红色
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(255, 77, 79, 0.8)';
            element.style.fontWeight = 'bold';
        } else if (percent >= 60) {
            // 中负载黄色警告
            element.style.color = '#fadb14';  // 亮黄色，比之前的更亮
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(250, 219, 20, 0.8)';
            element.style.fontWeight = 'bold';
        } else {
            // 正常状态
            element.style.color = '#fff';
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.8)';
            element.style.fontWeight = 'normal';
        }
    }
    
    // 温度告警样式
    function getTempWarningStyle(temp, element) {
        if (temp >= 85) {
            // 高温红色告警
            element.style.color = '#ff4d4f';  // 鲜红色
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(255, 77, 79, 0.8)';
            element.style.fontWeight = 'bold';
        } else if (temp >= 75) {
            // 中温黄色警告
            element.style.color = '#fadb14';  // 亮黄色，比之前的更亮
            element.style.textShadow = '0 0 2px rgba(0, 0, 0, 0.9), 0 0 4px rgba(250, 219, 20, 0.8)';
            element.style.fontWeight = 'bold';
        } else {
            // 正常温度
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
                
                // 添加告警颜色
                if (!label.includes('温度') && !label.includes('T')) {
                    // 对于CPU、内存和GPU负载，根据百分比设置颜色
                    getWarningStyle(percent, this.value);
                    getWarningStyle(percent, this.label);
                }
            }
            
            if (used !== undefined && total !== undefined) {
                const usedStr = formatBytes(used);
                const totalStr = formatBytes(total);
                const tooltip = `${label}: ${usedStr}/${totalStr}`;
                this.container.title = tooltip;
            } else {
                this.container.title = `${label}: ${newValue}`;
            }
            
            // 温度特殊处理 - 颜色随温度变化
            if (label.includes('T') || label.includes('温度')) {
                const tempValue = parseFloat(newValue);
                
                // 颜色从绿色渐变到红色，60度以下绿色，90度以上红色
                const hue = Math.max(0, 120 - (tempValue - 60) * 2);
                this.progress.style.backgroundColor = `hsla(${hue}, 80%, 50%, 0.7)`;
                
                // 温度文字颜色也随温度变化
                getTempWarningStyle(tempValue, this.value);
                getTempWarningStyle(tempValue, this.label);
            }
        }
    };
}

// 设置WebSocket监听
function setupWebSocketListener(monitorElements) {
    // 监听已有的WebSocket连接
    const originalWebSocket = window.WebSocket;
    window.WebSocket = function(url, protocols) {
        const socket = new originalWebSocket(url, protocols);
        
        // 监听消息
        socket.addEventListener('message', function(event) {
            try {
                const message = JSON.parse(event.data);
                
                // 监控器数据 - 支持多种数据类型
                if (message.type === "crystools.monitor" || message.type === "hardware.monitor") {
                    updateMonitorUI(monitorElements, message.data);
                }
            } catch (error) {
                // 忽略非JSON消息
            }
        });
        
        return socket;
    };
    
    // 对于已有的WebSocket连接
    if (window.app && window.app.socketio) {
        const originalOnMessage = window.app.socketio.onmessage;
        window.app.socketio.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                
                // 监控器数据 - 支持多种数据类型
                if (message.type === "crystools.monitor" || message.type === "hardware.monitor") {
                    updateMonitorUI(monitorElements, message.data);
                }
            } catch (error) {
                // 忽略非JSON消息
            }
            
            // 调用原有的处理函数
            if (originalOnMessage) {
                originalOnMessage.call(this, event);
            }
        };
    }
    
    console.log("Hardware Monitor Plugin - WebSocket listener setup complete");
}

// 获取系统信息以确定活跃的GPU
async function fetchSystemStats(monitorElements) {
    try {
        // 获取当前页面的主机地址
        const currentUrl = window.location.href;
        const url = new URL(currentUrl);
        const baseUrl = `${url.protocol}//${url.host}`;
        
        // 构建API URL
        const apiUrl = `${baseUrl}/api/system_stats`;
        
        const response = await fetch(apiUrl);
        if (response.ok) {
            const data = await response.json();
            
            // 存储活跃GPU索引和设备信息
            monitorElements.activeGpuIndices = [];
            monitorElements.cmdGpuIndices = [];
            monitorElements.gpuDevices = {};
            
            // 解析设备信息，存储显卡型号
            if (data.devices && data.devices.length > 0) {
                data.devices.forEach(device => {
                    if (device.type === 'cuda') {
                        // 提取显卡型号名称
                        let gpuName = device.name || '';
                        if (gpuName.includes('NVIDIA')) {
                            // 从完整名称中提取型号部分
                            const match = gpuName.match(/NVIDIA\s+(GeForce\s+RTX\s+\w+|Tesla\s+\w+|Quadro\s+\w+|RTX\s+\w+|GTX\s+\w+)/i);
                            if (match && match[1]) {
                                gpuName = match[1];
                            }
                        }
                        
                        // 存储设备信息
                        monitorElements.gpuDevices[device.index] = {
                            name: gpuName,
                            vram_total: device.vram_total,
                            index: device.index
                        };
                    }
                });
                console.log("Hardware Monitor Plugin - GPU devices info:", monitorElements.gpuDevices);
            }
            
            // 检查命令行参数中的CUDA设备
            if (data.system && data.system.argv) {
                const cudaDeviceArg = data.system.argv.indexOf('--cuda-device');
                if (cudaDeviceArg !== -1 && cudaDeviceArg + 1 < data.system.argv.length) {
                    const deviceArg = data.system.argv[cudaDeviceArg + 1];
                    // 解析设备参数（可能是单个数字或逗号分隔的列表）
                    const devices = deviceArg.split(',').map(d => parseInt(d.trim()));
                    
                    // 存储命令行指定的GPU索引
                    monitorElements.cmdGpuIndices = devices;
                    console.log("Hardware Monitor Plugin - Command line GPU indices:", devices);
                    
                    // 为命令行指定的设备添加提示信息
                    devices.forEach(cmdIndex => {
                        // 尝试从设备列表中找到对应的实际设备
                        for (const deviceIndex in monitorElements.gpuDevices) {
                            if (parseInt(deviceIndex) === cmdIndex) {
                                return; // 已存在，不需要添加
                            }
                        }
                        
                        // 如果找不到对应的实际设备，添加一个假设的设备
                        // 这主要是处理命令行指定了设备，但API中没有返回该设备的情况
                        monitorElements.gpuDevices[cmdIndex] = {
                            name: "Unknown GPU",
                            index: cmdIndex
                        };
                    });
                }
            }
            
            // 不再提前创建GPU指示器，而是在收到WebSocket数据时动态创建
            console.log("Hardware Monitor Plugin - Will create GPU indicators when data is received");
        }
    } catch (error) {
        console.error("Hardware Monitor Plugin - Error fetching system stats:", error);
    }
}

// 更新监控UI
function updateMonitorUI(monitorElements, data) {
    if (!monitorElements || !monitorElements.container) return;
    
    // 确保悬浮窗可见
    const floatContainer = document.getElementById('hardware-monitor-float');
    if (floatContainer && floatContainer.style.display === 'none') {
        floatContainer.style.display = 'block';
    }
    
    // 更新CPU使用率
    if (data.cpu_utilization !== undefined) {
        const cpuPercent = Math.round(data.cpu_utilization);
        monitorElements.cpu.update(`${cpuPercent}%`, cpuPercent);
    }
    
    // 更新RAM使用率
    if (data.ram_used_percent !== undefined) {
        const ramPercent = Math.round(data.ram_used_percent);
        monitorElements.ram.update(`${ramPercent}%`, ramPercent, data.ram_used, data.ram_total);
    }
    
    // 更新GPU信息
    if (data.gpus && data.gpus.length > 0) {
        // 检查GPU列表变化
        const allGpuIds = data.gpus.map(gpu => gpu.index);
        if (JSON.stringify(monitorElements.lastGpuIds || []) !== JSON.stringify(allGpuIds)) {
            // GPU列表有变化，清理旧的GPU组
            for (const groupId in monitorElements.gpuGroups) {
                if (monitorElements.gpuGroups[groupId] && monitorElements.gpuGroups[groupId].element) {
                    monitorElements.gpuGroups[groupId].element.remove();
                }
            }
            
            // 清空相关数组
            monitorElements.gpuGroups = {};
            monitorElements.gpus = [];
            monitorElements.vrams = [];
            monitorElements.temps = [];
            monitorElements.lastGpuIds = [...allGpuIds];
        }
        
        // 确定要显示哪些GPU
        let gpuIndicesToShow = [];
        const cmdLineSpecifiedGpus = data.gpus.filter(gpu => gpu.is_cmdline_specified);
        
        // 优先显示命令行指定的GPU
        if (cmdLineSpecifiedGpus.length > 0) {
            gpuIndicesToShow = cmdLineSpecifiedGpus.map(gpu => gpu.index);
        } else {
            // 否则显示所有GPU
            gpuIndicesToShow = allGpuIds;
        }
        
        // 处理每个要显示的GPU
        gpuIndicesToShow.forEach(gpuIndex => {
            // 从data.gpus中找到对应的GPU数据
            const gpuData = data.gpus.find(gpu => gpu.index === gpuIndex);
            if (!gpuData) return;  // 如果找不到则跳过
            
            // 格式化标签
            const gpuLabel = `GPU ${gpuIndex}`;
            const vramLabel = `显存 ${gpuIndex}`;
            const tempLabel = `温度 ${gpuIndex}`;
            
            // 准备提示信息
            const gpuName = gpuData.name || `GPU ${gpuIndex}`;
            const gpuTooltip = `GPU ${gpuIndex}: ${gpuName}`;
            
            // 确保已创建了对应的GPU组
            if (!monitorElements.gpuGroups[gpuIndex]) {
                // 创建新的GPU组
                const gpuGroup = document.createElement('div');
                gpuGroup.className = 'monitor-group gpu-group';
                
                // 添加GPU组标题
                const gpuTitle = document.createElement('div');
                gpuTitle.className = 'monitor-group-title';
                gpuTitle.textContent = `GPU ${gpuIndex}: ${gpuName.split(' ').pop()}`; // 只显示型号部分
                gpuGroup.appendChild(gpuTitle);
                
                // 添加分隔线
                if (Object.keys(monitorElements.gpuGroups).length === 0) {
                    const divider = document.createElement('div');
                    divider.className = 'monitor-divider';
                    monitorElements.container.appendChild(divider);
                }
                
                monitorElements.container.appendChild(gpuGroup);
                
                monitorElements.gpuGroups[gpuIndex] = {
                    element: gpuGroup,
                    index: gpuIndex,
                    gpuEl: null,
                    vramEl: null,
                    tempEl: null
                };
                
                // 创建GPU指示器
                const gpuEl = createMonitorElement(gpuLabel, '0%', 'gpu-progress');
                gpuEl.container.title = gpuTooltip;
                gpuGroup.appendChild(gpuEl.container);
                monitorElements.gpuGroups[gpuIndex].gpuEl = gpuEl;
                
                // 创建VRAM指示器
                const vramEl = createMonitorElement(vramLabel, '0%', 'vram-progress');
                vramEl.container.title = `${gpuTooltip} - VRAM`;
                gpuGroup.appendChild(vramEl.container);
                monitorElements.gpuGroups[gpuIndex].vramEl = vramEl;
                
                // 创建温度指示器
                const tempEl = createMonitorElement(tempLabel, '0°', 'temp-progress');
                tempEl.container.title = `${gpuTooltip} - 温度`;
                gpuGroup.appendChild(tempEl.container);
                monitorElements.gpuGroups[gpuIndex].tempEl = tempEl;
                
                // 如果有保存的字体大小，应用到新创建的GPU组
                if (monitorElements.currentFontScale && monitorElements.currentFontScale !== 1) {
                    const scale = monitorElements.currentFontScale;
                    
                    // 应用到标题
                    gpuTitle.style.fontSize = `${13 * scale}px`;
                    
                    // 应用到监控项
                    [gpuEl, vramEl, tempEl].forEach(item => {
                        if (item && item.container) {
                            item.container.style.height = `${26 * scale}px`;
                            if (item.label) item.label.style.fontSize = `${12 * scale}px`;
                            if (item.value) item.value.style.fontSize = `${12 * scale}px`;
                        }
                    });
                }
            }
            
            // 获取当前GPU组的监控元素
            const currentGroup = monitorElements.gpuGroups[gpuIndex];
            
            // 更新GPU利用率
            const gpuPercent = Math.round(gpuData.gpu_utilization);
            currentGroup.gpuEl.update(`${gpuPercent}%`, gpuPercent);
            currentGroup.gpuEl.container.title = `${gpuTooltip} - 使用率: ${gpuPercent}%`;
            
            // 更新VRAM使用率
            const vramPercent = Math.round(gpuData.vram_used_percent);
            currentGroup.vramEl.update(`${vramPercent}%`, vramPercent, gpuData.vram_used, gpuData.vram_total);
            
            // 更新温度
            const tempValue = Math.round(gpuData.gpu_temperature);
            currentGroup.tempEl.update(`${tempValue}°C`, tempValue);
            currentGroup.tempEl.container.title = `${gpuTooltip} - 温度: ${tempValue}°C`;
        });
        
        // 确保自动清理按钮始终在最后
        const cleanupButton = document.querySelector('.monitor-cleanup-button');
        if (cleanupButton) {
            cleanupButton.parentNode.appendChild(cleanupButton);
        }
    }
}
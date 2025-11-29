import { app } from "../../../scripts/app.js";

// 鲸鱼遮挡插件
class WhaleCover {
    constructor() {
        this.whales = [];
        this.container = null;
        this.panel = null;
        this.isMinimized = true; // 默认折叠
        this.init();
    }

    init() {
        // 创建容器 - 固定在body上
        this.container = document.createElement("div");
        this.container.id = "whale-cover-container";
        this.container.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9998;
            overflow: hidden;
        `;
        document.body.appendChild(this.container);

        // 创建控制面板
        this.createControlPanel();
        
        // 等待ComfyUI加载完成后加载鲸鱼
        const waitForApp = setInterval(() => {
            if (app && app.canvas) {
                clearInterval(waitForApp);
                this.loadWhales();
                this.startCanvasSync();
                this.syncStarted = true;
            }
        }, 100);
    }

    createControlPanel() {
        this.panel = document.createElement("div");
        this.panel.id = "whale-control-panel";
        this.panel.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: transparent;
            border-radius: 12px;
            padding: 5px;
            z-index: 10000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            pointer-events: auto;
            user-select: none;
        `;

        // 默认折叠状态的HTML
        this.panel.innerHTML = `
            <div id="whale-panel-header" style="
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 56px;
                cursor: pointer;
            ">🐳</div>
            <div id="whale-panel-content" style="display: none;">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid rgba(255,255,255,0.2);
                ">
                    <span style="color: white; font-weight: bold; font-size: 14px;">🐳 鲸鱼遮挡器</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <label style="color: white; font-size: 12px; display: block; margin-bottom: 5px;">数量:</label>
                    <input type="number" id="whale-count" value="1" min="1" max="50" style="
                        width: 60px;
                        padding: 5px;
                        border: none;
                        border-radius: 5px;
                        text-align: center;
                    ">
                </div>
                <div style="margin-bottom: 10px;">
                    <label style="color: white; font-size: 12px; display: block; margin-bottom: 5px;">大小:</label>
                    <input type="range" id="whale-size" min="20" max="200" value="50" style="
                        width: 100%;
                    ">
                    <span id="whale-size-label" style="color: white; font-size: 11px;">50px</span>
                </div>
                <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                    <button id="whale-add-btn" style="
                        flex: 1;
                        padding: 8px;
                        background: #4CAF50;
                        border: none;
                        border-radius: 5px;
                        color: white;
                        cursor: pointer;
                        font-size: 12px;
                    ">添加 🐳</button>
                    <button id="whale-clear-btn" style="
                        flex: 1;
                        padding: 8px;
                        background: #f44336;
                        border: none;
                        border-radius: 5px;
                        color: white;
                        cursor: pointer;
                        font-size: 12px;
                    ">清除全部</button>
                </div>
                <div style="margin-top: 10px; display: flex; gap: 5px;">
                    <button id="whale-save-btn" style="
                        flex: 1;
                        padding: 8px;
                        background: #2196F3;
                        border: none;
                        border-radius: 5px;
                        color: white;
                        cursor: pointer;
                        font-size: 12px;
                    ">保存位置</button>
                    <button id="whale-toggle-btn" style="
                        flex: 1;
                        padding: 8px;
                        background: #FF9800;
                        border: none;
                        border-radius: 5px;
                        color: white;
                        cursor: pointer;
                        font-size: 12px;
                    ">显示/隐藏</button>
                </div>
                <div style="margin-top: 8px; color: rgba(255,255,255,0.7); font-size: 10px; text-align: center;">
                    💡 双击删除 | 边缘单向拉伸 | 角落等比缩放
                </div>
            </div>
        `;

        document.body.appendChild(this.panel);

        // 绑定事件
        this.bindPanelEvents();
        this.makePanelDraggable();
    }

    bindPanelEvents() {
        const header = document.getElementById("whale-panel-header");
        const content = document.getElementById("whale-panel-content");

        // 点击鲸鱼图标展开/折叠
        header.addEventListener("click", (e) => {
            if (this.panelDragged) {
                this.panelDragged = false;
                return;
            }
            this.isMinimized = !this.isMinimized;
            if (this.isMinimized) {
                content.style.display = "none";
                header.style.fontSize = "56px";
                this.panel.style.padding = "5px";
                this.panel.style.background = "transparent";
                this.panel.style.boxShadow = "none";
            } else {
                content.style.display = "block";
                header.style.fontSize = "16px";
                this.panel.style.padding = "15px";
                this.panel.style.background = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
                this.panel.style.boxShadow = "0 4px 15px rgba(0,0,0,0.3)";
            }
        });

        // 大小滑块
        document.getElementById("whale-size").addEventListener("input", (e) => {
            document.getElementById("whale-size-label").textContent = e.target.value + "px";
        });

        // 添加鲸鱼
        document.getElementById("whale-add-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            const count = parseInt(document.getElementById("whale-count").value) || 1;
            const size = parseInt(document.getElementById("whale-size").value) || 50;
            
            // 获取ComfyUI画布
            const canvas = app.canvas;
            const canvasEl = canvas.canvas;
            const rect = canvasEl.getBoundingClientRect();
            const scale = canvas.ds?.scale || canvas.scale || 1;
            const offset = canvas.ds?.offset || canvas.offset || [0, 0];
            
            for (let i = 0; i < count; i++) {
                // 在当前视口中心附近随机生成
                const screenCenterX = window.innerWidth / 2 + (Math.random() - 0.5) * 400;
                const screenCenterY = window.innerHeight / 2 + (Math.random() - 0.5) * 300;
                
                // 屏幕坐标转图形坐标: graphPos = (screenPos - canvasOffset) / scale - offset
                const graphX = (screenCenterX - rect.left) / scale - offset[0];
                const graphY = (screenCenterY - rect.top) / scale - offset[1];
                
                this.addWhale(graphX, graphY, size, size);
            }
        });

        // 清除全部
        document.getElementById("whale-clear-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            this.clearAllWhales();
        });

        // 保存位置
        document.getElementById("whale-save-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            this.saveWhales();
            this.showNotification("位置已保存！");
        });

        // 显示/隐藏
        document.getElementById("whale-toggle-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            this.container.style.display = this.container.style.display === "none" ? "block" : "none";
        });

        // 阻止输入框事件冒泡
        document.getElementById("whale-count").addEventListener("click", e => e.stopPropagation());
        document.getElementById("whale-size").addEventListener("click", e => e.stopPropagation());
        document.getElementById("whale-count").addEventListener("mousedown", e => e.stopPropagation());
        document.getElementById("whale-size").addEventListener("mousedown", e => e.stopPropagation());
    }

    makePanelDraggable() {
        let isDragging = false;
        let startX, startY, startLeft, startTop;
        this.panelDragged = false;

        this.panel.addEventListener("mousedown", (e) => {
            if (e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
            isDragging = true;
            this.panelDragged = false;
            startX = e.clientX;
            startY = e.clientY;
            const rect = this.panel.getBoundingClientRect();
            startLeft = rect.left;
            startTop = rect.top;
        });

        document.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
                this.panelDragged = true;
                this.panel.style.left = (startLeft + dx) + "px";
                this.panel.style.top = (startTop + dy) + "px";
                this.panel.style.right = "auto";
            }
        });

        document.addEventListener("mouseup", () => {
            isDragging = false;
        });
    }

    addWhale(x, y, width, height) {
        const whale = document.createElement("div");
        whale.className = "whale-item";
        whale.dataset.canvasX = x;
        whale.dataset.canvasY = y;
        whale.dataset.width = width;
        whale.dataset.height = height;
        
        whale.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: ${width}px;
            height: ${height}px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: move;
            pointer-events: auto;
            user-select: none;
            filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
            z-index: 9999;
            transform-origin: top left;
        `;
        
        // 鲸鱼emoji
        const emoji = document.createElement("span");
        emoji.textContent = "🐳";
        emoji.style.cssText = `
            font-size: ${Math.min(width, height)}px;
            line-height: 1;
            transform: scale(${width / Math.min(width, height)}, ${height / Math.min(width, height)});
        `;
        whale.appendChild(emoji);
        
        whale.dataset.width = width;
        whale.dataset.height = height;

        // 添加8个方向的resize handles
        const handles = [
            { pos: 'n', cursor: 'ns-resize', style: 'top: -2px; left: 50%; transform: translateX(-50%); width: 16px; height: 4px;' },
            { pos: 's', cursor: 'ns-resize', style: 'bottom: -2px; left: 50%; transform: translateX(-50%); width: 16px; height: 4px;' },
            { pos: 'e', cursor: 'ew-resize', style: 'right: -2px; top: 50%; transform: translateY(-50%); width: 4px; height: 16px;' },
            { pos: 'w', cursor: 'ew-resize', style: 'left: -2px; top: 50%; transform: translateY(-50%); width: 4px; height: 16px;' },
            { pos: 'ne', cursor: 'nesw-resize', style: 'top: -3px; right: -3px; width: 6px; height: 6px; border-radius: 50%;' },
            { pos: 'nw', cursor: 'nwse-resize', style: 'top: -3px; left: -3px; width: 6px; height: 6px; border-radius: 50%;' },
            { pos: 'se', cursor: 'nwse-resize', style: 'bottom: -3px; right: -3px; width: 6px; height: 6px; border-radius: 50%;' },
            { pos: 'sw', cursor: 'nesw-resize', style: 'bottom: -3px; left: -3px; width: 6px; height: 6px; border-radius: 50%;' },
        ];

        handles.forEach(h => {
            const handle = document.createElement("div");
            handle.className = `whale-handle whale-handle-${h.pos}`;
            handle.dataset.direction = h.pos;
            handle.style.cssText = `
                position: absolute;
                ${h.style}
                background: rgba(102, 126, 234, 0.8);
                cursor: ${h.cursor};
                opacity: 0;
                transition: opacity 0.2s;
                z-index: 10;
            `;
            whale.appendChild(handle);
        });

        // 鼠标悬停显示handles
        whale.addEventListener("mouseenter", () => {
            whale.querySelectorAll(".whale-handle").forEach(h => h.style.opacity = "1");
        });
        whale.addEventListener("mouseleave", () => {
            whale.querySelectorAll(".whale-handle").forEach(h => h.style.opacity = "0");
        });

        // 拖拽功能
        this.makeWhaleDraggable(whale, emoji);

        // 缩放功能
        this.makeWhaleResizable(whale, emoji);

        // 双击删除
        whale.addEventListener("dblclick", (e) => {
            if (e.target.classList.contains("whale-handle")) return;
            whale.remove();
            this.whales = this.whales.filter(w => w !== whale);
        });

        this.container.appendChild(whale);
        this.whales.push(whale);

        return whale;
    }
    
    startCanvasSync() {
        // 监听画布变换，同步更新所有鲸鱼
        const updateWhalesTransform = () => {
            const canvas = app.canvas;
            if (canvas && canvas.canvas) {
                // 获取canvas元素在页面中的位置
                const canvasEl = canvas.canvas;
                const rect = canvasEl.getBoundingClientRect();
                
                // LiteGraph坐标转换
                const scale = canvas.ds?.scale || canvas.scale || 1;
                const offset = canvas.ds?.offset || canvas.offset || [0, 0];
                
                this.whales.forEach(whale => {
                    const canvasX = parseFloat(whale.dataset.canvasX);
                    const canvasY = parseFloat(whale.dataset.canvasY);
                    const width = parseFloat(whale.dataset.width);
                    const height = parseFloat(whale.dataset.height);
                    
                    // 图形坐标转屏幕坐标: screenPos = (graphPos + offset) * scale + canvasOffset
                    const screenX = (canvasX + offset[0]) * scale + rect.left;
                    const screenY = (canvasY + offset[1]) * scale + rect.top;
                    const screenWidth = width * scale;
                    const screenHeight = height * scale;
                    
                    whale.style.left = screenX + "px";
                    whale.style.top = screenY + "px";
                    whale.style.width = screenWidth + "px";
                    whale.style.height = screenHeight + "px";
                    
                    // 更新emoji大小
                    const emoji = whale.querySelector("span");
                    if (emoji) {
                        const baseSize = Math.min(screenWidth, screenHeight);
                        emoji.style.fontSize = baseSize + "px";
                        emoji.style.transform = `scale(${width / Math.min(width, height)}, ${height / Math.min(width, height)})`;
                    }
                });
            }
            requestAnimationFrame(updateWhalesTransform);
        };
        requestAnimationFrame(updateWhalesTransform);
    }

    makeWhaleDraggable(whale, emoji) {
        let isDragging = false;
        let startMouseX, startMouseY, startCanvasX, startCanvasY;

        whale.addEventListener("mousedown", (e) => {
            if (e.target.classList.contains("whale-handle")) return;
            isDragging = true;
            startMouseX = e.clientX;
            startMouseY = e.clientY;
            startCanvasX = parseFloat(whale.dataset.canvasX);
            startCanvasY = parseFloat(whale.dataset.canvasY);
            whale.style.zIndex = "10000";
            e.preventDefault();
        });

        document.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            const canvas = app.canvas;
            
            // 计算鼠标移动的屏幕距离，然后转换为图形空间的距离
            const scale = canvas.ds?.scale || canvas.scale || 1;
            const dx = (e.clientX - startMouseX) / scale;
            const dy = (e.clientY - startMouseY) / scale;
            
            whale.dataset.canvasX = startCanvasX + dx;
            whale.dataset.canvasY = startCanvasY + dy;
        });

        document.addEventListener("mouseup", () => {
            if (isDragging) {
                isDragging = false;
                whale.style.zIndex = "9999";
            }
        });
    }

    makeWhaleResizable(whale, emoji) {
        let isResizing = false;
        let startMouseX, startMouseY, startWidth, startHeight, startCanvasX, startCanvasY, direction;

        whale.querySelectorAll(".whale-handle").forEach(handle => {
            handle.addEventListener("mousedown", (e) => {
                isResizing = true;
                direction = handle.dataset.direction;
                startMouseX = e.clientX;
                startMouseY = e.clientY;
                startWidth = parseFloat(whale.dataset.width);
                startHeight = parseFloat(whale.dataset.height);
                startCanvasX = parseFloat(whale.dataset.canvasX);
                startCanvasY = parseFloat(whale.dataset.canvasY);
                e.preventDefault();
                e.stopPropagation();
            });
        });

        document.addEventListener("mousemove", (e) => {
            if (!isResizing) return;

            const canvas = app.canvas;
            const scale = canvas.ds?.scale || canvas.scale || 1;
            
            // 将屏幕像素移动转换为图形空间的移动
            const dx = (e.clientX - startMouseX) / scale;
            const dy = (e.clientY - startMouseY) / scale;
            let newWidth = startWidth;
            let newHeight = startHeight;
            let newCanvasX = startCanvasX;
            let newCanvasY = startCanvasY;

            // 角落：等比例缩放
            if (direction.length === 2) {
                const ratio = startWidth / startHeight;
                let delta;
                
                if (direction === 'se') {
                    delta = Math.max(dx, dy * ratio);
                    newWidth = Math.max(20, startWidth + delta);
                    newHeight = newWidth / ratio;
                } else if (direction === 'sw') {
                    delta = Math.max(-dx, dy * ratio);
                    newWidth = Math.max(20, startWidth + delta);
                    newHeight = newWidth / ratio;
                    newCanvasX = startCanvasX - (newWidth - startWidth);
                } else if (direction === 'ne') {
                    delta = Math.max(dx, -dy * ratio);
                    newWidth = Math.max(20, startWidth + delta);
                    newHeight = newWidth / ratio;
                    newCanvasY = startCanvasY - (newHeight - startHeight);
                } else if (direction === 'nw') {
                    delta = Math.max(-dx, -dy * ratio);
                    newWidth = Math.max(20, startWidth + delta);
                    newHeight = newWidth / ratio;
                    newCanvasX = startCanvasX - (newWidth - startWidth);
                    newCanvasY = startCanvasY - (newHeight - startHeight);
                }
            } else {
                // 边缘：单向拉伸
                if (direction === 'e') {
                    newWidth = Math.max(20, startWidth + dx);
                } else if (direction === 'w') {
                    newWidth = Math.max(20, startWidth - dx);
                    newCanvasX = startCanvasX + (startWidth - newWidth);
                } else if (direction === 's') {
                    newHeight = Math.max(20, startHeight + dy);
                } else if (direction === 'n') {
                    newHeight = Math.max(20, startHeight - dy);
                    newCanvasY = startCanvasY + (startHeight - newHeight);
                }
            }

            // 更新画布坐标数据
            whale.dataset.width = newWidth;
            whale.dataset.height = newHeight;
            whale.dataset.canvasX = newCanvasX;
            whale.dataset.canvasY = newCanvasY;
        });

        document.addEventListener("mouseup", () => {
            isResizing = false;
        });
    }

    clearAllWhales() {
        this.whales.forEach(whale => whale.remove());
        this.whales = [];
    }

    saveWhales() {
        const data = this.whales.map(whale => ({
            x: parseFloat(whale.dataset.canvasX),
            y: parseFloat(whale.dataset.canvasY),
            width: parseFloat(whale.dataset.width),
            height: parseFloat(whale.dataset.height)
        }));
        localStorage.setItem("comfyui-whale-cover", JSON.stringify(data));
    }

    loadWhales() {
        try {
            const data = JSON.parse(localStorage.getItem("comfyui-whale-cover") || "[]");
            data.forEach(item => {
                this.addWhale(item.x, item.y, item.width, item.height || item.width);
            });
        } catch (e) {
            console.log("No saved whales found");
        }
    }

    showNotification(message) {
        const notification = document.createElement("div");
        notification.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            z-index: 10001;
            animation: fadeInOut 2s ease;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => notification.remove(), 2000);
    }
}

// 添加动画样式
const style = document.createElement("style");
style.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateX(-50%) translateY(20px); }
        20% { opacity: 1; transform: translateX(-50%) translateY(0); }
        80% { opacity: 1; transform: translateX(-50%) translateY(0); }
        100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
    }
    
    .whale-item:hover {
        filter: drop-shadow(3px 3px 6px rgba(0,0,0,0.5)) !important;
    }
    
    #whale-control-panel button:hover {
        filter: brightness(1.1);
        transform: scale(1.02);
    }
    
    #whale-control-panel button:active {
        transform: scale(0.98);
    }
`;
document.head.appendChild(style);

// 注册扩展
app.registerExtension({
    name: "Comfy.WhaleCover",
    async setup() {
        new WhaleCover();
        console.log("🐳 Whale Cover Plugin Loaded!");
    }
});
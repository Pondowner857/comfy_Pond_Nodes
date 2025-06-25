// web/js/math_tools.js
import { app } from "../../../../scripts/app.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";

app.registerExtension({
    name: "MathTools.DynamicInputs",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 处理多数值比较节点的动态输入
        if (nodeData.name === "MultiNumberCompare") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                const numInputsWidget = this.widgets.find(w => w.name === "num_inputs");
                
                // 添加更新输入数量的按钮
                const updateButton = this.addWidget("button", "Update Inputs", null, () => {
                    this.updateInputCount();
                });
                
                // 保存原始的输入配置
                this.originalInputs = [...this.inputs];
                
                // 更新输入数量的函数
                this.updateInputCount = function() {
                    const count = numInputsWidget.value;
                    
                    // 移除所有现有的数字输入
                    while (this.inputs.length > 0) {
                        this.removeInput(0);
                    }
                    
                    // 添加指定数量的输入
                    for (let i = 1; i <= count; i++) {
                        this.addInput(`int_${i}`, "INT");
                    }
                    
                    // 调整节点大小
                    this.size = this.computeSize();
                    
                    // 重新连接已有的连接（如果有）
                    if (this.graph) {
                        this.graph.connectionChange(this);
                    }
                };
                
                // 初始化输入
                setTimeout(() => {
                    this.updateInputCount();
                }, 100);
            };
            
            // 序列化和反序列化支持
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) {
                    onSerialize.apply(this, arguments);
                }
                o.num_dynamic_inputs = this.widgets.find(w => w.name === "num_inputs").value;
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                if (o.num_dynamic_inputs) {
                    const widget = this.widgets.find(w => w.name === "num_inputs");
                    if (widget) {
                        widget.value = o.num_dynamic_inputs;
                        setTimeout(() => {
                            this.updateInputCount();
                        }, 100);
                    }
                }
            };
        }
        
        // 处理宽高比计算器节点的增强功能
        if (nodeData.name === "AspectRatioCalculator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // 添加常用比例快捷按钮
                const ratioButtons = [
                    { label: "HD (16:9)", width: 1920, height: 1080 },
                    { label: "4K (16:9)", width: 3840, height: 2160 },
                    { label: "Square", width: 1024, height: 1024 },
                    { label: "Portrait", width: 768, height: 1024 },
                    { label: "Landscape", width: 1024, height: 768 }
                ];
                
                ratioButtons.forEach(btn => {
                    this.addWidget("button", btn.label, null, () => {
                        const widthWidget = this.widgets.find(w => w.name === "width");
                        const heightWidget = this.widgets.find(w => w.name === "height");
                        if (widthWidget && heightWidget) {
                            widthWidget.value = btn.width;
                            heightWidget.value = btn.height;
                        }
                    });
                });
                
                // 添加交换宽高的按钮
                this.addWidget("button", "Swap W/H", null, () => {
                    const widthWidget = this.widgets.find(w => w.name === "width");
                    const heightWidget = this.widgets.find(w => w.name === "height");
                    if (widthWidget && heightWidget) {
                        const temp = widthWidget.value;
                        widthWidget.value = heightWidget.value;
                        heightWidget.value = temp;
                    }
                });
            };
        }
        
        // 为数学运算节点添加实时预览
        if (nodeData.name === "MathOperations") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // 添加结果预览widget
                const previewWidget = this.addWidget("text", "preview", "", () => {}, { 
                    multiline: false,
                    disabled: true 
                });
                
                // 监听值变化并更新预览（确保整数运算）
                const updatePreview = () => {
                    const a = this.widgets.find(w => w.name === "a")?.value || 0;
                    const b = this.widgets.find(w => w.name === "b")?.value || 0;
                    const op = this.widgets.find(w => w.name === "operation")?.value || "add";
                    
                    let result = 0;
                    switch(op) {
                        case "add": result = a + b; break;
                        case "subtract": result = a - b; break;
                        case "multiply": result = a * b; break;
                        case "divide": result = b !== 0 ? Math.round(a / b) : 0; break;
                        case "power": result = Math.round(Math.pow(a, b)); break;
                        case "min": result = Math.min(a, b); break;
                        case "max": result = Math.max(a, b); break;
                        case "average": result = Math.round((a + b) / 2); break;
                        case "modulo": result = b !== 0 ? a % b : 0; break;
                        case "distance": result = Math.round(Math.sqrt(a*a + b*b)); break;
                        case "log": result = (a > 0 && b > 0) ? Math.round(Math.log(a) / Math.log(b)) : 0; break;
                        case "exp": result = Math.round(Math.exp(a)); break;
                        case "sin": result = Math.round(Math.sin(a * Math.PI / 180) * 1000); break;
                        case "cos": result = Math.round(Math.cos(a * Math.PI / 180) * 1000); break;
                        case "tan": result = Math.round(Math.tan(a * Math.PI / 180) * 1000); break;
                        case "atan2": result = Math.round(Math.atan2(b, a) * 180 / Math.PI); break;
                        default: result = 0;
                    }
                    
                    // 确保结果是整数
                    result = Math.floor(result);
                    previewWidget.value = `Result: ${result}`;
                };
                
                // 为所有相关widget添加变化监听
                ["a", "b", "operation"].forEach(name => {
                    const widget = this.widgets.find(w => w.name === name);
                    if (widget) {
                        const originalCallback = widget.callback;
                        widget.callback = function() {
                            if (originalCallback) {
                                originalCallback.apply(this, arguments);
                            }
                            updatePreview();
                        };
                    }
                });
                
                // 初始预览
                setTimeout(updatePreview, 100);
            };
        }
    },
    
    // 节点类别样式
    async setup() {
        // 为数学工具节点添加自定义样式
        const style = document.createElement("style");
        style.textContent = `
            .comfy-menu-math-tools {
                background-color: #2a4d6b !important;
            }
            
            .node.MultiNumberCompare .title {
                background-color: #3a5f7d !important;
            }
            
            .node.AspectRatioCalculator .title {
                background-color: #4a6f8d !important;
            }
            
            .node.MathOperations .title {
                background-color: #5a7f9d !important;
            }
            
            /* 按钮样式 */
            .node button.comfy-widget-button {
                margin: 2px;
                padding: 4px 8px;
                border-radius: 4px;
                background-color: #4a6f8d;
                color: white;
                border: none;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .node button.comfy-widget-button:hover {
                background-color: #5a7f9d;
            }
            
            .node button.comfy-widget-button:active {
                background-color: #3a5f7d;
            }
        `;
        document.head.appendChild(style);
    }
});
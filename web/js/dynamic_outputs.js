import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.PromptParser.DynamicOutputs",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "TextFormatParser") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            // 初始化：删除多余的输出端口，只保留3个
            const initialCount = 3;
            const currentCount = this.outputs ? this.outputs.length : 0;
            if (currentCount > initialCount) {
                for (let i = currentCount; i > initialCount; i--) {
                    this.removeOutput(i - 1);
                }
            }
            
            // 设置text输入框的初始高度
            const textWidget = this.widgets?.find(w => w.name === "text");
            if (textWidget) {
                textWidget.options = textWidget.options || {};
                textWidget.options.max_height = 120;  // 限制最大高度
                textWidget.options.min_height = 60;   // 设置最小高度
                
                // 延迟设置textarea行数，确保DOM元素已创建
                setTimeout(() => {
                    if (textWidget.inputEl && textWidget.inputEl.tagName === 'TEXTAREA') {
                        textWidget.inputEl.rows = 3;
                    }
                }, 100);
            }
            
            // 重新计算节点大小
            this.setSize(this.computeSize());
            
            const outputCountWidget = this.widgets?.find(w => w.name === "output_count");
            if (!outputCountWidget) return result;
            
            // 添加"更新输出"按钮
            this.addWidget("button", "更新输出", null, () => {
                const targetCount = outputCountWidget.value;
                const currentCount = this.outputs ? this.outputs.length : 0;
                
                if (currentCount === targetCount) return;
                
                if (currentCount < targetCount) {
                    // 添加输出端口
                    for (let i = currentCount + 1; i <= targetCount; i++) {
                        this.addOutput(`输出_${i}`, "STRING");
                    }
                } else {
                    // 从后往前移除输出端口
                    for (let i = currentCount; i > targetCount; i--) {
                        this.removeOutput(i - 1);
                    }
                }
                
                this.setSize(this.computeSize());
                app.graph.setDirtyCanvas(true, true);
            });
            
            // 序列化保存
            const onSerialize = this.onSerialize;
            this.onSerialize = function(info) {
                if (onSerialize) onSerialize.call(this, info);
                info.output_count_actual = this.outputs ? this.outputs.length : 0;
            };
            
            return result;
        };
        
        // 恢复时重建输出端口
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) onConfigure.call(this, info);
            
            if (info.output_count_actual !== undefined) {
                const savedCount = info.output_count_actual;
                const currentCount = this.outputs ? this.outputs.length : 0;
                
                if (currentCount < savedCount) {
                    // 添加端口
                    for (let i = currentCount + 1; i <= savedCount; i++) {
                        this.addOutput(`输出_${i}`, "STRING");
                    }
                } else if (currentCount > savedCount) {
                    // 删除多余端口
                    for (let i = currentCount; i > savedCount; i--) {
                        this.removeOutput(i - 1);
                    }
                }
            }
        };
    },
});
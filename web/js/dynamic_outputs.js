import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.PromptParser.DynamicOutputs",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "TextFormatParser") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
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
                
                for (let i = currentCount + 1; i <= savedCount; i++) {
                    this.addOutput(`输出_${i}`, "STRING");
                }
            }
        };
    },
});

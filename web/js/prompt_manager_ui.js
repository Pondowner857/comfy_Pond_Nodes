import { app } from "../../../scripts/app.js";

const DEFAULT_PROMPT = {
    name: "新Prompt",
    text: "",
    enabled: false,
    weight: 1.0,
    type: "positive"
};

const NEGATIVE_PRESETS = [
    { name: "质量控制", text: "low quality, worst quality" },
    { name: "清晰度", text: "blurry, out of focus" },
    { name: "无水印", text: "watermark, signature, text" },
    { name: "解剖结构", text: "bad anatomy, bad hands" },
    { name: "畸形控制", text: "ugly, duplicate, mutated" },
    { name: "肢体控制", text: "extra limbs, missing limbs" },
    { name: "面部质量", text: "poorly drawn face" },
    { name: "比例控制", text: "bad proportions" },
    { name: "内容限制", text: "nsfw, nude" }
];

function createModalEditor(node) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 10000;
        display: flex;
        justify-content: center;
        align-items: center;
    `;
    
    const modal = document.createElement('div');
    modal.style.cssText = `
        background: #1a2332;
        border-radius: 12px;
        width: 900px;
        max-height: 85vh;
        display: flex;
        flex-direction: column;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    `;
    
    const header = document.createElement('div');
    header.style.cssText = `
        padding: 16px 20px;
        background: #2a5a8a;
        border-radius: 12px 12px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    `;
    
    const title = document.createElement('h2');
    title.textContent = '🐳编辑Prompt';
    title.style.cssText = 'color: white; margin: 0; font-size: 16px;';
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
        background: transparent;
        border: none;
        color: white;
        font-size: 22px;
        cursor: pointer;
        padding: 0;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: opacity 0.2s;
    `;
    closeBtn.onmouseover = () => closeBtn.style.opacity = '0.7';
    closeBtn.onmouseout = () => closeBtn.style.opacity = '1';
    closeBtn.onclick = () => {
        document.body.removeChild(overlay);
        node.updateSimpleUI();
    };
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    modal.appendChild(header);
    
    const content = document.createElement('div');
    content.style.cssText = `
        padding: 16px;
        overflow-y: auto;
        flex: 1;
        min-height: 400px;
    `;
    
    const topBar = document.createElement('div');
    topBar.style.cssText = `
        display: flex; 
        gap: 8px; 
        margin-bottom: 12px; 
        flex-wrap: wrap;
        justify-content: center;
        padding: 8px;
        background: rgba(42, 90, 138, 0.2);
        border-radius: 6px;
    `;
    
    const addBtn = document.createElement('button');
    addBtn.innerHTML = '➕ 添加正面';
    addBtn.style.cssText = `
        background: #2ecc71;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        font-weight: bold;
        transition: background 0.2s;
    `;
    addBtn.onmouseover = () => addBtn.style.background = '#27ae60';
    addBtn.onmouseout = () => addBtn.style.background = '#2ecc71';
    addBtn.onclick = () => {
        node.prompts.push({
            ...DEFAULT_PROMPT,
            name: `Prompt ${node.prompts.filter(p => p.type !== 'negative').length + 1}`,
            type: 'positive'
        });
        updateModalContent();
        node.updateJSON();
    };
    topBar.appendChild(addBtn);
    
    const addNegBtn = document.createElement('button');
    addNegBtn.innerHTML = '➕ 添加负面';
    addNegBtn.style.cssText = `
        background: #e74c3c;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        font-weight: bold;
        transition: background 0.2s;
    `;
    addNegBtn.onmouseover = () => addNegBtn.style.background = '#c0392b';
    addNegBtn.onmouseout = () => addNegBtn.style.background = '#e74c3c';
    addNegBtn.onclick = () => {
        const negCount = node.prompts.filter(p => p.type === 'negative').length;
        node.prompts.push({
            ...DEFAULT_PROMPT,
            name: `负面Prompt ${negCount + 1}`,
            type: 'negative'
        });
        updateModalContent();
        node.updateJSON();
    };
    topBar.appendChild(addNegBtn);
    
    const presetBtn = document.createElement('button');
    presetBtn.innerHTML = '📋 负面预设';
    presetBtn.style.cssText = `
        background: #9b59b6;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        font-weight: bold;
        transition: background 0.2s;
    `;
    presetBtn.onmouseover = () => presetBtn.style.background = '#8e44ad';
    presetBtn.onmouseout = () => presetBtn.style.background = '#9b59b6';
    presetBtn.onclick = () => {
        showPresetMenu(node, updateModalContent);
    };
    topBar.appendChild(presetBtn);
    
    content.appendChild(topBar);
    
    const promptsList = document.createElement('div');
    content.appendChild(promptsList);
    
    const updateModalContent = () => {
        promptsList.innerHTML = '';
        
        if (node.prompts.length === 0) {
            const emptyMsg = document.createElement('div');
            emptyMsg.textContent = '暂无Prompt，点击上方按钮添加';
            emptyMsg.style.cssText = `
                text-align: center;
                padding: 30px;
                color: #888;
                font-style: italic;
            `;
            promptsList.appendChild(emptyMsg);
            updateFooterStats();
            return;
        }
        
        // 创建两列容器
        const twoColumnContainer = document.createElement('div');
        twoColumnContainer.style.cssText = `
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        `;
        
        const positivePrompts = node.prompts.filter(p => p.type !== 'negative');
        const negativePrompts = node.prompts.filter(p => p.type === 'negative');
        
        // 左列：正面
        const posSection = createPromptSection('✅ 正面提示词', positivePrompts, false, node, updateModalContent);
        twoColumnContainer.appendChild(posSection);
        
        // 右列：负面
        const negSection = createPromptSection('❌ 负面提示词', negativePrompts, true, node, updateModalContent);
        twoColumnContainer.appendChild(negSection);
        
        promptsList.appendChild(twoColumnContainer);
        updateFooterStats();
    };
    
    const createPromptSection = (title, prompts, isNegative, node, updateCallback) => {
        const section = document.createElement('div');
        section.style.cssText = `
            border: 2px solid ${isNegative ? '#e74c3c' : '#2ecc71'};
            border-radius: 8px;
            padding: 12px;
            background: ${isNegative ? 'rgba(231, 76, 60, 0.05)' : 'rgba(46, 204, 113, 0.05)'};
            height: 100%;
            display: flex;
            flex-direction: column;
        `;
        
        const header = document.createElement('div');
        header.style.cssText = `
            font-weight: bold;
            color: ${isNegative ? '#ff9999' : '#99ccff'};
            font-size: 13px;
            margin-bottom: 12px;
            padding: 6px 10px;
            background: ${isNegative ? 'rgba(231, 76, 60, 0.15)' : 'rgba(46, 204, 113, 0.15)'};
            border-radius: 4px;
            text-align: center;
        `;
        header.textContent = title;
        section.appendChild(header);
        
        const promptsContainer = document.createElement('div');
        promptsContainer.style.cssText = 'flex: 1; overflow-y: auto;';
        
        if (prompts.length === 0) {
            const emptyMsg = document.createElement('div');
            emptyMsg.textContent = '暂无内容';
            emptyMsg.style.cssText = `
                text-align: center;
                padding: 40px 20px;
                color: #666;
                font-size: 12px;
                font-style: italic;
            `;
            promptsContainer.appendChild(emptyMsg);
        } else {
            prompts.forEach((prompt) => {
                const globalIdx = node.prompts.indexOf(prompt);
                const promptCard = createPromptCard(prompt, globalIdx, isNegative, node, updateCallback);
                promptsContainer.appendChild(promptCard);
            });
        }
        
        section.appendChild(promptsContainer);
        return section;
    };
    
    const createPromptCard = (prompt, index, isNegative, node, updateCallback) => {
        const card = document.createElement('div');
        card.style.cssText = `
            background: ${isNegative ? '#2a1a1a' : '#1a2a3a'};
            border: 1px solid ${isNegative ? '#e74c3c' : '#2ecc71'};
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            transition: all 0.2s;
        `;
        
        card.onmouseover = () => {
            card.style.background = isNegative ? '#3a2020' : '#1f3a4a';
            card.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
        };
        card.onmouseout = () => {
            card.style.background = isNegative ? '#2a1a1a' : '#1a2a3a';
            card.style.boxShadow = 'none';
        };
        
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 8px;';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = prompt.enabled;
        checkbox.style.cssText = `
            width: 16px;
            height: 16px;
            cursor: pointer;
            flex-shrink: 0;
        `;
        checkbox.onchange = (e) => {
            node.prompts[index].enabled = e.target.checked;
            node.updateJSON();
            updateCallback();
        };
        topRow.appendChild(checkbox);
        
        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.value = prompt.name;
        nameInput.style.cssText = `
            flex: 1;
            background: #0d1f2d;
            border: 1px solid #2a5a8a;
            color: white;
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 12px;
            min-width: 0;
        `;
        nameInput.oninput = (e) => {
            node.prompts[index].name = e.target.value;
            node.updateJSON();
        };
        topRow.appendChild(nameInput);
        
        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = '🗑️';
        deleteBtn.style.cssText = `
            background: #c0392b;
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
            flex-shrink: 0;
        `;
        deleteBtn.onmouseover = () => deleteBtn.style.background = '#a93226';
        deleteBtn.onmouseout = () => deleteBtn.style.background = '#c0392b';
        deleteBtn.onclick = () => {
            if (confirm(`确定删除 "${prompt.name}" 吗？`)) {
                node.prompts.splice(index, 1);
                node.updateJSON();
                updateCallback();
            }
        };
        topRow.appendChild(deleteBtn);
        
        card.appendChild(topRow);
        
        const textarea = document.createElement('textarea');
        textarea.value = prompt.text;
        textarea.style.cssText = `
            width: 100%;
            background: #0d1f2d;
            border: 1px solid #2a5a8a;
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            min-height: 60px;
            resize: vertical;
            box-sizing: border-box;
            margin-bottom: 8px;
        `;
        textarea.oninput = (e) => {
            node.prompts[index].text = e.target.value;
            node.updateJSON();
        };
        card.appendChild(textarea);
        
        const weightRow = document.createElement('div');
        weightRow.style.cssText = 'display: flex; align-items: center; gap: 8px;';
        
        const weightLabel = document.createElement('label');
        weightLabel.textContent = '权重:';
        weightLabel.style.cssText = 'color: #aaa; font-size: 11px; flex-shrink: 0;';
        weightRow.appendChild(weightLabel);
        
        const weightSlider = document.createElement('input');
        weightSlider.type = 'range';
        weightSlider.min = '0.1';
        weightSlider.max = '2.0';
        weightSlider.step = '0.01';
        weightSlider.value = prompt.weight;
        weightSlider.style.cssText = 'flex: 1;';
        weightSlider.oninput = (e) => {
            const value = parseFloat(e.target.value);
            node.prompts[index].weight = value;
            weightValue.textContent = value.toFixed(2);
            node.updateJSON();
        };
        weightRow.appendChild(weightSlider);
        
        const weightValue = document.createElement('span');
        weightValue.textContent = prompt.weight.toFixed(2);
        weightValue.style.cssText = `
            color: ${isNegative ? '#ff9999' : '#99ccff'};
            font-size: 12px;
            font-weight: bold;
            min-width: 35px;
            text-align: right;
            flex-shrink: 0;
        `;
        weightRow.appendChild(weightValue);
        
        card.appendChild(weightRow);
        
        return card;
    };
    
    const footer = document.createElement('div');
    footer.style.cssText = `
        padding: 12px 16px;
        background: #0d1f2d;
        border-radius: 0 0 12px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #2a5a8a;
    `;
    
    const stats = document.createElement('div');
    stats.style.cssText = 'color: #aaa; font-size: 11px;';
    footer.appendChild(stats);
    
    const updateFooterStats = () => {
        const positiveCount = node.prompts.filter(p => p.type !== 'negative' && p.enabled).length;
        const negativeCount = node.prompts.filter(p => p.type === 'negative' && p.enabled).length;
        stats.innerHTML = `
            <span style="color: #2ecc71;">✅ 正面: ${positiveCount}</span> | 
            <span style="color: #e74c3c;">❌ 负面: ${negativeCount}</span> | 
            总计: ${node.prompts.length}
        `;
    };
    
    modal.appendChild(content);
    modal.appendChild(footer);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    
    updateModalContent();
}

function showPresetMenu(node, updateCallback) {
    const menu = document.createElement('div');
    menu.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #1a2332;
        border: 2px solid #2a5a8a;
        border-radius: 8px;
        padding: 12px;
        z-index: 10001;
        min-width: 300px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    `;
    
    const title = document.createElement('div');
    title.textContent = '📋 负面提示词预设';
    title.style.cssText = `
        color: white;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 14px;
    `;
    menu.appendChild(title);
    
    NEGATIVE_PRESETS.forEach(preset => {
        const btn = document.createElement('button');
        btn.textContent = `${preset.name}: ${preset.text}`;
        btn.style.cssText = `
            display: block;
            width: 100%;
            background: #2a1a1a;
            border: 1px solid #e74c3c;
            color: white;
            padding: 8px;
            margin-bottom: 6px;
            border-radius: 4px;
            cursor: pointer;
            text-align: left;
            font-size: 11px;
            transition: background 0.2s;
        `;
        btn.onmouseover = () => btn.style.background = '#3a2020';
        btn.onmouseout = () => btn.style.background = '#2a1a1a';
        btn.onclick = () => {
            node.prompts.push({
                name: preset.name,
                text: preset.text,
                enabled: true,
                weight: 1.0,
                type: 'negative'
            });
            node.updateJSON();
            updateCallback();
            document.body.removeChild(menu);
        };
        menu.appendChild(btn);
    });
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = `
        width: 100%;
        background: #2a5a8a;
        border: none;
        color: white;
        padding: 8px;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 6px;
        transition: background 0.2s;
    `;
    closeBtn.onmouseover = () => closeBtn.style.background = '#3a6a9a';
    closeBtn.onmouseout = () => closeBtn.style.background = '#2a5a8a';
    closeBtn.onclick = () => document.body.removeChild(menu);
    menu.appendChild(closeBtn);
    
    document.body.appendChild(menu);
}

app.registerExtension({
    name: "Pond.CustomPromptManager.TwoColumns",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CustomPromptManagerWithNegative") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.prompts = [];
                
                setTimeout(() => {
                    if (this.widgets) {
                        const jsonWidget = this.widgets.find(w => w.name === "prompts_json");
                        if (jsonWidget) {
                            jsonWidget.type = "converted-widget";
                            jsonWidget.computeSize = () => [0, -4];
                        }
                    }
                }, 10);
                
                const widget = this.addDOMWidget("prompts_selector", "div", document.createElement("div"));
                widget.computeSize = () => [this.size[0], Math.max(140, Math.max(
                    this.prompts.filter(p => p.type !== 'negative').length,
                    this.prompts.filter(p => p.type === 'negative').length
                ) * 22 + 90)];
                
                const container = widget.element;
                container.style.cssText = `
                    width: 100%;
                    background: #0a1929;
                    border-radius: 6px;
                    padding: 8px;
                `;
                
                this.updateSimpleUI = () => {
                    container.innerHTML = '';
                    
                    const header = document.createElement('div');
                    header.style.cssText = `
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 6px 8px;
                        background: #2a5a8a;
                        border-radius: 4px;
                        margin-bottom: 8px;
                    `;
                    
                    const title = document.createElement('span');
                    const posCount = this.prompts.filter(p => p.type !== 'negative').length;
                    const negCount = this.prompts.filter(p => p.type === 'negative').length;
                    const posEnabled = this.prompts.filter(p => p.type !== 'negative' && p.enabled).length;
                    const negEnabled = this.prompts.filter(p => p.type === 'negative' && p.enabled).length;
                    
                    title.innerHTML = `📄 正面: <span style="color: #2ecc71;">${posEnabled}</span> | 负面: <span style="color: #e74c3c;">${negEnabled}</span>`;
                    title.style.cssText = 'color: white; font-size: 11px; font-weight: bold;';
                    header.appendChild(title);
                    
                    const editBtn = document.createElement('button');
                    editBtn.textContent = '✏️ 编辑';
                    editBtn.style.cssText = `
                        background: #3498db;
                        color: white;
                        border: none;
                        padding: 3px 8px;
                        border-radius: 3px;
                        cursor: pointer;
                        font-size: 11px;
                        font-weight: bold;
                        transition: background 0.2s;
                    `;
                    editBtn.onmouseover = () => editBtn.style.background = '#2980b9';
                    editBtn.onmouseout = () => editBtn.style.background = '#3498db';
                    editBtn.onclick = () => createModalEditor(this);
                    header.appendChild(editBtn);
                    
                    container.appendChild(header);
                    
                    if (this.prompts.length === 0) {
                        const emptyMsg = document.createElement('div');
                        emptyMsg.textContent = '暂无Prompt，点击"编辑"添加';
                        emptyMsg.style.cssText = `
                            text-align: center;
                            padding: 20px;
                            color: #888;
                            font-style: italic;
                            font-size: 11px;
                        `;
                        container.appendChild(emptyMsg);
                    } else {
                        const twoColumnContainer = document.createElement('div');
                        twoColumnContainer.style.cssText = `
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 6px;
                        `;
                        
                        const positiveColumn = document.createElement('div');
                        positiveColumn.style.cssText = `
                            border: 1px solid #2ecc71;
                            border-radius: 4px;
                            padding: 4px;
                            background: rgba(46, 204, 113, 0.05);
                        `;
                        
                        const positiveHeader = document.createElement('div');
                        positiveHeader.textContent = '✅ 正面';
                        positiveHeader.style.cssText = `
                            color: #2ecc71;
                            font-size: 9px;
                            font-weight: bold;
                            margin-bottom: 4px;
                            text-align: center;
                            padding: 2px;
                            background: rgba(46, 204, 113, 0.1);
                            border-radius: 2px;
                        `;
                        positiveColumn.appendChild(positiveHeader);
                        
                        const negativeColumn = document.createElement('div');
                        negativeColumn.style.cssText = `
                            border: 1px solid #e74c3c;
                            border-radius: 4px;
                            padding: 4px;
                            background: rgba(231, 76, 60, 0.05);
                        `;
                        
                        const negativeHeader = document.createElement('div');
                        negativeHeader.textContent = '❌ 负面';
                        negativeHeader.style.cssText = `
                            color: #e74c3c;
                            font-size: 9px;
                            font-weight: bold;
                            margin-bottom: 4px;
                            text-align: center;
                            padding: 2px;
                            background: rgba(231, 76, 60, 0.1);
                            border-radius: 2px;
                        `;
                        negativeColumn.appendChild(negativeHeader);
                        
                        const positivePrompts = this.prompts.filter(p => p.type !== 'negative');
                        if (positivePrompts.length === 0) {
                            const emptyMsg = document.createElement('div');
                            emptyMsg.textContent = '暂无';
                            emptyMsg.style.cssText = `
                                text-align: center;
                                padding: 8px;
                                color: #666;
                                font-size: 9px;
                                font-style: italic;
                            `;
                            positiveColumn.appendChild(emptyMsg);
                        } else {
                            positivePrompts.forEach((prompt) => {
                                const index = this.prompts.indexOf(prompt);
                                const row = createPromptRow(this, prompt, index, false);
                                positiveColumn.appendChild(row);
                            });
                        }
                        
                        const negativePrompts = this.prompts.filter(p => p.type === 'negative');
                        if (negativePrompts.length === 0) {
                            const emptyMsg = document.createElement('div');
                            emptyMsg.textContent = '暂无';
                            emptyMsg.style.cssText = `
                                text-align: center;
                                padding: 8px;
                                color: #666;
                                font-size: 9px;
                                font-style: italic;
                            `;
                            negativeColumn.appendChild(emptyMsg);
                        } else {
                            negativePrompts.forEach((prompt) => {
                                const index = this.prompts.indexOf(prompt);
                                const row = createPromptRow(this, prompt, index, true);
                                negativeColumn.appendChild(row);
                            });
                        }
                        
                        twoColumnContainer.appendChild(positiveColumn);
                        twoColumnContainer.appendChild(negativeColumn);
                        container.appendChild(twoColumnContainer);
                    }
                };
                
                const createPromptRow = (node, prompt, index, isNegative) => {
                    const row = document.createElement('label');
                    row.style.cssText = `
                        display: flex;
                        align-items: center;
                        padding: 3px 4px;
                        background: ${prompt.enabled ? (isNegative ? '#4a1a1a' : '#1a3a5a') : '#0d1f2d'};
                        border: 1px solid ${prompt.enabled ? (isNegative ? '#e74c3c' : '#2ecc71') : '#2a5a8a'};
                        border-radius: 3px;
                        margin-bottom: 2px;
                        cursor: pointer;
                        transition: all 0.2s;
                    `;
                    
                    row.onmouseover = () => {
                        row.style.background = prompt.enabled ? (isNegative ? '#5a2020' : '#1f4a6a') : '#162837';
                    };
                    row.onmouseout = () => {
                        row.style.background = prompt.enabled ? (isNegative ? '#4a1a1a' : '#1a3a5a') : '#0d1f2d';
                    };
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.checked = prompt.enabled;
                    checkbox.style.cssText = `
                        width: 11px;
                        height: 11px;
                        margin-right: 4px;
                        cursor: pointer;
                        flex-shrink: 0;
                    `;
                    checkbox.onchange = (e) => {
                        e.stopPropagation();
                        node.prompts[index].enabled = e.target.checked;
                        node.updateJSON();
                        node.updateSimpleUI();
                    };
                    
                    row.appendChild(checkbox);
                    
                    const textContainer = document.createElement('div');
                    textContainer.style.cssText = 'flex: 1; display: flex; align-items: center; min-width: 0;';
                    
                    const name = document.createElement('span');
                    name.textContent = prompt.name;
                    name.style.cssText = `
                        color: ${prompt.enabled ? (isNegative ? '#ff9999' : 'white') : '#aaa'};
                        font-size: 9px;
                        font-weight: ${prompt.enabled ? 'bold' : 'normal'};
                        flex: 1;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    `;
                    
                    textContainer.appendChild(name);
                    
                    if (prompt.weight !== 1.0) {
                        const weight = document.createElement('span');
                        weight.textContent = `${prompt.weight.toFixed(2)}`;
                        weight.style.cssText = `
                            color: ${isNegative ? '#ff9999' : '#888'};
                            font-size: 8px;
                            margin-left: 3px;
                            flex-shrink: 0;
                        `;
                        textContainer.appendChild(weight);
                    }
                    
                    row.appendChild(textContainer);
                    return row;
                };
                
                this.updateJSON = () => {
                    const jsonValue = JSON.stringify(this.prompts);
                    if (this.widgets) {
                        const jsonWidget = this.widgets.find(w => w.name === "prompts_json");
                        if (jsonWidget) {
                            jsonWidget.value = jsonValue;
                        }
                    }
                };
                
                this.updateSimpleUI();
                this.updateJSON();
                
                const maxPromptsInColumn = Math.max(
                    this.prompts.filter(p => p.type !== 'negative').length,
                    this.prompts.filter(p => p.type === 'negative').length,
                    1
                );
                const height = Math.max(180, maxPromptsInColumn * 22 + 90);
                this.setSize([480, height]);
                
                return r;
            };
            
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                const menuOptions = [
                    {
                        content: "🐳编辑Prompt",
                        callback: () => createModalEditor(this)
                    },
                    null,
                    {
                        content: "✓ 启用所有Prompt",
                        callback: () => {
                            this.prompts.forEach(p => p.enabled = true);
                            this.updateJSON();
                            this.updateSimpleUI();
                        }
                    },
                    {
                        content: "✗ 关闭所有Prompt",
                        callback: () => {
                            this.prompts.forEach(p => p.enabled = false);
                            this.updateJSON();
                            this.updateSimpleUI();
                        }
                    },
                    null,
                    {
                        content: "✓ 启用所有正面",
                        callback: () => {
                            this.prompts.forEach(p => {
                                if (p.type !== 'negative') p.enabled = true;
                            });
                            this.updateJSON();
                            this.updateSimpleUI();
                        }
                    },
                    {
                        content: "✗ 关闭所有正面",
                        callback: () => {
                            this.prompts.forEach(p => {
                                if (p.type !== 'negative') p.enabled = false;
                            });
                            this.updateJSON();
                            this.updateSimpleUI();
                        }
                    },
                    null,
                    {
                        content: "✓ 启用所有负面",
                        callback: () => {
                            this.prompts.forEach(p => {
                                if (p.type === 'negative') p.enabled = true;
                            });
                            this.updateJSON();
                            this.updateSimpleUI();
                        }
                    },
                    {
                        content: "✗ 关闭所有负面",
                        callback: () => {
                            this.prompts.forEach(p => {
                                if (p.type === 'negative') p.enabled = false;
                            });
                            this.updateJSON();
                            this.updateSimpleUI();
                        }
                    }
                ];
                
                options.unshift(...menuOptions, null);
            };
            
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) {
                    onSerialize.apply(this, arguments);
                }
                o.prompts = this.prompts;
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                if (o.prompts) {
                    this.prompts = o.prompts;
                    setTimeout(() => {
                        if (this.updateSimpleUI) {
                            this.updateSimpleUI();
                            this.updateJSON();
                        }
                    }, 100);
                }
            };
        }
    }
});

console.log("✅ Custom Prompt Manager (Two Columns + 2 Decimals) loaded");

import { app } from "../../../scripts/app.js";

// ==================== 常量定义 ====================
const DEFAULT_PROMPT = {
    name: "新Prompt",
    text: "",
    enabled: false,  // 默认关闭
    weight: 1.0,     // 默认权重1
    type: "positive",
    tags: [],
    id: Date.now() + Math.random()
};

// 本地存储键名
const STORAGE_KEYS = {
    COLUMN_WIDTH: 'prompt_manager_column_width',
    COLLAPSED_COLUMNS: 'prompt_manager_collapsed',
    TEMPLATES: 'prompt_manager_templates',
    HISTORY: 'prompt_manager_history',
    NODE_DATA: 'prompt_manager_node_data'  // 新增：保存节点数据
};

// ==================== 工具函数 ====================
function generateId() {
    return Date.now() + Math.random();
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function uploadJSON(callback) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);
                    callback(data);
                } catch (error) {
                    alert('JSON解析失败: ' + error.message);
                }
            };
            reader.readAsText(file);
        }
    };
    input.click();
}

// 批量导入Text文件
function uploadTextFiles(callback) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt';
    input.multiple = true;
    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        const prompts = [];
        
        for (const file of files) {
            try {
                const text = await readFileAsText(file);
                const name = file.name.replace('.txt', '');
                prompts.push({
                    name: name,
                    text: text.trim(),
                    enabled: false,
                    weight: 1.0,
                    type: 'positive',
                    tags: [],
                    id: generateId()
                });
            } catch (error) {
                console.error(`读取文件失败 ${file.name}:`, error);
            }
        }
        
        callback(prompts);
    };
    input.click();
}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

// 显示提示消息
function showToast(message, color = '#27ae60', duration = 2000) {
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${color};
        color: white;
        padding: 12px 20px;
        border-radius: 6px;
        z-index: 10002;
        font-size: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = 'opacity 0.3s';
        toast.style.opacity = '0';
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 300);
    }, duration);
}

// 批量导出为Text文件
function exportToTextFiles(prompts) {
    prompts.forEach((prompt, index) => {
        const blob = new Blob([prompt.text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${String(index + 1).padStart(3, '0')}_${prompt.name}.txt`;
        setTimeout(() => {
            a.click();
            URL.revokeObjectURL(url);
        }, index * 100);
    });
}

// ==================== 本地存储管理 ====================
const LocalStorage = {
    getColumnWidth() {
        const stored = localStorage.getItem(STORAGE_KEYS.COLUMN_WIDTH);
        return stored ? parseFloat(stored) : 50;
    },
    
    setColumnWidth(width) {
        localStorage.setItem(STORAGE_KEYS.COLUMN_WIDTH, width.toString());
    },
    
    getCollapsedColumns() {
        const stored = localStorage.getItem(STORAGE_KEYS.COLLAPSED_COLUMNS);
        return stored ? JSON.parse(stored) : { positive: false, negative: false };
    },
    
    setCollapsedColumns(collapsed) {
        localStorage.setItem(STORAGE_KEYS.COLLAPSED_COLUMNS, JSON.stringify(collapsed));
    },
    
    getTemplates() {
        const stored = localStorage.getItem(STORAGE_KEYS.TEMPLATES);
        return stored ? JSON.parse(stored) : {};
    },
    
    setTemplates(templates) {
        localStorage.setItem(STORAGE_KEYS.TEMPLATES, JSON.stringify(templates));
    },
    
    saveTemplate(name, data) {
        const templates = this.getTemplates();
        templates[name] = {
            name: name,
            prompts: data,
            createdAt: new Date().toISOString()
        };
        this.setTemplates(templates);
    },
    
    deleteTemplate(name) {
        const templates = this.getTemplates();
        delete templates[name];
        this.setTemplates(templates);
    },
    
    // 历史记录管理
    getHistory() {
        const stored = localStorage.getItem(STORAGE_KEYS.HISTORY);
        return stored ? JSON.parse(stored) : [];
    },
    
    addHistory(prompts) {
        const history = this.getHistory();
        const newEntry = {
            id: generateId(),
            prompts: prompts,
            timestamp: new Date().toISOString(),
            date: new Date().toLocaleString('zh-CN')
        };
        history.unshift(newEntry);
        // 保留最近50条
        if (history.length > 50) {
            history.splice(50);
        }
        localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(history));
    },
    
    deleteHistory(id) {
        const history = this.getHistory();
        const filtered = history.filter(h => h.id !== id);
        localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(filtered));
    },
    
    clearHistory() {
        localStorage.removeItem(STORAGE_KEYS.HISTORY);
    },
    
    // 节点数据管理（永久保存）
    getNodeData(nodeType) {
        const stored = localStorage.getItem(STORAGE_KEYS.NODE_DATA);
        const allData = stored ? JSON.parse(stored) : {};
        return allData[nodeType] || null;
    },
    
    saveNodeData(nodeType, prompts) {
        const stored = localStorage.getItem(STORAGE_KEYS.NODE_DATA);
        const allData = stored ? JSON.parse(stored) : {};
        allData[nodeType] = {
            prompts: prompts,
            savedAt: new Date().toISOString(),
            date: new Date().toLocaleString('zh-CN')
        };
        localStorage.setItem(STORAGE_KEYS.NODE_DATA, JSON.stringify(allData));
    },
    
    deleteNodeData(nodeType) {
        const stored = localStorage.getItem(STORAGE_KEYS.NODE_DATA);
        const allData = stored ? JSON.parse(stored) : {};
        delete allData[nodeType];
        localStorage.setItem(STORAGE_KEYS.NODE_DATA, JSON.stringify(allData));
    }
};

// ==================== 模态编辑器 ====================
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
        width: 95%;
        max-width: 1400px;
        max-height: 90vh;
        display: flex;
        flex-direction: column;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    `;
    
    const state = {
        searchText: '',
        selectedTags: [],
        collapsedColumns: LocalStorage.getCollapsedColumns(),
        columnWidth: LocalStorage.getColumnWidth(),
        draggedItem: null
    };
    
    const header = createHeader(overlay, node, state);
    modal.appendChild(header);
    
    const tagFilterBar = createTagFilterBar(node, state, () => updateModalContent());
    modal.appendChild(tagFilterBar);
    
    const toolbar = createToolbar(node, state, () => updateModalContent());
    modal.appendChild(toolbar);
    
    const content = document.createElement('div');
    content.style.cssText = `
        padding: 16px;
        overflow-y: auto;
        flex: 1;
        min-height: 400px;
    `;
    
    const promptsList = document.createElement('div');
    content.appendChild(promptsList);
    modal.appendChild(content);
    
    const footer = createFooter();
    modal.appendChild(footer);
    
    function updateModalContent() {
        const newTagFilterBar = createTagFilterBar(node, state, () => updateModalContent());
        modal.replaceChild(newTagFilterBar, modal.children[1]);
        
        promptsList.innerHTML = '';
        
        const filteredPrompts = filterPrompts(node.prompts, state.searchText, state.selectedTags);
        
        if (filteredPrompts.length === 0) {
            const emptyMsg = document.createElement('div');
            emptyMsg.textContent = '没有找到匹配的Prompt';
            emptyMsg.style.cssText = `
                text-align: center;
                padding: 40px;
                color: #666;
                font-size: 14px;
            `;
            promptsList.appendChild(emptyMsg);
            updateFooter();
            return;
        }
        
        const columnsContainer = document.createElement('div');
        columnsContainer.style.cssText = `
            display: flex;
            gap: 0;
            min-height: 400px;
            position: relative;
        `;
        
        const positiveColumn = createColumn('positive', filteredPrompts, node, state, updateModalContent);
        const negativeColumn = createColumn('negative', filteredPrompts, node, state, updateModalContent);
        const resizer = createResizer(positiveColumn, negativeColumn, state, columnsContainer);
        
        columnsContainer.appendChild(positiveColumn);
        columnsContainer.appendChild(resizer);
        columnsContainer.appendChild(negativeColumn);
        
        promptsList.appendChild(columnsContainer);
        updateFooter();
    }
    
    function updateFooter() {
        const totalCount = node.prompts.length;
        const enabledCount = node.prompts.filter(p => p.enabled).length;
        const positiveCount = node.prompts.filter(p => p.type !== 'negative').length;
        const negativeCount = node.prompts.filter(p => p.type === 'negative').length;
        
        footer.innerHTML = `
            <div style="display: flex; gap: 20px; font-size: 12px; color: #aaa;">
                <span>总计: ${totalCount}</span>
                <span>启用: ${enabledCount}</span>
                <span style="color: #2ecc71;">正面: ${positiveCount}</span>
                <span style="color: #e74c3c;">负面: ${negativeCount}</span>
            </div>
        `;
    }
    
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    updateModalContent();
    
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            document.body.removeChild(overlay);
        }
    });
}

function createHeader(overlay, node, state) {
    const header = document.createElement('div');
    header.style.cssText = `
        padding: 16px 20px;
        border-bottom: 1px solid #2a5a8a;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(180deg, #243447 0%, #1a2332 100%);
        border-radius: 12px 12px 0 0;
    `;
    
    const title = document.createElement('h3');
    title.textContent = '🐳Prompt管理器星球版';
    title.style.cssText = `
        margin: 0;
        color: #fff;
        font-size: 18px;
        font-weight: bold;
    `;
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
        background: transparent;
        border: none;
        color: #fff;
        font-size: 24px;
        cursor: pointer;
        width: 32px;
        height: 32px;
        border-radius: 4px;
        transition: background 0.2s;
    `;
    closeBtn.onmouseover = () => closeBtn.style.background = 'rgba(255,255,255,0.1)';
    closeBtn.onmouseout = () => closeBtn.style.background = 'transparent';
    closeBtn.onclick = () => {
        // 显示保存确认对话框
        const confirmDialog = document.createElement('div');
        confirmDialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #1a2332;
            border: 2px solid #2a5a8a;
            border-radius: 12px;
            padding: 24px;
            z-index: 10002;
            min-width: 400px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        `;
        
        const dialogTitle = document.createElement('h3');
        dialogTitle.textContent = '💾 保存提示';
        dialogTitle.style.cssText = `
            margin: 0 0 16px 0;
            color: #fff;
            font-size: 18px;
        `;
        
        const dialogText = document.createElement('p');
        dialogText.textContent = '是否保存当前编辑的Prompt数据？保存后下次打开会自动加载。';
        dialogText.style.cssText = `
            margin: 0 0 20px 0;
            color: #aaa;
            font-size: 14px;
            line-height: 1.6;
        `;
        
        const buttonGroup = document.createElement('div');
        buttonGroup.style.cssText = `
            display: flex;
            gap: 12px;
            justify-content: flex-end;
        `;
        
        const saveBtn = document.createElement('button');
        saveBtn.textContent = '💾 保存';
        saveBtn.style.cssText = `
            padding: 10px 20px;
            background: #27ae60;
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        `;
        saveBtn.onmouseover = () => saveBtn.style.background = '#229954';
        saveBtn.onmouseout = () => saveBtn.style.background = '#27ae60';
        saveBtn.onclick = () => {
            // 保存节点数据
            const nodeType = node.comfyClass || 'CustomPromptManagerEnhanced';
            LocalStorage.saveNodeData(nodeType, node.prompts);
            document.body.removeChild(confirmDialog);
            document.body.removeChild(overlay);
            // 显示成功提示
            showToast('✓ 已保存，下次打开将自动加载', '#27ae60');
        };
        
        const noSaveBtn = document.createElement('button');
        noSaveBtn.textContent = '不保存';
        noSaveBtn.style.cssText = `
            padding: 10px 20px;
            background: #95a5a6;
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        `;
        noSaveBtn.onmouseover = () => noSaveBtn.style.background = '#7f8c8d';
        noSaveBtn.onmouseout = () => noSaveBtn.style.background = '#95a5a6';
        noSaveBtn.onclick = () => {
            document.body.removeChild(confirmDialog);
            document.body.removeChild(overlay);
        };
        
        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = '取消';
        cancelBtn.style.cssText = `
            padding: 10px 20px;
            background: #e74c3c;
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        `;
        cancelBtn.onmouseover = () => cancelBtn.style.background = '#c0392b';
        cancelBtn.onmouseout = () => cancelBtn.style.background = '#e74c3c';
        cancelBtn.onclick = () => {
            document.body.removeChild(confirmDialog);
        };
        
        buttonGroup.appendChild(saveBtn);
        buttonGroup.appendChild(noSaveBtn);
        buttonGroup.appendChild(cancelBtn);
        
        confirmDialog.appendChild(dialogTitle);
        confirmDialog.appendChild(dialogText);
        confirmDialog.appendChild(buttonGroup);
        document.body.appendChild(confirmDialog);
    };
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    return header;
}

function createTagFilterBar(node, state, onUpdate) {
    const bar = document.createElement('div');
    bar.style.cssText = `
        padding: 12px 20px;
        border-bottom: 1px solid #2a5a8a;
        background: #1d2d3d;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
    `;
    
    const label = document.createElement('span');
    label.textContent = '🏷️ 标签筛选:';
    label.style.cssText = `
        color: #aaa;
        font-size: 12px;
        font-weight: bold;
    `;
    bar.appendChild(label);
    
    const allTags = new Set();
    node.prompts.forEach(p => {
        (p.tags || []).forEach(tag => allTags.add(tag));
    });
    
    if (allTags.size === 0) {
        const noTags = document.createElement('span');
        noTags.textContent = '(暂无标签)';
        noTags.style.cssText = 'color: #666; font-size: 11px; font-style: italic;';
        bar.appendChild(noTags);
        return bar;
    }
    
    const allBtn = createTagButton('全部', state.selectedTags.length === 0, () => {
        state.selectedTags = [];
        onUpdate();
    });
    bar.appendChild(allBtn);
    
    Array.from(allTags).sort().forEach(tag => {
        const isActive = state.selectedTags.includes(tag);
        const btn = createTagButton(tag, isActive, () => {
            if (isActive) {
                state.selectedTags = state.selectedTags.filter(t => t !== tag);
            } else {
                state.selectedTags.push(tag);
            }
            onUpdate();
        });
        bar.appendChild(btn);
    });
    
    return bar;
}

function createTagButton(text, isActive, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = `
        padding: 6px 12px;
        background: ${isActive ? '#3498db' : '#2a3f5f'};
        border: 1px solid ${isActive ? '#3498db' : '#2a5a8a'};
        border-radius: 6px;
        color: white;
        font-size: 11px;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
    `;
    btn.onmouseover = () => {
        if (!isActive) btn.style.background = '#3a5f8f';
    };
    btn.onmouseout = () => {
        if (!isActive) btn.style.background = '#2a3f5f';
    };
    btn.onclick = onClick;
    return btn;
}

function createToolbar(node, state, onUpdate) {
    const toolbar = document.createElement('div');
    toolbar.style.cssText = `
        padding: 12px 20px;
        border-bottom: 1px solid #2a5a8a;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        background: #1d2d3d;
    `;
    
    const searchBox = document.createElement('input');
    searchBox.type = 'text';
    searchBox.placeholder = '🔍 搜索Prompt...';
    searchBox.style.cssText = `
        flex: 1;
        min-width: 200px;
        padding: 8px 12px;
        background: #0d1f2d;
        border: 1px solid #2a5a8a;
        border-radius: 6px;
        color: #fff;
        font-size: 13px;
    `;
    searchBox.oninput = (e) => {
        state.searchText = e.target.value;
        onUpdate();
    };
    
    const btnGroup = document.createElement('div');
    btnGroup.style.cssText = 'display: flex; gap: 8px; flex-wrap: wrap;';
    
    const addPositiveBtn = createToolbarButton('+ 正面', '#2ecc71', () => {
        node.prompts.push({...DEFAULT_PROMPT, id: generateId(), type: 'positive'});
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    });
    
    const addNegativeBtn = createToolbarButton('+ 负面', '#e74c3c', () => {
        node.prompts.push({...DEFAULT_PROMPT, id: generateId(), type: 'negative'});
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    });
    
    const batchEnableBtn = createToolbarButton('✓ 全部启用', '#3498db', () => {
        node.prompts.forEach(p => p.enabled = true);
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    });
    
    const batchDisableBtn = createToolbarButton('✗ 全部禁用', '#95a5a6', () => {
        node.prompts.forEach(p => p.enabled = false);
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    });
    
    const toggleEnableBtn = createToolbarButton('🔄 反转启用', '#9b59b6', () => {
        node.prompts.forEach(p => p.enabled = !p.enabled);
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    });
    
    const importTextBtn = createToolbarButton('📁 导入Text', '#16a085', () => {
        uploadTextFiles((newPrompts) => {
            node.prompts.push(...newPrompts);
            node.updateJSON();
            node.updateSimpleUI();
            // 自动保存到历史记录
            LocalStorage.addHistory(node.prompts);
            onUpdate();
            showToast(`✓ 已导入 ${newPrompts.length} 个Prompt并自动保存到历史记录`, '#27ae60');
        });
    });
    
    const exportTextBtn = createToolbarButton('💾 导出Text', '#27ae60', () => {
        if (node.prompts.length === 0) {
            alert('没有可导出的Prompt');
            return;
        }
        exportToTextFiles(node.prompts);
    });
    
    const exportBtn = createToolbarButton('📤 导出JSON', '#f39c12', () => {
        downloadJSON(node.prompts, 'prompts_export.json');
    });
    
    const importBtn = createToolbarButton('📥 导入JSON', '#e67e22', () => {
        uploadJSON((data) => {
            if (Array.isArray(data)) {
                node.prompts = data.map(p => ({...p, id: p.id || generateId()}));
                node.updateJSON();
                node.updateSimpleUI();
                // 自动保存到历史记录
                LocalStorage.addHistory(node.prompts);
                onUpdate();
                showToast('✓ 已导入并自动保存到历史记录', '#27ae60');
            }
        });
    });
    
    const templateBtn = createToolbarButton('📋 模板', '#8e44ad', () => {
        showTemplateManager(node, onUpdate);
    });
    
    const historyBtn = createToolbarButton('📜 历史', '#34495e', () => {
        showHistoryManager(node, onUpdate);
    });
    
    const saveHistoryBtn = createToolbarButton('💾 保存历史', '#2c3e50', () => {
        LocalStorage.addHistory(node.prompts);
        alert('已保存到历史记录');
    });
    
    btnGroup.appendChild(addPositiveBtn);
    btnGroup.appendChild(addNegativeBtn);
    btnGroup.appendChild(batchEnableBtn);
    btnGroup.appendChild(batchDisableBtn);
    btnGroup.appendChild(toggleEnableBtn);
    btnGroup.appendChild(importTextBtn);
    btnGroup.appendChild(exportTextBtn);
    btnGroup.appendChild(exportBtn);
    btnGroup.appendChild(importBtn);
    btnGroup.appendChild(templateBtn);
    btnGroup.appendChild(historyBtn);
    btnGroup.appendChild(saveHistoryBtn);
    
    toolbar.appendChild(searchBox);
    toolbar.appendChild(btnGroup);
    
    return toolbar;
}

function createToolbarButton(text, color, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = `
        padding: 8px 16px;
        background: ${color};
        border: none;
        border-radius: 6px;
        color: white;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
    `;
    btn.onmouseover = () => btn.style.opacity = '0.8';
    btn.onmouseout = () => btn.style.opacity = '1';
    btn.onclick = onClick;
    return btn;
}

function createFooter() {
    const footer = document.createElement('div');
    footer.style.cssText = `
        padding: 12px 20px;
        border-top: 1px solid #2a5a8a;
        background: #1d2d3d;
        border-radius: 0 0 12px 12px;
    `;
    return footer;
}

function filterPrompts(prompts, searchText, selectedTags) {
    return prompts.filter(p => {
        const matchSearch = !searchText || 
            p.name.toLowerCase().includes(searchText.toLowerCase()) ||
            p.text.toLowerCase().includes(searchText.toLowerCase());
        
        const matchTags = selectedTags.length === 0 ||
            selectedTags.some(tag => (p.tags || []).includes(tag));
        
        return matchSearch && matchTags;
    });
}

function createColumn(type, filteredPrompts, node, state, onUpdate) {
    const column = document.createElement('div');
    const isPositive = type === 'positive';
    const columnPrompts = filteredPrompts.filter(p => isPositive ? p.type !== 'negative' : p.type === 'negative');
    
    const initialWidth = isPositive ? state.columnWidth : (100 - state.columnWidth);
    
    column.style.cssText = `
        flex: 0 0 ${initialWidth}%;
        background: #0d1f2d;
        border-radius: 8px;
        padding: 12px;
        display: flex;
        flex-direction: column;
        transition: flex 0.1s ease-out;
    `;
    
    const columnHeader = document.createElement('div');
    columnHeader.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding: 8px;
        background: ${isPositive ? 'rgba(46, 204, 113, 0.2)' : 'rgba(231, 76, 60, 0.2)'};
        border-radius: 6px;
    `;
    
    const columnTitle = document.createElement('h4');
    columnTitle.textContent = `${isPositive ? '✓ 正面' : '✗ 负面'} (${columnPrompts.length})`;
    columnTitle.style.cssText = `
        margin: 0;
        color: ${isPositive ? '#2ecc71' : '#e74c3c'};
        font-size: 14px;
        font-weight: bold;
    `;
    
    const collapseBtn = document.createElement('button');
    collapseBtn.textContent = state.collapsedColumns[type] ? '▶' : '▼';
    collapseBtn.style.cssText = `
        background: transparent;
        border: none;
        color: ${isPositive ? '#2ecc71' : '#e74c3c'};
        cursor: pointer;
        font-size: 14px;
        padding: 4px 8px;
    `;
    collapseBtn.onclick = () => {
        state.collapsedColumns[type] = !state.collapsedColumns[type];
        LocalStorage.setCollapsedColumns(state.collapsedColumns);
        onUpdate();
    };
    
    columnHeader.appendChild(columnTitle);
    columnHeader.appendChild(collapseBtn);
    column.appendChild(columnHeader);
    
    if (state.collapsedColumns[type]) {
        return column;
    }
    
    const listContainer = document.createElement('div');
    listContainer.style.cssText = `
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;
    
    if (columnPrompts.length === 0) {
        const emptyMsg = document.createElement('div');
        emptyMsg.textContent = '无Prompt';
        emptyMsg.style.cssText = `
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 13px;
        `;
        listContainer.appendChild(emptyMsg);
    } else {
        columnPrompts.forEach((prompt, index) => {
            const promptCard = createPromptCard(prompt, node, type, onUpdate, state);
            listContainer.appendChild(promptCard);
        });
    }
    
    column.appendChild(listContainer);
    return column;
}

function createPromptCard(prompt, node, type, onUpdate, state) {
    const card = document.createElement('div');
    card.draggable = true;
    card.style.cssText = `
        background: ${prompt.enabled ? (type === 'negative' ? '#2a1515' : '#152a3a') : '#1a2332'};
        border: 1px solid ${prompt.enabled ? (type === 'negative' ? '#e74c3c' : '#2ecc71') : '#2a5a8a'};
        border-radius: 6px;
        padding: 10px;
        cursor: move;
        transition: all 0.2s;
    `;
    
    card.ondragstart = (e) => {
        state.draggedItem = prompt;
        card.style.opacity = '0.5';
        e.dataTransfer.effectAllowed = 'move';
    };
    
    card.ondragend = () => {
        card.style.opacity = '1';
        state.draggedItem = null;
    };
    
    card.ondragover = (e) => {
        e.preventDefault();
        if (state.draggedItem && state.draggedItem !== prompt) {
            card.style.borderColor = '#3498db';
        }
    };
    
    card.ondragleave = () => {
        card.style.borderColor = prompt.enabled ? (type === 'negative' ? '#e74c3c' : '#2ecc71') : '#2a5a8a';
    };
    
    card.ondrop = (e) => {
        e.preventDefault();
        if (state.draggedItem && state.draggedItem !== prompt) {
            const draggedIndex = node.prompts.indexOf(state.draggedItem);
            const targetIndex = node.prompts.indexOf(prompt);
            
            node.prompts.splice(draggedIndex, 1);
            node.prompts.splice(targetIndex, 0, state.draggedItem);
            
            node.updateJSON();
            node.updateSimpleUI();
            onUpdate();
        }
        card.style.borderColor = prompt.enabled ? (type === 'negative' ? '#e74c3c' : '#2ecc71') : '#2a5a8a';
    };
    
    const topRow = document.createElement('div');
    topRow.style.cssText = `
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    `;
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = prompt.enabled;
    checkbox.style.cssText = `
        width: 16px;
        height: 16px;
        cursor: pointer;
    `;
    checkbox.onchange = (e) => {
        prompt.enabled = e.target.checked;
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    };
    
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = prompt.name;
    nameInput.style.cssText = `
        flex: 1;
        background: rgba(0,0,0,0.3);
        border: 1px solid #2a5a8a;
        border-radius: 4px;
        padding: 4px 8px;
        color: white;
        font-size: 12px;
    `;
    nameInput.onchange = (e) => {
        prompt.name = e.target.value;
        node.updateJSON();
        node.updateSimpleUI();
    };
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑';
    deleteBtn.style.cssText = `
        background: #e74c3c;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        padding: 4px 8px;
        font-size: 12px;
    `;
    deleteBtn.onclick = () => {
        const index = node.prompts.indexOf(prompt);
        if (index > -1) {
            node.prompts.splice(index, 1);
            node.updateJSON();
            node.updateSimpleUI();
            onUpdate();
        }
    };
    
    topRow.appendChild(checkbox);
    topRow.appendChild(nameInput);
    topRow.appendChild(deleteBtn);
    card.appendChild(topRow);
    
    const textArea = document.createElement('textarea');
    textArea.value = prompt.text;
    textArea.placeholder = '输入prompt内容...';
    textArea.style.cssText = `
        width: 100%;
        min-height: 60px;
        background: rgba(0,0,0,0.3);
        border: 1px solid #2a5a8a;
        border-radius: 4px;
        padding: 8px;
        color: white;
        font-size: 12px;
        resize: vertical;
        margin-bottom: 8px;
    `;
    textArea.onchange = (e) => {
        prompt.text = e.target.value;
        node.updateJSON();
        node.updateSimpleUI();
    };
    card.appendChild(textArea);
    
    const bottomRow = document.createElement('div');
    bottomRow.style.cssText = `
        display: flex;
        gap: 8px;
        align-items: center;
    `;
    
    const weightLabel = document.createElement('label');
    weightLabel.textContent = '权重:';
    weightLabel.style.cssText = 'color: #aaa; font-size: 11px;';
    
    const weightInput = document.createElement('input');
    weightInput.type = 'number';
    weightInput.value = prompt.weight;
    weightInput.min = '0';
    weightInput.max = '2';
    weightInput.step = '0.1';
    weightInput.style.cssText = `
        width: 60px;
        background: rgba(0,0,0,0.3);
        border: 1px solid #2a5a8a;
        border-radius: 4px;
        padding: 4px;
        color: white;
        font-size: 11px;
    `;
    weightInput.onchange = (e) => {
        prompt.weight = parseFloat(e.target.value) || 1.0;
        node.updateJSON();
        node.updateSimpleUI();
    };
    
    const tagsInput = document.createElement('input');
    tagsInput.type = 'text';
    tagsInput.value = (prompt.tags || []).join(', ');
    tagsInput.placeholder = '标签(逗号分隔)';
    tagsInput.style.cssText = `
        flex: 1;
        background: rgba(0,0,0,0.3);
        border: 1px solid #2a5a8a;
        border-radius: 4px;
        padding: 4px 8px;
        color: white;
        font-size: 11px;
    `;
    tagsInput.onchange = (e) => {
        prompt.tags = e.target.value.split(',').map(t => t.trim()).filter(t => t);
        node.updateJSON();
        node.updateSimpleUI();
        onUpdate();
    };
    
    bottomRow.appendChild(weightLabel);
    bottomRow.appendChild(weightInput);
    bottomRow.appendChild(tagsInput);
    card.appendChild(bottomRow);
    
    return card;
}

function createResizer(leftColumn, rightColumn, state, container) {
    const resizer = document.createElement('div');
    resizer.style.cssText = `
        width: 8px;
        cursor: col-resize;
        background: #2a5a8a;
        border-radius: 4px;
        transition: background 0.2s;
        flex-shrink: 0;
        position: relative;
        user-select: none;
    `;
    
    resizer.onmouseover = () => resizer.style.background = '#3498db';
    resizer.onmouseout = () => resizer.style.background = '#2a5a8a';
    
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    resizer.onmousedown = (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = state.columnWidth;
        e.preventDefault();
        
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';
    };
    
    const onMouseMove = (e) => {
        if (!isResizing) return;
        
        const containerRect = container.getBoundingClientRect();
        const deltaX = e.clientX - startX;
        const deltaPercent = (deltaX / containerRect.width) * 100;
        const newWidth = startWidth + deltaPercent;
        
        if (newWidth >= 20 && newWidth <= 80) {
            state.columnWidth = newWidth;
            LocalStorage.setColumnWidth(newWidth);
            
            leftColumn.style.transition = 'none';
            rightColumn.style.transition = 'none';
            leftColumn.style.flex = `0 0 ${newWidth}%`;
            rightColumn.style.flex = `0 0 ${100 - newWidth}%`;
        }
    };
    
    const onMouseUp = () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
            
            leftColumn.style.transition = 'flex 0.1s ease-out';
            rightColumn.style.transition = 'flex 0.1s ease-out';
        }
    };
    
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    
    return resizer;
}

function showHistoryManager(node, onUpdate) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 10001;
        display: flex;
        justify-content: center;
        align-items: center;
    `;
    
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        background: #1a2332;
        border-radius: 12px;
        width: 90%;
        max-width: 600px;
        padding: 20px;
        max-height: 80vh;
        overflow-y: auto;
    `;
    
    const title = document.createElement('h3');
    title.textContent = '📜 历史记录';
    title.style.cssText = 'color: white; margin-bottom: 16px;';
    dialog.appendChild(title);
    
    const clearBtn = document.createElement('button');
    clearBtn.textContent = '🗑️ 清空所有历史';
    clearBtn.style.cssText = `
        padding: 8px 16px;
        background: #e74c3c;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        margin-bottom: 16px;
    `;
    clearBtn.onclick = () => {
        if (confirm('确定要清空所有历史记录吗？')) {
            LocalStorage.clearHistory();
            refreshHistoryList();
        }
    };
    dialog.appendChild(clearBtn);
    
    const historyList = document.createElement('div');
    historyList.style.cssText = 'display: flex; flex-direction: column; gap: 8px;';
    dialog.appendChild(historyList);
    
    function refreshHistoryList() {
        historyList.innerHTML = '';
        const history = LocalStorage.getHistory();
        
        if (history.length === 0) {
            const emptyMsg = document.createElement('div');
            emptyMsg.textContent = '暂无历史记录';
            emptyMsg.style.cssText = 'text-align: center; padding: 20px; color: #666;';
            historyList.appendChild(emptyMsg);
            return;
        }
        
        history.forEach((entry) => {
            const item = document.createElement('div');
            item.style.cssText = `
                background: #0d1f2d;
                padding: 12px;
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            `;
            
            const info = document.createElement('div');
            info.style.cssText = 'color: white; flex: 1;';
            info.innerHTML = `
                <div style="font-weight: bold;">${entry.date}</div>
                <div style="font-size: 11px; color: #aaa;">${entry.prompts.length} 个Prompt</div>
            `;
            
            const btnGroup = document.createElement('div');
            btnGroup.style.cssText = 'display: flex; gap: 8px;';
            
            const loadBtn = document.createElement('button');
            loadBtn.textContent = '加载';
            loadBtn.style.cssText = `
                padding: 6px 12px;
                background: #3498db;
                border: none;
                border-radius: 4px;
                color: white;
                cursor: pointer;
            `;
            loadBtn.onclick = () => {
                node.prompts = entry.prompts.map(p => ({...p, id: p.id || generateId()}));
                node.updateJSON();
                node.updateSimpleUI();
                onUpdate();
                document.body.removeChild(overlay);
            };
            
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = '删除';
            deleteBtn.style.cssText = `
                padding: 6px 12px;
                background: #e74c3c;
                border: none;
                border-radius: 4px;
                color: white;
                cursor: pointer;
            `;
            deleteBtn.onclick = () => {
                LocalStorage.deleteHistory(entry.id);
                refreshHistoryList();
            };
            
            btnGroup.appendChild(loadBtn);
            btnGroup.appendChild(deleteBtn);
            
            item.appendChild(info);
            item.appendChild(btnGroup);
            historyList.appendChild(item);
        });
    }
    
    refreshHistoryList();
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = `
        margin-top: 16px;
        padding: 8px 16px;
        background: #95a5a6;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        width: 100%;
    `;
    closeBtn.onclick = () => document.body.removeChild(overlay);
    dialog.appendChild(closeBtn);
    
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
}

function showTemplateManager(node, onUpdate) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 10001;
        display: flex;
        justify-content: center;
        align-items: center;
    `;
    
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        background: #1a2332;
        border-radius: 12px;
        width: 90%;
        max-width: 600px;
        padding: 20px;
        max-height: 80vh;
        overflow-y: auto;
    `;
    
    const title = document.createElement('h3');
    title.textContent = '📋 模板管理';
    title.style.cssText = 'color: white; margin-bottom: 16px;';
    dialog.appendChild(title);
    
    const saveSection = document.createElement('div');
    saveSection.style.cssText = 'margin-bottom: 20px; display: flex; gap: 8px;';
    
    const saveInput = document.createElement('input');
    saveInput.type = 'text';
    saveInput.placeholder = '输入模板名称...';
    saveInput.style.cssText = `
        flex: 1;
        padding: 8px;
        background: #0d1f2d;
        border: 1px solid #2a5a8a;
        border-radius: 4px;
        color: white;
    `;
    
    const saveBtn = document.createElement('button');
    saveBtn.textContent = '保存';
    saveBtn.style.cssText = `
        padding: 8px 16px;
        background: #2ecc71;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
    `;
    saveBtn.onclick = () => {
        const name = saveInput.value.trim();
        if (name) {
            LocalStorage.saveTemplate(name, node.prompts);
            saveInput.value = '';
            refreshTemplateList();
        }
    };
    
    saveSection.appendChild(saveInput);
    saveSection.appendChild(saveBtn);
    dialog.appendChild(saveSection);
    
    const templateList = document.createElement('div');
    templateList.style.cssText = 'display: flex; flex-direction: column; gap: 8px;';
    dialog.appendChild(templateList);
    
    function refreshTemplateList() {
        templateList.innerHTML = '';
        const templates = LocalStorage.getTemplates();
        
        if (Object.keys(templates).length === 0) {
            const emptyMsg = document.createElement('div');
            emptyMsg.textContent = '暂无保存的模板';
            emptyMsg.style.cssText = 'text-align: center; padding: 20px; color: #666;';
            templateList.appendChild(emptyMsg);
            return;
        }
        
        Object.entries(templates).forEach(([name, template]) => {
            const item = document.createElement('div');
            item.style.cssText = `
                background: #0d1f2d;
                padding: 12px;
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            `;
            
            const info = document.createElement('div');
            info.style.cssText = 'color: white;';
            info.innerHTML = `
                <div style="font-weight: bold;">${name}</div>
                <div style="font-size: 11px; color: #aaa;">${template.prompts.length} 个Prompt</div>
            `;
            
            const btnGroup = document.createElement('div');
            btnGroup.style.cssText = 'display: flex; gap: 8px;';
            
            const loadBtn = document.createElement('button');
            loadBtn.textContent = '加载';
            loadBtn.style.cssText = `
                padding: 6px 12px;
                background: #3498db;
                border: none;
                border-radius: 4px;
                color: white;
                cursor: pointer;
            `;
            loadBtn.onclick = () => {
                node.prompts = template.prompts.map(p => ({...p, id: p.id || generateId()}));
                node.updateJSON();
                node.updateSimpleUI();
                onUpdate();
                document.body.removeChild(overlay);
            };
            
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = '删除';
            deleteBtn.style.cssText = `
                padding: 6px 12px;
                background: #e74c3c;
                border: none;
                border-radius: 4px;
                color: white;
                cursor: pointer;
            `;
            deleteBtn.onclick = () => {
                if (confirm(`确定删除模板"${name}"？`)) {
                    LocalStorage.deleteTemplate(name);
                    refreshTemplateList();
                }
            };
            
            btnGroup.appendChild(loadBtn);
            btnGroup.appendChild(deleteBtn);
            
            item.appendChild(info);
            item.appendChild(btnGroup);
            templateList.appendChild(item);
        });
    }
    
    refreshTemplateList();
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = `
        margin-top: 16px;
        padding: 8px 16px;
        background: #95a5a6;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        width: 100%;
    `;
    closeBtn.onclick = () => document.body.removeChild(overlay);
    dialog.appendChild(closeBtn);
    
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
}

// ==================== 注册扩展 ====================
app.registerExtension({
    name: "comfy_Pond.CustomPromptManagerEnhanced",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CustomPromptManagerEnhanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 尝试从localStorage加载保存的数据
                const nodeType = 'CustomPromptManagerEnhanced';
                const savedData = LocalStorage.getNodeData(nodeType);
                if (savedData && savedData.prompts) {
                    this.prompts = savedData.prompts.map(p => ({
                        ...p,
                        id: p.id || generateId()
                    }));
                    console.log(`[Prompt Manager Enhanced] 已自动加载保存的数据 (${this.prompts.length}项)`);
                } else {
                    this.prompts = [];
                }
                
                const container = document.createElement('div');
                container.style.cssText = `
                    background: #1a2332;
                    border: 1px solid #2a5a8a;
                    border-radius: 6px;
                    padding: 8px;
                    margin-top: 8px;
                    max-height: 300px;
                    overflow-y: auto;
                `;
                
                this.updateSimpleUI = () => {
                    container.innerHTML = '';
                    
                    const openEditorBtn = document.createElement('button');
                    openEditorBtn.textContent = '🐳Prompt编辑器';
                    openEditorBtn.style.cssText = `
                        width: 100%;
                        padding: 10px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: none;
                        border-radius: 6px;
                        color: white;
                        font-size: 13px;
                        font-weight: bold;
                        cursor: pointer;
                        margin-bottom: 10px;
                        transition: all 0.2s;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    `;
                    openEditorBtn.onmouseover = () => {
                        openEditorBtn.style.transform = 'translateY(-2px)';
                        openEditorBtn.style.boxShadow = '0 6px 12px rgba(0,0,0,0.4)';
                    };
                    openEditorBtn.onmouseout = () => {
                        openEditorBtn.style.transform = 'translateY(0)';
                        openEditorBtn.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
                    };
                    openEditorBtn.onclick = () => createModalEditor(this);
                    container.appendChild(openEditorBtn);
                    
                    const positivePrompts = this.prompts.filter(p => p.type !== 'negative');
                    const negativePrompts = this.prompts.filter(p => p.type === 'negative');
                    
                    const twoColumnContainer = document.createElement('div');
                    twoColumnContainer.style.cssText = `
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 8px;
                    `;
                    
                    const positiveColumn = document.createElement('div');
                    positiveColumn.style.cssText = `
                        background: rgba(46, 204, 113, 0.1);
                        border: 1px solid rgba(46, 204, 113, 0.3);
                        border-radius: 4px;
                        padding: 6px;
                    `;
                    
                    const posHeader = document.createElement('div');
                    posHeader.textContent = `✓ 正面 (${positivePrompts.length})`;
                    posHeader.style.cssText = `
                        color: #2ecc71;
                        font-weight: bold;
                        font-size: 10px;
                        margin-bottom: 4px;
                        padding: 4px;
                        background: rgba(46, 204, 113, 0.2);
                        border-radius: 3px;
                    `;
                    positiveColumn.appendChild(posHeader);
                    
                    if (positivePrompts.length === 0) {
                        const emptyMsg = document.createElement('div');
                        emptyMsg.textContent = '(无)';
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
                            const row = createSimpleRow(this, prompt, index, false);
                            positiveColumn.appendChild(row);
                        });
                    }
                    
                    const negativeColumn = document.createElement('div');
                    negativeColumn.style.cssText = `
                        background: rgba(231, 76, 60, 0.1);
                        border: 1px solid rgba(231, 76, 60, 0.3);
                        border-radius: 4px;
                        padding: 6px;
                    `;
                    
                    const negHeader = document.createElement('div');
                    negHeader.textContent = `✗ 负面 (${negativePrompts.length})`;
                    negHeader.style.cssText = `
                        color: #e74c3c;
                        font-weight: bold;
                        font-size: 10px;
                        margin-bottom: 4px;
                        padding: 4px;
                        background: rgba(231, 76, 60, 0.2);
                        border-radius: 3px;
                    `;
                    negativeColumn.appendChild(negHeader);
                    
                    if (negativePrompts.length === 0) {
                        const emptyMsg = document.createElement('div');
                        emptyMsg.textContent = '(无)';
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
                            const row = createSimpleRow(this, prompt, index, true);
                            negativeColumn.appendChild(row);
                        });
                    }
                    
                    twoColumnContainer.appendChild(positiveColumn);
                    twoColumnContainer.appendChild(negativeColumn);
                    container.appendChild(twoColumnContainer);
                };
                
                const createSimpleRow = (node, prompt, index, isNegative) => {
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
                    console.log('[Prompt Manager] Updating JSON:', jsonValue);
                    if (this.widgets) {
                        const jsonWidget = this.widgets.find(w => w.name === "prompts_json");
                        if (jsonWidget) {
                            jsonWidget.value = jsonValue;
                            console.log('[Prompt Manager] Widget updated with:', jsonValue);
                        } else {
                            console.warn('[Prompt Manager] prompts_json widget not found!');
                        }
                    }
                };
                
                this.addDOMWidget("custom_ui", "customWidget", container);
                
                this.updateSimpleUI();
                this.updateJSON();
                
                const maxPromptsInColumn = Math.max(
                    this.prompts.filter(p => p.type !== 'negative').length,
                    this.prompts.filter(p => p.type === 'negative').length,
                    1
                );
                const height = Math.max(200, maxPromptsInColumn * 22 + 100);
                this.setSize([500, height]);
                
                return r;
            };
            
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                options.unshift(
                    {
                        content: "🐳Prompt编辑器",
                        callback: () => createModalEditor(this)
                    },
                    null
                );
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

console.log("✅ Custom Prompt Manager Enhanced loaded");
